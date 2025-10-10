from tqdm import tqdm
import argparse
import h5py
import heapq
import numpy as np
import os
import sqlite3
import time
import xxhash
from dataclasses import dataclass
from litestar import Litestar, get, post
from litestar.response import Response
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Tuple, Iterator
from litestar.datastructures import State
import bm25s
import Stemmer  # type:ignore

SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "target",
    "build",
    "dist",
}
SKIP_EXTS = {
    ".pyc",
    ".pyo",
    ".webp",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
}


@dataclass
class AppState:
    model: SentenceTransformer
    cross_encoder: CrossEncoder
    conn: sqlite3.Connection
    embeddings: np.ndarray
    file_cache: Dict[str, str]
    stemmer: Stemmer
    bm25_retriever: bm25s.BM25
    corpus: List


def create_app_state() -> AppState:
    # Get model name from embeddings file
    with h5py.File("embeddings.h5", "r") as f:
        model_name = f.attrs.get("model")
        if not model_name:
            raise AttributeError("couldnt find model name in embeddings file!")
        embeddings = f["embeddings"][:]

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Loading cross-encoder...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Open persistent DB connection
    conn = sqlite3.connect("index.db", check_same_thread=False)

    rows = conn.execute(
        """
        SELECT c.id, c.chunk_text
        FROM chunks c
        """,
    ).fetchall()

    corpus = [row[1] for row in sorted(rows, key=lambda x: x[0])]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)

    print("Server initialized")

    return AppState(
        model=model,
        cross_encoder=cross_encoder,
        embeddings=embeddings,
        conn=conn,
        file_cache={},
        stemmer=stemmer,
        bm25_retriever=bm25_retriever,
        corpus=corpus,
    )


def file_content_from_cache(state: AppState, filepath: str) -> str:
    if filepath not in state.file_cache:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                state.file_cache[filepath] = f.read()
        except Exception:
            state.file_cache[filepath] = ""
    else:
        print(f"Found in cache {filepath}")
    return state.file_cache[filepath]


@dataclass
class SearchResult:
    rank: int
    bi_encoder_rank: int | None
    bi_encoder_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    cross_encoder_score: float
    filename: str
    file_id: int
    chunk_content: str
    embedded_chunk_text: str
    start_byte: int
    end_byte: int


@dataclass
class FileInfo:
    path: str
    mtime: float
    hash: str


@dataclass
class Chunk:
    file_id: int
    start_byte: int
    end_byte: int
    text: str
    hash: str


def hash_content(content: str) -> str:
    return xxhash.xxh64(content.encode()).hexdigest()[:16]


def should_skip_file(path: Path) -> bool:
    return path.suffix.lower() in SKIP_EXTS or path.name.startswith(".")


def iter_files(directory: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            path = Path(root) / file
            if not should_skip_file(path):
                yield path


@dataclass
class ChunkIndices:
    start: int
    end: int


def merge_chunks(buncha_chunks: List[ChunkIndices]) -> ChunkIndices:
    start = min([c.start for c in buncha_chunks])
    end = max([c.end for c in buncha_chunks])

    return ChunkIndices(start=start, end=end)


def chunk_text(text: str, max_tokens: int) -> List[ChunkIndices]:
    paragraph_chunks: List[ChunkIndices] = []
    start = 0

    while start < len(text):
        end = start + 1

        while end < len(text) and text[end : end + 2] != "\n\n":
            end += 1

        paragraph_chunks.append(ChunkIndices(start, end))

        start = end

    final_chunks: List[ChunkIndices] = []

    for num_paras in [8]:
        step = num_paras // 2
        for idx in range(0, len(paragraph_chunks), step):
            chunks = paragraph_chunks[idx : idx + num_paras]
            final_chunks.append(merge_chunks(chunks))

    return final_chunks


def process_file(
    path: Path, max_tokens: int, file_id: int
) -> Tuple[FileInfo, List[Chunk]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    file_hash = hash_content(content)
    file_info = FileInfo(str(path), path.stat().st_mtime, file_hash)

    chunk_ranges: List[ChunkIndices] = chunk_text(content, max_tokens)
    chunks = []

    for chunk_idx, chunk_range in enumerate(chunk_ranges):
        (start_byte, end_byte) = (chunk_range.start, chunk_range.end)
        text = content[start_byte:end_byte]

        embed_text = f"File: {path.name}\n\n{text}"

        chunk = Chunk(
            file_id=file_id,
            start_byte=max(0, start_byte),
            end_byte=end_byte,
            text=embed_text,
            hash=hash_content(embed_text),
        )
        chunks.append(chunk)

    return file_info, chunks


def create_index(
    index_filename: str, directory: str, model_name: str, batch_size: int
) -> None:
    model = SentenceTransformer(model_name)
    max_tokens = model.get_max_seq_length()
    if max_tokens is None:
        raise AttributeError("idk max tokens!")

    print(f"Model token size is {max_tokens}")

    # Collect all files and chunks
    all_files: List[FileInfo] = []
    all_chunks: List[Chunk] = []

    print(f"Processing files in {directory}...")
    for file_idx, file_path in enumerate(iter_files(Path(directory))):
        try:
            file_info, chunks = process_file(file_path, max_tokens, file_idx + 1)
            print(f"File {file_path.name} with {len(chunks)} chunks")
            all_files.append(file_info)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    all_chunks = list(sorted(all_chunks, key=lambda chunk: len(chunk.text)))

    print(f"Found {len(all_files)} files, {len(all_chunks)} chunks")

    # Create embeddings in batches
    embeddings = []
    texts = [chunk.text for chunk in all_chunks]

    embedding_start_time = time.time()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        if hasattr(model, "encode_documents"):
            assert callable(model.encode_documents)
            batch_embeddings = model.encode_documents(batch)
        else:
            batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        # print(
        #     f"Embedded batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
        # )

    # Save embeddings to HDF5
    embeddings_array = np.array(embeddings, dtype=np.float32)

    embedding_total_time = time.time() - embedding_start_time
    embedding_size = embeddings_array.nbytes

    print(
        f"Embeddings: {embedding_size / (1024.0 * 1024):.2f} MiB in {embedding_total_time: .1f}s"
    )
    print(f"Total size of embedding matrix is {embeddings_array.shape}")
    with h5py.File("embeddings.h5", "w") as f:
        f.create_dataset("embeddings", data=embeddings_array, compression="gzip")
        f.attrs["model"] = model_name
        f.attrs["num_chunks"] = len(all_chunks)

    # Save metadata to SQLite
    conn = sqlite3.connect(index_filename)
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            mtime REAL NOT NULL,
            hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            start_byte INTEGER NOT NULL,
            end_byte INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            matrix_index INTEGER NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files (id)
        )
    """)
    conn.execute("""
        CREATE INDEX idx_chunks_matrix_index ON chunks(matrix_index);
    """)

    # Insert data
    for i, file_info in enumerate(all_files):
        conn.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?)",
            (i + 1, file_info.path, file_info.mtime, file_info.hash),
        )

    for i, chunk in enumerate(all_chunks):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                i + 1,
                chunk.file_id,
                chunk.start_byte,
                chunk.end_byte,
                chunk.text,
                chunk.hash,
                i,
            ),
        )

    conn.commit()
    conn.close()
    db_size = os.path.getsize(index_filename)
    print(
        f"Created index with {len(embeddings)} embeddings of size {db_size / (1024.0 * 1024):.1f} MiB"
    )


@dataclass
class BM25Result:
    chunk_id: int
    rank: int
    score: float
    filename: str
    file_id: int
    start_byte: int
    end_byte: int


def search_bm25(state: AppState, query: str, k: int) -> List[BM25Result]:
    bm25_query_tokens = bm25s.tokenize(query, stemmer=state.stemmer)
    bm25_results, bm25_scores = state.bm25_retriever.retrieve(bm25_query_tokens, k=k)

    chunk_ids = [int(bm25_results[0, i]) + 1 for i in range(bm25_results.shape[1])]

    placeholders = ",".join("?" * len(chunk_ids))
    rows = state.conn.execute(
        f"""
        SELECT c.id, f.filename, c.start_byte, c.end_byte, f.id
        FROM chunks c JOIN files f on c.file_id = f.id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    chunk_id2row = {row[0]: row for row in rows}

    answer = []
    for i, chunk_id in enumerate(chunk_ids):
        _, filename, start_byte, end_byte, file_id = chunk_id2row[chunk_id]
        answer.append(
            BM25Result(
                chunk_id=chunk_id,
                rank=i + 1,
                score=float(bm25_scores[0, i]),
                filename=filename,
                start_byte=start_byte,
                end_byte=end_byte,
                file_id=file_id,
            )
        )

    return answer


@dataclass
class EmbeddingResult:
    chunk_id: int
    rank: int
    score: float
    filename: str
    file_id: int
    start_byte: int
    end_byte: int


def search_embeddings(state: AppState, query: str, k: int) -> List[EmbeddingResult]:
    if hasattr(state.model, "encode_queries"):
        assert callable(state.model.encode_queries)
        query_embedding = state.model.encode_queries([query])[0]
    else:
        query_embedding = state.model.encode([query], prompt_name="query")[0]
    # Compute similarities
    similarities = np.dot(state.embeddings, query_embedding)

    top_indices = heapq.nlargest(
        k, range(len(similarities)), key=similarities.__getitem__
    )

    placeholders = ",".join("?" * len(top_indices))
    rows = state.conn.execute(
        f"""
        SELECT c.matrix_index, f.filename, c.start_byte, c.end_byte, c.chunk_text, f.id, c.id
        FROM chunks c JOIN files f ON c.file_id = f.id
        WHERE c.matrix_index IN ({placeholders})
    """,
        top_indices,
    ).fetchall()
    index_to_row = {row[0]: row for row in rows}

    answer = []
    for bi_encoder_rank, idx in enumerate(top_indices, 1):
        if idx not in index_to_row:
            continue

        _, filename, start_byte, end_byte, chunk_text, file_id, chunk_id = index_to_row[
            idx
        ]

        # content = file_content_from_cache(state, filename)
        # chunk_content = content[start_byte:end_byte]
        answer.append(
            EmbeddingResult(
                chunk_id=chunk_id,
                rank=bi_encoder_rank,
                score=float(similarities[idx]),
                filename=filename,
                file_id=file_id,
                start_byte=start_byte,
                end_byte=end_byte,
            )
        )

    return answer


def search_combined(state: AppState, query: str, k: int) -> List[SearchResult]:
    total_search_start_time = time.time()

    embedding_results = search_embeddings(state, query, k)
    bm25_results = search_bm25(state, query, k)

    seen = set()

    @dataclass
    class QP:
        chunk_id: int
        query: str
        passage: str
        filename: str
        file_id: int
        chunk_content: str
        start_byte: int
        end_byte: int

    for_cross_encoder: List[QP] = []
    for result in embedding_results + bm25_results:
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)

        content = file_content_from_cache(state, result.filename)
        chunk_content = content[result.start_byte : result.end_byte]
        passage = f"File: {result.filename}. {chunk_content}"
        for_cross_encoder.append(
            QP(
                chunk_id=result.chunk_id,
                query=query,
                passage=passage,
                filename=result.filename,
                file_id=result.file_id,
                start_byte=result.start_byte,
                end_byte=result.end_byte,
                chunk_content=chunk_content,
            )
        )

    cross_encoding_start_time = time.time()
    cross_scores = state.cross_encoder.predict(
        [(x.query, x.passage) for x in for_cross_encoder]
    )
    cross_encoding_end_time = time.time()

    @dataclass
    class QS:
        qp: QP
        cross_score: float

    qs: List[QS] = [
        QS(qp=qp, cross_score=float(cross_score))
        for (qp, cross_score) in zip(for_cross_encoder, cross_scores)
    ]
    qs.sort(key=lambda x: x.cross_score, reverse=True)

    chunk_id2er = {er.chunk_id: er for er in embedding_results}
    chunk_id2br = {br.chunk_id: br for br in bm25_results}

    answer = []
    for idx, q in enumerate(qs):
        result = SearchResult(
            rank=idx + 1,
            bi_encoder_rank=None,
            bi_encoder_score=None,
            bm25_rank=None,
            bm25_score=None,
            cross_encoder_score=q.cross_score,
            filename=Path(q.qp.filename).name,
            file_id=q.qp.file_id,
            chunk_content=q.qp.chunk_content,
            embedded_chunk_text="",
            start_byte=q.qp.start_byte,
            end_byte=q.qp.end_byte,
        )

        if q.qp.chunk_id in chunk_id2er:
            er = chunk_id2er[q.qp.chunk_id]
            result.bi_encoder_rank = er.rank
            result.bi_encoder_score = er.score

        if q.qp.chunk_id in chunk_id2br:
            br = chunk_id2br[q.qp.chunk_id]
            result.bm25_rank = br.rank
            result.bm25_score = br.score

        answer.append(result)

    total_search_end_time = time.time()

    timings = {
        "cross_encode": cross_encoding_end_time - cross_encoding_start_time,
        "total": total_search_end_time - total_search_start_time,
    }
    for label, duration in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {duration:.2f}s")

    return answer


@get("/")
async def serve_index() -> Response:
    with open("index.html", "r") as f:
        content = f.read()
    return Response(content=content, media_type="text/html")


@post("/search")
async def search(data: Dict[str, Any], state: State) -> Dict[str, Any]:
    query = data.get("query", "").strip()
    if not query:
        return {"results": []}

    top_k = data.get("top_k")
    if top_k is None:
        raise KeyError("didn't find top_k")
    top_k = int(top_k.strip())

    results = search_combined(state.inner, query, top_k)

    return {
        "results": [
            {
                "rank": r.rank,
                "bi_encoder_score": r.bi_encoder_score,
                "bi_encoder_rank": r.bi_encoder_rank,
                "bm25_rank": r.bm25_rank,
                "bm25_score": r.bm25_score,
                "cross_encoder_score": r.cross_encoder_score,
                "filename": r.filename,
                "chunk_content": r.chunk_content,
                "embedded_chunk_text": r.embedded_chunk_text,
                "start_byte": r.start_byte,
                "end_byte": r.end_byte,
                "file_id": r.file_id,
            }
            for r in results
        ]
    }


@get("/file/{file_id:int}")
async def get_file_content(file_id: int, state: State) -> Dict[str, Any]:
    try:
        # Find full path from database
        row = state.inner.conn.execute(
            "SELECT filename FROM files WHERE id = ?", (f"{file_id}",)
        ).fetchone()

        if not row:
            return {"error": "File not found"}

        full_path = row[0]
        content = file_content_from_cache(state.inner, full_path)

        return {"content": content}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "search"])
    parser.add_argument("--directory", "-d", default=".", help="Directory to index")
    parser.add_argument("--index-db", "-i", default="index.db", help="Index db file")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2", help="Model name")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=128, help="Batch size for embedding"
    )
    parser.add_argument("--query", "-q", help="Search query")

    args = parser.parse_args()

    if args.command == "index":
        create_index(args.index_db, args.directory, args.model, args.batch_size)
    elif args.command == "search":
        import uvicorn

        app = Litestar(
            route_handlers=[serve_index, search, get_file_content],
            debug=True,
        )
        app.state.inner = create_app_state()
        uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
