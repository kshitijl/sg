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
    print("Server initialized")

    return AppState(
        model=model,
        cross_encoder=cross_encoder,
        embeddings=embeddings,
        conn=conn,
        file_cache={},
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
    similarity: float
    filename: str
    file_id: int
    chunk_content: str
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


def chunk_text(text: str, max_tokens: int) -> List[Tuple[int, int]]:
    # Rough estimate: 4 chars = 1 token
    max_chars = int(max_tokens * 4 * 0.8)  # 80% of max size

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append((start, len(text)))
            break

        # Find last word boundary
        while end > start and not text[end].isspace():
            end -= 1

        # If no space found, just cut at max_chars
        if end == start:
            end = start + max_chars

        chunks.append((start, end))
        start = end

        # Skip whitespace
        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def process_file(
    path: Path, max_tokens: int, file_id: int
) -> Tuple[FileInfo, List[Chunk]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    file_hash = hash_content(content)
    file_info = FileInfo(str(path), path.stat().st_mtime, file_hash)

    chunk_ranges = chunk_text(content, max_tokens)
    chunks = []

    for chunk_idx, chunk_range in enumerate(chunk_ranges):
        (start_byte, end_byte) = chunk_range
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

    print(f"Found {len(all_files)} files, {len(all_chunks)} chunks")

    # Create embeddings in batches
    embeddings = []
    texts = [chunk.text for chunk in all_chunks]

    embedding_start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if hasattr(model, "encode_documents"):
            assert callable(model.encode_documents)
            batch_embeddings = model.encode_documents(batch)
        else:
            batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        print(
            f"Embedded batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
        )

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
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            (i + 1, chunk.file_id, chunk.start_byte, chunk.end_byte, chunk.hash, i),
        )

    conn.commit()
    conn.close()
    db_size = os.path.getsize(index_filename)
    print(
        f"Created index with {len(embeddings)} embeddings of size {db_size / (1024.0 * 1024):.1f} MiB"
    )


def search_embeddings(state: AppState, query: str, k: int = 50) -> List[SearchResult]:
    total_search_start_time = time.time()
    embed_start_time = time.time()
    if hasattr(state.model, "encode_queries"):
        assert callable(state.model.encode_queries)
        query_embedding = state.model.encode_queries([query])[0]
    else:
        query_embedding = state.model.encode([query])[0]
    embed_end_time = time.time()

    search_start_time = time.time()
    # Compute similarities
    similarities = np.dot(state.embeddings, query_embedding)
    search_end_time = time.time()

    sort_start_time = time.time()
    top_indices = heapq.nlargest(
        k, range(len(similarities)), key=similarities.__getitem__
    )
    sort_end_time = time.time()

    chunk_retrieval_start_time = time.time()
    placeholders = ",".join("?" * len(top_indices))
    rows = state.conn.execute(
        f"""
        SELECT c.matrix_index, f.filename, c.start_byte, c.end_byte, f.id
        FROM chunks c JOIN files f ON c.file_id = f.id
        WHERE c.matrix_index IN ({placeholders})
    """,
        top_indices,
    ).fetchall()
    chunk_retrieval_end_time = time.time()

    build_results_start_time = time.time()
    index_to_row = {row[0]: row for row in rows}

    candidates = []
    for idx in top_indices:
        if idx not in index_to_row:
            continue

        _, filename, start_byte, end_byte, file_id = index_to_row[idx]

        try:
            content = file_content_from_cache(state, filename)
            chunk_content = content[start_byte:end_byte]
            candidates.append(
                {
                    "idx": idx,
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_content": chunk_content,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "bi_encoder_score": float(similarities[idx]),
                }
            )
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    build_results_end_time = time.time()

    if candidates:
        query_passage_pairs = []
        for c in candidates:
            cross_encoder_input = f"File: {c['filename']}. {c['chunk_content']}"
            query_passage_pairs.append([query, cross_encoder_input])
        cross_encoding_start_time = time.time()
        cross_scores = state.cross_encoder.predict(query_passage_pairs)
        cross_encoding_end_time = time.time()

        for i, candidate in enumerate(candidates):
            candidate["cross_score"] = float(cross_scores[i])

        candidates.sort(key=lambda x: x["cross_score"], reverse=True)

    results = []

    for rank, candidate in enumerate(candidates, 1):
        results.append(
            SearchResult(
                rank=rank,
                similarity=candidate["cross_score"],
                file_id=candidate["file_id"],
                filename=Path(candidate["filename"]).name,
                chunk_content=candidate["chunk_content"],
                start_byte=candidate["start_byte"],
                end_byte=candidate["end_byte"],
            )
        )

    total_search_end_time = time.time()

    timings = {
        "embed": embed_end_time - embed_start_time,
        "search": search_end_time - search_start_time,
        "sort": sort_end_time - sort_start_time,
        "chunk_retrieval": chunk_retrieval_end_time - chunk_retrieval_start_time,
        "build_results": build_results_end_time - build_results_start_time,
        "cross_encode": cross_encoding_end_time - cross_encoding_start_time,
        "total": total_search_end_time - total_search_start_time,
    }
    for label, duration in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {duration:.2f}s")

    return results


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

    results = search_embeddings(state.inner, query)

    return {
        "results": [
            {
                "rank": r.rank,
                "similarity": r.similarity,
                "filename": r.filename,
                "chunk_content": r.chunk_content,
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
    parser.add_argument("--top-k", "-k", type=int, default=50, help="Number of results")

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
