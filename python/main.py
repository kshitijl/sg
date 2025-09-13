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
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Iterator

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

# Global model (loaded once)
model = None
conn = None
file_cache = {}


def init_server():
    global model, conn

    # Get model name from embeddings file
    with h5py.File("embeddings.h5", "r") as f:
        model_name = f.attrs.get("model")
        if not model_name:
            raise AttributeError("couldnt find model name in embeddings file!")

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Open persistent DB connection
    conn = sqlite3.connect("index.db", check_same_thread=False)
    print("Server initialized")


def file_content_from_cache(filepath: str) -> str:
    if filepath not in file_cache:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_cache[filepath] = f.read()
        except Exception:
            file_cache[filepath] = ""
    else:
        print(f"Found in cache {filepath}")
    return file_cache[filepath]


@dataclass
class SearchResult:
    rank: int
    similarity: float
    filename: str
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

    for chunk_range in chunk_ranges:
        (start_byte, end_byte) = chunk_range
        text = content[start_byte:end_byte]
        chunk = Chunk(
            file_id=file_id,  # Will be set later
            start_byte=max(0, start_byte),
            end_byte=end_byte,
            text=text,
            hash=hash_content(text),
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


def search_embeddings(query: str, k: int = 50) -> List[SearchResult]:
    global model, conn
    assert model is not None
    assert conn is not None

    # Embed query
    if hasattr(model, "encode_queries") and callable(getattr(model, "encode_queries")):
        query_embedding = model.encode_queries([query])[0]
    else:
        query_embedding = model.encode([query])[0]

    # Load embeddings
    with h5py.File("embeddings.h5", "r") as f:
        embeddings = f["embeddings"][:]

    # Compute similarities
    similarities = np.dot(embeddings, query_embedding)
    top_indices = heapq.nlargest(
        k, range(len(similarities)), key=similarities.__getitem__
    )

    # Get chunk metadata
    placeholders = ",".join("?" * len(top_indices))
    rows = conn.execute(
        f"""
        SELECT c.matrix_index, f.filename, c.start_byte, c.end_byte
        FROM chunks c JOIN files f ON c.file_id = f.id
        WHERE c.matrix_index IN ({placeholders})
    """,
        top_indices,
    ).fetchall()

    # Build results with file content
    index_to_row = {row[0]: row for row in rows}
    results = []

    for rank, idx in enumerate(top_indices, 1):
        if idx not in index_to_row:
            continue

        _, filename, start_byte, end_byte = index_to_row[idx]

        file_content = file_content_from_cache(filename)
        chunk_content = file_content[start_byte:end_byte]

        results.append(
            SearchResult(
                rank=rank,
                similarity=float(similarities[idx]),
                filename=Path(filename).name,
                chunk_content=chunk_content,
                start_byte=start_byte,
                end_byte=end_byte,
            )
        )

    return results


@get("/")
async def serve_index() -> Response:
    with open("index.html", "r") as f:
        content = f.read()
    return Response(content=content, media_type="text/html")


@post("/search")
async def search(data: Dict[str, Any]) -> Dict[str, Any]:
    query = data.get("query", "").strip()
    if not query:
        return {"results": []}

    results = search_embeddings(query)

    return {
        "results": [
            {
                "rank": r.rank,
                "similarity": r.similarity,
                "filename": r.filename,
                "chunk_content": r.chunk_content,
                "start_byte": r.start_byte,
                "end_byte": r.end_byte,
            }
            for r in results
        ]
    }


@get("/file/{filename:str}")
async def get_file_content(
    filename: str, start_byte: int, end_byte: int
) -> Dict[str, Any]:
    try:
        # Find full path from database
        conn = sqlite3.connect("index.db")
        row = conn.execute(
            "SELECT filename FROM files WHERE filename LIKE ?", (f"%{filename}",)
        ).fetchone()
        conn.close()

        if not row:
            return {"error": "File not found"}

        full_path = row[0]
        content = file_content_from_cache(full_path)

        return {"content": content, "start_byte": start_byte, "end_byte": end_byte}
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

        init_server()

        uvicorn.run(app, host="127.0.0.1", port=8000)


app = Litestar(route_handlers=[serve_index, search, get_file_content], debug=True)

if __name__ == "__main__":
    main()
