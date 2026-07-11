import json
from pathlib import Path

import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from app.config import (
    S3_BUCKET, AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION, EMBEDDING_MODEL_NAME
)
from app.classify import get_source_type, get_path_type, chunk_id, chunk_context_header
import boto3

# ── Setup ─────────────────────────────────────────────
parser = Parser(Language(tspython.language()))

enc = tiktoken.get_encoding("cl100k_base")
# model = SentenceTransformer("nomic-ai/nomic-embed-code")
model = SentenceTransformer(EMBEDDING_MODEL_NAME,
                            trust_remote_code=True,
                            device="cuda" if torch.cuda.is_available() else "cpu")

REPOS_DIR = Path("data/repos")
MANIFESTS_DIR = Path("data/manifests")
CHROMA_DIR = Path("data/chroma")
STATE_DIR = Path("data/index_state")

MAX_CHUNK_TOKENS = 512

# Bump whenever chunking, metadata, or embedding logic changes in a way that
# makes previously-indexed chunks stale (e.g. adding source_type/path_type).
# v3: chunk_context_header() prepended to function/function_part content --
# changes what gets embedded, not just metadata, so old chunks are stale.
PIPELINE_VERSION = 3

# ── Token counting ────────────────────────────────────
def estimate_tokens(text: str) -> int:
    return len(enc.encode(text))


# ── Fixed token splitter (fallback) ──────────────────
def fixed_token_split(text: str, max_tokens: int, overlap: int = 64) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks


# ── AST parsing ───────────────────────────────────────
def extract_functions(source_code: str) -> list[dict]:
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
    except Exception:
        return []

    root = tree.root_node
    functions = []

    def _leading_docstring(body_node) -> str:
        if body_node and body_node.child_count > 0:
            first = body_node.children[0]
            if first.type == "expression_statement" and first.children:
                inner = first.children[0]
                if inner.type == "string":
                    return inner.text.decode().strip("\"' ")
        return ""

    def walk(node, class_name=None, class_docstring=""):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = name_node.text.decode()
            # Scoped to this class specifically -- used to prime method
            # chunks with vocabulary that otherwise only lives in the class
            # docstring / __init__'s Doc() annotations (see
            # chunk_context_header's docstring for why this matters).
            class_docstring = _leading_docstring(node.child_by_field_name("body"))

        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode() if name_node else "unknown"
            docstring = _leading_docstring(node.child_by_field_name("body"))

            functions.append({
                "name": func_name,
                "class": class_name,
                "class_docstring": class_docstring,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": node.text.decode(),
                "docstring": docstring
            })

        for child in node.children:
            walk(child, class_name, class_docstring)

    walk(root)
    return functions


# ── Chunk builder ─────────────────────────────────────
def build_chunk(content: str, file_info: dict, chunk_type: str,
                func_name: str = None, class_name: str = None,
                start_line: int = None, part_index: int = None) -> dict:
    return {
        "content": content,
        "metadata": {
            "repo": file_info["repo"],
            "rel_path": file_info["rel_path"],
            "filename": file_info["filename"],
            "extension": file_info["extension"],
            "source_type": get_source_type(file_info["extension"]),
            "path_type": get_path_type(file_info["rel_path"]),
            "func_name": func_name or "",
            "class_name": class_name or "",
            "start_line": start_line or 0,
            "chunk_type": chunk_type,
            "part_index": part_index or 0,
            "char_count": len(content)
        }
    }


# ── Chunking ──────────────────────────────────────────
def chunk_file(file_info: dict, source_code: str) -> list[dict]:
    ext = file_info["extension"]
    chunks = []

    if ext == ".py":
        functions = extract_functions(source_code)

        if not functions:
            # No functions found, fall back to fixed split
            for i, text in enumerate(fixed_token_split(source_code, MAX_CHUNK_TOKENS)):
                chunks.append(build_chunk(
                    content=text, file_info=file_info,
                    chunk_type="text_block", part_index=i
                ))
        else:
            for func in functions:
                code = func["code"]
                # Prepended to every function/function_part chunk's content
                # (embedded and stored, not just metadata) -- see
                # chunk_context_header's docstring for the two real
                # retrieval failures this fixes.
                header = chunk_context_header(func["name"], func["class"], func["class_docstring"])
                if estimate_tokens(code) > MAX_CHUNK_TOKENS:
                    sub_chunks = fixed_token_split(code, MAX_CHUNK_TOKENS)
                    for i, sub in enumerate(sub_chunks):
                        chunks.append(build_chunk(
                            content=header + sub, file_info=file_info,
                            func_name=func["name"], class_name=func["class"],
                            start_line=func["start_line"],
                            chunk_type="function_part", part_index=i
                        ))
                else:
                    chunks.append(build_chunk(
                        content=header + code, file_info=file_info,
                        func_name=func["name"], class_name=func["class"],
                        start_line=func["start_line"],
                        chunk_type="function"
                    ))
    else:
        for i, text in enumerate(fixed_token_split(source_code, MAX_CHUNK_TOKENS)):
            chunks.append(build_chunk(
                content=text, file_info=file_info,
                chunk_type="text_block", part_index=i
            ))

    return chunks


# ── Embedding ─────────────────────────────────────────
def embed_chunks(chunks: list[dict], batch_size: int = 128) -> list[dict]:
    print(f"Embedding {len(chunks)} chunks...")
    contents = [f"search_document: {c['content']}" for c in chunks]

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        for j, embedding in enumerate(embeddings):
            chunks[i + j]["embedding"] = embedding

        print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")

    return chunks

# TODO: Switch to OpenAI embeddings once the new API is available for local use -- 
# (simply uncomment the below and comment the above function)

# from openai import OpenAI   
# import os
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# client = OpenAI(api_key=OPENAI_API_KEY)

# def embed_chunks(chunks: list[dict]) -> list[dict]:
#     contents = [c["content"] for c in chunks]
#     response = client.embeddings.create(
#         input=contents,
#         model="text-embedding-3-small"
#     )
#     for chunk, item in zip(chunks, response.data):
#         chunk["embedding"] = item.embedding
#     return chunks

# ── Chroma storage ────────────────────────────────────
def store_chunks(chunks: list[dict], repo_id: str):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR / repo_id))
    collection = client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
    )

    print(f"Storing {len(chunks)} chunks in Chroma...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[chunk_id(c["metadata"]) for c in batch],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch]
        )
        print(f"  Stored {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Chroma DB saved to: {CHROMA_DIR / repo_id}")


# ── S3 snapshot ───────────────────────────────────────
def upload_chroma_to_s3(repo_id: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

    chroma_path = CHROMA_DIR / repo_id
    for file in chroma_path.rglob("*"):
        if file.is_file():
            s3_key = f"vector_stores/{repo_id}/{file.relative_to(chroma_path)}"
            s3_key = s3_key.replace("\\", "/")
            s3.upload_file(str(file), S3_BUCKET, s3_key)

    print(f"Chroma snapshot uploaded to s3://{S3_BUCKET}/vector_stores/{repo_id}/")


# ── Change detection (skip re-encoding when nothing changed) ──
def _state_path(repo_id: str) -> Path:
    return STATE_DIR / f"{repo_id}.json"


def needs_reindex(repo_id: str, manifest: dict) -> bool:
    state_path = _state_path(repo_id)
    if not state_path.exists():
        return True

    state = json.loads(state_path.read_text())
    return (
        state.get("commit_sha") != manifest["commit_sha"]
        or state.get("pipeline_version") != PIPELINE_VERSION
        or state.get("embedding_model") != EMBEDDING_MODEL_NAME
    )


def mark_indexed(repo_id: str, manifest: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(repo_id).write_text(json.dumps({
        "commit_sha": manifest["commit_sha"],
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }, indent=2))


# ── Full pipeline ─────────────────────────────────────
def run_pipeline(repo_id: str):
    manifest_path = MANIFESTS_DIR / f"{repo_id}.json"
    manifest = json.loads(manifest_path.read_text())
    repo_path = REPOS_DIR / repo_id

    print(f"Processing {manifest['total_files']} files...")
    all_chunks = []

    for i, file_info in enumerate(manifest["files"]):
        file_info["repo"] = manifest["repo"]
        abs_path = repo_path / file_info["rel_path"]

        try:
            source_code = abs_path.read_text(errors="ignore")
        except Exception as e:
            print(f"  Skipping {file_info['rel_path']}: {e}")
            continue

        chunks = chunk_file(file_info, source_code)
        all_chunks.extend(chunks)

        if (i + 1) % 100 == 0:
            print(f"  Chunked {i + 1}/{manifest['total_files']} files — {len(all_chunks)} chunks so far")

    print(f"\nTotal chunks: {len(all_chunks)}")

    all_chunks = embed_chunks(all_chunks)
    store_chunks(all_chunks, repo_id)
    upload_chroma_to_s3(repo_id)
    mark_indexed(repo_id, manifest)

    return len(all_chunks)