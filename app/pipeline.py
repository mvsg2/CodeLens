import json
import uuid
from pathlib import Path

import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from app.config import S3_BUCKET, AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
import boto3

# ── Setup ─────────────────────────────────────────────
parser = Parser(Language(tspython.language()))

enc = tiktoken.get_encoding("cl100k_base")
# model = SentenceTransformer("nomic-ai/nomic-embed-code")
model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", 
                            trust_remote_code=True,
                            device="cuda" if torch.cuda.is_available() else "cpu")

REPOS_DIR = Path("data/repos")
MANIFESTS_DIR = Path("data/manifests")
CHROMA_DIR = Path("data/chroma")

MAX_CHUNK_TOKENS = 512


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

    def walk(node, class_name=None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = name_node.text.decode()

        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode() if name_node else "unknown"

            docstring = ""
            body = node.child_by_field_name("body")
            if body and body.child_count > 0:
                first = body.children[0]
                if first.type == "expression_statement" and first.children:
                    inner = first.children[0]
                    if inner.type == "string":
                        docstring = inner.text.decode().strip("\"' ")

            functions.append({
                "name": func_name,
                "class": class_name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": node.text.decode(),
                "docstring": docstring
            })

        for child in node.children:
            walk(child, class_name)

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
                if estimate_tokens(code) > MAX_CHUNK_TOKENS:
                    sub_chunks = fixed_token_split(code, MAX_CHUNK_TOKENS)
                    for i, sub in enumerate(sub_chunks):
                        chunks.append(build_chunk(
                            content=sub, file_info=file_info,
                            func_name=func["name"], class_name=func["class"],
                            start_line=func["start_line"],
                            chunk_type="function_part", part_index=i
                        ))
                else:
                    chunks.append(build_chunk(
                        content=code, file_info=file_info,
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
            ids=[str(uuid.uuid4()) for _ in batch],
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

    return len(all_chunks)