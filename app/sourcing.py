import os
import json
import subprocess
from pathlib import Path
from app.config import GITHUB_TOKEN, S3_BUCKET
import requests
import boto3

# ── Constants ────────────────────────────────────────
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go",
    ".cpp", ".c", ".h", ".rs", ".rb",
    ".md", ".txt"
}
EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__",
    "dist", "build", ".venv", "venv",
    "migrations", "vendor"
}
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock",
    "poetry.lock", "Pipfile.lock"
}

REPOS_DIR = Path("data/repos")
MANIFESTS_DIR = Path("data/manifests")

# ── Functions ─────────────────────────────────────────

def clone_repo(github_url: str) -> Path:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = github_url.rstrip("/").split("/")[-1]
    owner = github_url.rstrip("/").split("/")[-2]
    dest_path = REPOS_DIR / f"{owner}__{repo_name}"

    if dest_path.exists():
        print(f"Repo exists, pulling latest: {dest_path}")
        subprocess.run(["git", "-C", str(dest_path), "pull"], check=True)
    else:
        print(f"Cloning {github_url} into {dest_path}")
        subprocess.run(["git", "clone", github_url, str(dest_path)], check=True)

    return dest_path


def get_repo_metadata(owner: str, repo: str) -> dict:
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(url, headers=headers)
    data = r.json()

    if "message" in data:
        raise RuntimeError(f"GitHub API error: {data['message']}")

    return {
        "full_name": data["full_name"],
        "description": data["description"],
        "language": data["language"],
        "stars": data["stargazers_count"],
        "default_branch": data["default_branch"],
        "topics": data.get("topics", [])
    }


def collect_files(repo_path: Path) -> list[dict]:
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix not in INCLUDE_EXTENSIONS:
            continue
        if path.stat().st_size > 500_000:
            continue
        files.append({
            "rel_path": str(path.relative_to(repo_path)),
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
            "filename": path.name
        })
    return files


def get_commit_sha(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def build_manifest(github_url: str, repo_path: Path, files: list[dict]) -> dict:
    parts = github_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    meta = get_repo_metadata(owner, repo)

    return {
        "repo": meta["full_name"],
        "description": meta["description"],
        "language": meta["language"],
        "stars": meta["stars"],
        "commit_sha": get_commit_sha(repo_path),
        "total_files": len(files),
        "total_size_bytes": sum(f["size_bytes"] for f in files),
        "files": files
    }


def save_manifest(manifest: dict, repo_id: str) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS_DIR / f"{repo_id}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved: {out}")
    return out


def upload_to_s3(local_path: Path, s3_key: str):
    endpoint = os.environ.get("AWS_ENDPOINT_URL", None)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    # Create bucket if it doesn't exist (LocalStack only)
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=S3_BUCKET)
        print(f"Created bucket: {S3_BUCKET}")

    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")