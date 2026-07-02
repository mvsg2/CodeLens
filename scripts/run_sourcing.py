from dotenv import load_dotenv
load_dotenv()

from app.sourcing import (
    clone_repo, collect_files,
    build_manifest, save_manifest, upload_to_s3
)

REPO_URL = "https://github.com/fastapi/fastapi"

if __name__ == "__main__":
    # Step 1: Clone
    repo_path = clone_repo(REPO_URL)

    # Step 2: Collect files
    files = collect_files(repo_path)
    print(f"Found {len(files)} indexable files")

    # Step 3: Build manifest
    manifest = build_manifest(REPO_URL, repo_path, files)
    print(f"Repo: {manifest['repo']} | Stars: {manifest['stars']} | Commit: {manifest['commit_sha'][:7]}")

    # Step 4: Save manifest locally
    repo_id = manifest["repo"].replace("/", "__")
    manifest_path = save_manifest(manifest, repo_id)

    # Step 5: Upload manifest to S3 (LocalStack)
    upload_to_s3(manifest_path, f"repos/{repo_id}/manifest.json")

    print("\nSourcing complete.")
    print(f"  Repo cloned to : data/repos/{repo_id}/")
    print(f"  Manifest saved : data/manifests/{repo_id}.json")
    print(f"  S3 key         : repos/{repo_id}/manifest.json")