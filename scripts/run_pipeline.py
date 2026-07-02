from app.pipeline import run_pipeline

REPO_ID = "fastapi__fastapi"

if __name__ == "__main__":
    print(f"Starting pipeline for: {REPO_ID}\n")
    total = run_pipeline(REPO_ID)
    print(f"\nPipeline Parsing and Encoding complete.")
    print(f"  Total chunks embedded and stored: {total}")
    print(f"  Chroma DB location: data/chroma/{REPO_ID}/")
    print(f"  S3 snapshot: s3://codelens-bucket/vector_stores/{REPO_ID}/")