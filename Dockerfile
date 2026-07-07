FROM python:3.11-slim

WORKDIR /app

# build-essential: some deps (tree-sitter-python) may need to compile on
# platforms without a prebuilt wheel. curl: used by the container healthcheck.
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU build of torch, installed first: this container serves queries against
# an already-built Chroma index (mounted at runtime), it does not re-encode,
# so no GPU is needed here. Re-encoding stays a local/dev-machine or
# dedicated GPU job. Installing it before requirements.txt means pip sees
# torch already satisfied when resolving sentence-transformers' dependency
# on it, instead of pulling the multi-GB default CUDA build first and
# discarding it (Docker layers are additive, so a later uninstall doesn't
# shrink the image, it only hides the earlier layer's files from view).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
