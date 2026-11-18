#!/bin/sh
set -e
if [ ! -f data/index/faiss.index ]; then
  python -m docintel.download
  python -m docintel.build
fi
exec uvicorn docintel.api:app --host 0.0.0.0 --port "${PORT:-8000}"
