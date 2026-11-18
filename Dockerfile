FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
COPY . .
RUN chmod +x scripts/start.sh
EXPOSE 8000
CMD ["scripts/start.sh"]
