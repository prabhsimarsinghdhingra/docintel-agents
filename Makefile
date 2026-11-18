setup:
	pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt
docs:
	python -m docintel.download
index:
	python -m docintel.build
eval:
	python -m docintel.evaluate
test:
	EMBEDDER=hashing pytest -q
serve:
	uvicorn docintel.api:app --reload --port 8000
mcp:
	python -m docintel.mcp_server
all: docs index eval
