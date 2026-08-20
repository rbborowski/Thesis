.PHONY: install install-full test lint sample demo clean

install:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[full,dev]"

ollama-setup:
	@command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com"; exit 1; }
	ollama pull qwen2.5:3b-instruct
	@echo "Ready. Run with --llm ollama --llm-model qwen2.5:3b-instruct"

test:
	pytest -q

lint:
	ruff check src tests scripts

catalog:
	pip install kaggle
	python scripts/download_catalog.py

sample:
	python scripts/make_sample_catalog.py --n 2000

demo: sample
	python -m fashion_retrieval prepare
	python -m fashion_retrieval label-space
	python -m fashion_retrieval embed
	python -m fashion_retrieval gen-queries -n 100
	python -m fashion_retrieval evaluate
	python -m fashion_retrieval search "a black midi dress for a summer wedding"

clean:
	rm -rf data/processed/* results/* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
