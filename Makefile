.PHONY: install test lint clean run docker demo

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

demo:
	python -m src.demo.webcam_demo

docker:
	docker build -t $(shell basename $(CURDIR)) .
	docker run -p 8000:8000 $(shell basename $(CURDIR))
