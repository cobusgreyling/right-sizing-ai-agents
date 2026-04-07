.PHONY: install run docker test clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

docker:
	docker compose up --build

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
