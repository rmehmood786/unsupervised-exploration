.PHONY: install test run lint

install:
	python -m pip install -e .

test:
	pytest -q

run:
	python scripts/run_experiment.py --dataset iris --embed pca --cluster kmeans --n-clusters 3

lint:
	python -m pip install ruff
	ruff check .
