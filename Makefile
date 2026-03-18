.PHONY: install dev test serve demo price clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

serve:
	python -m kairos_terminal serve

demo:
	python -m kairos_terminal demo --steps 250

price:
	python -m kairos_terminal price --process heston

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +