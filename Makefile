.PHONY: help install-train install-serving test ci

help:
	@echo "Available targets:"
	@echo "  install-train   Install training dependencies"
	@echo "  install-serving Install serving dependencies"
	@echo "  test            Run unit tests"
	@echo "  ci              Run CI checks (imports + tests)"

install-train:
	pip install -r requirements-train.txt

install-serving:
	pip install -r requirements-serving.txt

test:
	pytest -q

ci: install-train install-serving
	python -c "import sklearn, mlflow; print('ok')"
	pytest -q
