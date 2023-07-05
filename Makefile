pre-commit:
	pre-commit run --all-files

format/black:
	poetry run black .

format/isort:
	poetry run isort .

format: format/black format/isort

lint/flake8:
	poetry run flake8

lint/mypy:
	poetry run mypy .

lint/yamllint:
	poetry run yamllint .

lint: lint/flake8 lint/mypy lint/yamllint

clean: format lint
