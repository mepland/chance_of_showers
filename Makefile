pre-commit:
	pre-commit run --all-files

format/black:
	poetry run black .

format/isort:
	poetry run isort .

# TODO rework bash here
# https://stackoverflow.com/a/63044665
#format/pylint:
#	bash -c "pylint $(git ls-files '*.py')"

format: format/black format/isort # format/pylint

lint/flake8:
	poetry run flake8

lint/mypy:
	poetry run mypy .

lint/yamllint:
	poetry run yamllint .

lint: lint/flake8 lint/mypy lint/yamllint

clean: format lint
