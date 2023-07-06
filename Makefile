poetry:
	poetry check
	poetry lock

pre-commit:
	poetry run pre-commit autoupdate
	poetry run pre-commit run --all-files

black:
	poetry run black .

flake8:
	poetry run flake8

mypy:
	poetry run mypy .

isort:
	poetry run isort .

pylint:
	# https://stackoverflow.com/a/63044665
	poetry run pylint $(shell git ls-files '*.py')

bandit:
	poetry run bandit -r .

vulture:
	poetry run vulture

yamllint:
	poetry run yamllint .
