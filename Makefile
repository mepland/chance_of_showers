setupDAQ:
	poetry install --with daq,web,dev
	pre-commit install

setupANA:
	poetry install --with ana,dev
	pre-commit install

poetry:
	poetry check
	poetry lock

pre-commit:
	# poetry run pre-commit autoupdate
	poetry run pre-commit run --all-files

black:
	poetry run black .
	poetry run blacken-docs --line-length=100 $(shell git ls-files '*.py') $(shell git ls-files '*.md') $(shell git ls-files '*.rst')

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

detect-secrets:
	poetry run detect-secrets-hook --exclude-lines 'integrity='

vulture:
	poetry run vulture

pyupgrade:
	poetry run pyupgrade $(shell git ls-files '*.py')

yamllint:
	poetry run yamllint -c .dev_config/.yamllint.yaml --strict .
