pre-commit:
	poetry run pre-commit run --all-files

black:
	poetry run black .

flake8:
	poetry run flake8

mypy:
	poetry run mypy .

isort:
	poetry run isort .

#pylint:
# https://stackoverflow.com/a/63044665
# TODO rework bash here
#	bash -c "pylint $(git ls-files '*.py')"

bandit:
	poetry run bandit -r .

vulture:
	poetry run vulture

yamllint:
	poetry run yamllint .
