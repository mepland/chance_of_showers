setupDAQ:
	poetry install --with daq,web,dev --no-root
	poetry run pre-commit install

setupANA:
	poetry install --with ana,dev --no-root
	poetry run pre-commit install

poetry:
	poetry check
	poetry lock

pre-commit:
	poetry run pre-commit run --all-files

pre-commit-this-commit:
	poetry run pre-commit run

pre-commit-update:
	poetry run pre-commit autoupdate

black:
	poetry run black .
	poetry run blacken-docs --line-length=100 $(shell git ls-files '*.py') $(shell git ls-files '*.md') $(shell git ls-files '*.rst')

flake8:
	poetry run flake8

mypy:
	poetry run mypy .

isort:
	poetry run isort .

# https://stackoverflow.com/a/63044665
pylint:
	poetry run pylint $(shell git ls-files '*.py')

bandit:
	poetry run bandit -q -r .

detect-secrets:
	poetry run detect-secrets-hook --exclude-lines 'integrity='

vulture:
	poetry run vulture

# Note: Returns an error signal, but .dev_config/.vulture_ignore.py is built fine
vulture-update_ignore:
	@echo '# flake8: noqa' > .dev_config/.vulture_ignore.py
	@echo '# pylint: skip-file' >> .dev_config/.vulture_ignore.py
	@echo '# mypy: disable-error-code="name-defined"' >> .dev_config/.vulture_ignore.py
	-@poetry run vulture --make-whitelist >> .dev_config/.vulture_ignore.py || true # blocklint: pragma

pyupgrade:
	poetry run pyupgrade $(shell git ls-files '*.py')

yamllint:
	poetry run yamllint -c .dev_config/.yamllint.yaml --strict .

bklint: # Renamed to bklint so "make bl" autocompletes to "make black"
	poetry run blocklint --skip-files=poetry.lock --max-issue-threshold=1

markdownlint:
	markdownlint --config .dev_config/.markdownlint.yaml --ignore LICENSE.md --dot --fix .

# flake8 ~ noqa
# pylint ~ pylint
# mypy ~ type:
# bandit ~ nosec
# detect-secrets ~ pragma
# yamllint ~ yamllint
find_noqa_comments:
	@grep -rIn 'noqa\|pylint\|type:\|nosec\|pragma' $(shell git ls-files '*.py')
	@grep -rIn 'yamllint' $(shell git ls-files '*.yaml')
