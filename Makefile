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

isort:
	poetry run isort .

black:
	poetry run black .
	poetry run blacken-docs --line-length=100 $(shell git ls-files '*.py') $(shell git ls-files '*.md') $(shell git ls-files '*.rst')

flake8:
	poetry run flake8

mypy:
	poetry run mypy .

# https://stackoverflow.com/a/63044665
pylint:
	poetry run pylint $(shell git ls-files '*.py')

bandit:
	poetry run bandit -q -r .

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

detect-secrets:
	poetry run detect-secrets-hook --exclude-lines 'integrity='

yamllint:
	poetry run yamllint -c .dev_config/.yamllint.yaml --strict .

bklint: # Renamed to bklint so "make bl" autocompletes to "make black"
	poetry run blocklint --skip-files=poetry.lock --max-issue-threshold=1

markdownlint:
	markdownlint --config .dev_config/.markdownlint.yaml --ignore LICENSE.md --dot --fix .

standard:
	standard --fix

html5validator:
	html5validator --config .dev_config/.html5validator.yaml

fmt_prettier:
	prettier --ignore-path .dev_config/.prettierignore --ignore-path .gitignore --config .dev_config/.prettierrc.yaml --write .

# isort ~ isort:
# flake8 ~ noqa
# mypy ~ type:
# pylint ~ pylint
# bandit ~ nosec
# detect-secrets ~ pragma: allowlist
# yamllint ~ yamllint
# blocklint ~ blocklint: pragma
# markdownlint ~ <!-- markdownlint-disable -->
# standard ~ eslint
# prettier ~ <!-- prettier-ignore -->
find_noqa_comments:
	@grep -rIn 'isort:\|noqa\|type:\|pylint\|nosec' $(shell git ls-files '*.py')
	@grep -rIn 'yamllint' $(shell git ls-files '*.yaml' '*.yml')
	@grep -rIn 'pragma\|blocklint:' $(shell git ls-files '*')
	@grep -rIn 'markdownlint-' $(shell git ls-files '*.md')
	@grep -rIn 'eslint' $(shell git ls-files '*.js')
	@grep -rIn 'prettier-ignore' $(shell git ls-files '*.html' '*.scss' '*.css')
