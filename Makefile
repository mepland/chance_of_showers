.PHONY: all clean test

.PHONY: setupDAQ
setupDAQ:
	@poetry install --with daq,web,dev --no-root
	@poetry run pre-commit install

.PHONY: setupANA
setupANA:
	@poetry install --with ana,dev --no-root
	@poetry run pre-commit install

.PHONY: poetry
poetry:
	@poetry check
	@poetry lock --no-update

.PHONY: pre-commit
pre-commit:
	@poetry run pre-commit run --all-files

.PHONY: pre-commit-this-commit
pre-commit-this-commit:
	@poetry run pre-commit run

.PHONY: pre-commit-update
pre-commit-update:
	@poetry run pre-commit autoupdate

.PHONY: isort
isort:
	@poetry run isort .

.PHONY: black
black:
	@poetry run black .
	@poetry run blacken-docs --line-length=100 $(shell git ls-files '*.py') $(shell git ls-files '*.md') $(shell git ls-files '*.rst')

.PHONY: flake8
flake8:
	@poetry run flake8

.PHONY: mypy
mypy:
	@poetry run mypy .

# https://stackoverflow.com/a/63044665
.PHONY: pylint
pylint:
	@poetry run pylint $(shell git ls-files '*.py')

.PHONY: bandit
bandit:
	@poetry run bandit -q -r .

.PHONY: vulture
vulture:
	@poetry run vulture

# Note: Returns an error signal, but .dev_config/.vulture_ignore.py is built fine
.PHONY: vulture-update_ignore
vulture-update_ignore:
	@echo '# flake8: noqa' > .dev_config/.vulture_ignore.py
	@echo '# pylint: skip-file' >> .dev_config/.vulture_ignore.py
	@echo '# mypy: disable-error-code="name-defined"' >> .dev_config/.vulture_ignore.py
	-@poetry run vulture --make-whitelist >> .dev_config/.vulture_ignore.py || true # blocklint: pragma

.PHONY: pyupgrade
pyupgrade:
	@poetry run pyupgrade $(shell git ls-files '*.py')

.PHONY: deptry
deptry:
	@poetry run deptry .

.PHONY: detect-secrets
detect-secrets:
	@poetry run detect-secrets-hook --exclude-lines 'integrity='

.PHONY: yamllint
yamllint:
	@poetry run yamllint -c .dev_config/.yamllint.yaml --strict .

# Renamed to bklint so "make bl" autocompletes to "make black"
.PHONY: bklint
bklint:
	@poetry run blocklint --skip-files=poetry.lock --max-issue-threshold=1

.PHONY: markdownlint
markdownlint:
	@markdownlint --config .dev_config/.markdownlint.yaml --ignore LICENSE.md --dot --fix .

.PHONY: standard
standard:
	@standard --fix

.PHONY: html5validator
html5validator:
	@html5validator --config .dev_config/.html5validator.yaml

.PHONY: fmt_prettier
fmt_prettier:
	@prettier --ignore-path .dev_config/.prettierignore --ignore-path .gitignore --config .dev_config/.prettierrc.yaml --write .

.PHONY: checkmake
checkmake:
	@pre-commit run checkmake --all-files

.PHONY: shellcheck
shellcheck:
	@shellcheck $(shell git ls-files | grep -vE '.*\..*' | grep -v 'Makefile')

.PHONY: shfmt
shfmt:
	@shfmt -bn -ci -sr -s -w $(shell git ls-files | grep -vE '.*\..*' | grep -v 'Makefile')

.PHONY: typos
typos:
	@poetry run typos --format=brief

.PHONY: typos-long
typos-long:
	@poetry run typos --format=long

.PHONY: clean
clean:
	@find . -type d | grep -E "(.mypy_cache|.ipynb_checkpoints|.trash|__pycache__|.pytest_cache)" | xargs rm -rf
	@find . -type f | grep -E "(\.DS_Store|\.pyc|\.pyo)" | xargs rm -f

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
.PHONY: find_noqa_comments
find_noqa_comments:
	@grep -rIn 'isort:\|noqa\|type:\|pylint\|nosec' $(shell git ls-files '*.py')
	@grep -rIn 'yamllint' $(shell git ls-files '*.yaml' '*.yml')
	@grep -rIn 'pragma\|blocklint:' $(shell git ls-files '*')
	@grep -rIn 'markdownlint-' $(shell git ls-files '*.md')
	@grep -rIn 'eslint' $(shell git ls-files '*.js')
	@grep -rIn 'prettier-ignore' $(shell git ls-files '*.html' '*.scss' '*.css')
