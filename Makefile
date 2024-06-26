.PHONY: all clean test

.PHONY: setupDAQ
setupDAQ:
	@poetry env use python3.11
	@poetry install --with daq,web,dev --no-root
	@poetry run pre-commit install

.PHONY: setupANA
setupANA:
	@poetry env use python3.11
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

.PHONY: pyupgrade
pyupgrade:
	@poetry run pyupgrade $(shell git ls-files '*.py' '*.ipynb')

.PHONY: black
black:
	@poetry run black .
	@poetry run blacken-docs --line-length=100 $(shell git ls-files '*.py' '*.ipynb' '*.md' '*.rst')

.PHONY: flake8
flake8:
	@poetry run flake8

.PHONY: ruff-check
ruff-check:
	@poetry run ruff check

.PHONY: ruff-fix
ruff-fix:
	@poetry run ruff check --fix-only

.PHONY: ruff-check-unsafe
ruff-check-unsafe:
	@poetry run ruff check --unsafe-fixes

.PHONY: ruff-fix-unsafe
ruff-fix-unsafe:
	@poetry run ruff check --fix-only --unsafe-fixes

.PHONY: mypy
mypy:
	@poetry run mypy .

# https://stackoverflow.com/a/63044665
.PHONY: pylint
pylint:
	@poetry run pylint $(shell git ls-files '*.py' '*.ipynb')

.PHONY: bandit
bandit:
	@poetry run bandit -q -r .

.PHONY: vulture
vulture:
	@poetry run vulture

.PHONY: vulture-update_ignore
vulture-update_ignore:
	@echo '# flake8: noqa' > .dev_config/.vulture_ignore.py
	@echo '# pylint: skip-file' >> .dev_config/.vulture_ignore.py
	@echo '# mypy: disable-error-code="name-defined"' >> .dev_config/.vulture_ignore.py
	-@poetry run vulture --make-whitelist >> .dev_config/.vulture_ignore.py || true # blocklint: pragma

.PHONY: pymend
pymend:
	@poetry run pymend $(shell git ls-files '*.py' '*.ipynb')

.PHONY: pymend-fix
pymend-fix:
	@poetry run pymend --write $(shell git ls-files '*.py' '*.ipynb')

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
	@prettier --ignore-path .dev_config/.prettierignore --ignore-path .gitignore --config .dev_config/.prettierrc.yaml --plugin=prettier-plugin-toml --write .

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

.PHONY: typos-fix
typos-fix:
	@poetry run typos -w

.PHONY: clean
clean:
	@find . -type d | grep -E "(.mypy_cache|.ipynb_checkpoints|.trash|__pycache__|.pytest_cache)" | xargs rm -rf
	@find . -type f | grep -E "(\.DS_Store|\.pyc|\.pyo)" | xargs rm -f

# Reverse prose and lint to avoid autocomplete matches with pre-commit
.PHONY: lintprose
lintprose:
	@poetry run proselint --config .dev_config/.proselint.json $(shell git ls-files | grep -v -E "media/|circuit_diagram/|poetry.lock|Makefile|daq/cron_jobs.txt") | grep -v -E "30ppm|né|high and dry"

# isort ~ isort:
# flake8 ~ noqa
# ruff ~ noqa
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
	@grep -rInH 'isort:\|noqa\|type:\|pylint\|nosec' $(shell git ls-files '*.py' '*.ipynb') || true
	@grep -rInH 'yamllint' $(shell git ls-files '*.yaml' '*.yml') || true
	@grep -rInH 'pragma\|blocklint:' $(shell git ls-files) || true
	@grep -rInH 'markdownlint-' $(shell git ls-files '*.md') || true
	@grep -rInH 'eslint' $(shell git ls-files '*.js') || true
	@grep -rInH 'prettier-ignore' $(shell git ls-files '*.html' '*.scss' '*.css') || true

# Find double spaces that are not leading, and that are not before a `#` character,
# i.e. indents and `code  # comment` are fine, but `code  # comment with  extra space` is not
.PHONY: find_double_spaces
find_double_spaces:
	@grep -rInHE '[^ \n] {2,}[^#]' $(shell git ls-files ':!:poetry.lock' ':!:media' ':!:daq/logs') || true

# Find trailing spaces
.PHONY: find_trailing_spaces
find_trailing_spaces:
	@grep -rInHE ' $$' $(shell git ls-files ':!:poetry.lock' ':!:media') || true
