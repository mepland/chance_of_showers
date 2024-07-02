.PHONY: all clean test

# Clean temporary files

.PHONY: clean
clean:
	@find . -type d | grep -E "(.mypy_cache|.ipynb_checkpoints|.trash|__pycache__|.pytest_cache|.ruff_cache)" | xargs rm -rf
	@find . -type f | grep -E "(\.DS_Store|\.pyc|\.pyo)" | xargs rm -f

# Setup

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

# pre-commit

.PHONY: pre-commit
pre-commit:
	@poetry run pre-commit run --all-files

.PHONY: pre-commit-this-commit
pre-commit-this-commit:
	@poetry run pre-commit run

.PHONY: pre-commit-update
pre-commit-update:
	@poetry run pre-commit autoupdate

# Python

.PHONY: poetry
poetry:
	@poetry check
	@poetry lock --no-update

.PHONY: isort
isort:
	@poetry run isort .

.PHONY: pyupgrade
pyupgrade:
	@poetry run pyupgrade $(shell git ls-files '*.py' '*.ipynb')

.PHONY: black
black:
	@poetry run black .
	@poetry run blacken-docs --line-length=100 --target-version=py311 $(shell git ls-files '*.py' '*.ipynb' '*.md' '*.rst' '*.tex')

.PHONY: flake8
flake8:
	@poetry run flake8

# Warning: flake8-markdown does not use the pyproject.toml config correctly
.PHONY: flake8-markdown
flake8-markdown:
	@poetry run flake8-markdown $(shell git ls-files '*.md')

.PHONY: ruff
ruff:
	@poetry run ruff check

.PHONY: ruff-fix
ruff-fix:
	@poetry run ruff check --fix-only

.PHONY: ruff-fix-unsafe
ruff-fix-unsafe:
	@poetry run ruff check --fix-only --unsafe-fixes

# https://stackoverflow.com/a/63044665
.PHONY: pylint
pylint:
	@poetry run pylint $(shell git ls-files '*.py')

.PHONY: mypy
mypy:
	@poetry run mypy .

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

# Javascript

.PHONY: standard
standard:
	@standard --fix

# HTML

.PHONY: html5validator
html5validator:
	@html5validator --config .dev_config/.html5validator.yaml

# Prettier: HTML CSS SCSS JSON TOML

.PHONY: fmt_prettier
fmt_prettier:
	@prettier --ignore-path .dev_config/.prettierignore --ignore-path .gitignore --config .dev_config/.prettierrc.yaml --plugin=prettier-plugin-toml --write .

# YAML

.PHONY: yamllint
yamllint:
	@poetry run yamllint -c .dev_config/.yamllint.yaml --strict .

# Makefile

.PHONY: checkmake
checkmake:
	@pre-commit run checkmake --all-files

# Shell scripts

.PHONY: shellcheck
shellcheck:
	@shellcheck $(shell git ls-files | grep -vE '.*\..*' | grep -v 'Makefile')

.PHONY: shfmt
shfmt:
	@shfmt -bn -ci -sr -s -w $(shell git ls-files | grep -vE '.*\..*' | grep -v 'Makefile')

# Markdown

.PHONY: markdownlint
markdownlint:
	@markdownlint --config .dev_config/.markdownlint.yaml --ignore LICENSE.md --dot --fix .

# GitHub actions

.PHONY: actionlint
actionlint:
	@pre-commit run actionlint --all-files

# All files

.PHONY: detect-secrets
detect-secrets:
	@poetry run detect-secrets-hook --exclude-lines 'integrity='

# Renamed to bklint so "make bl" autocompletes to "make black"
.PHONY: bklint
bklint:
	@poetry run blocklint --skip-files=poetry.lock --max-issue-threshold=1

.PHONY: typos
typos:
	@poetry run typos --format=brief

.PHONY: typos-long
typos-long:
	@poetry run typos --format=long

.PHONY: typos-fix
typos-fix:
	@poetry run typos -w

# Reverse prose and lint to avoid autocomplete matches with pre-commit
# Runs slowly and has many false positives, use locally only
.PHONY: lintprose
lintprose:
	@poetry run proselint --config .dev_config/.proselint.json $(shell git ls-files | grep -v -E "media/|circuit_diagram/|poetry.lock|Makefile|daq/cron_jobs.txt") | grep -v -E "30ppm|né|high and dry"

# Find double spaces that are not leading, and that are not before a `#` character,
# i.e. indents and `code  # comment` are fine, but `code  # comment with  extra space` is not
# Some false positives, so do not implement in GitHub actions tests.yml
.PHONY: find_double_spaces
find_double_spaces:
	@grep -rInHE '[^ \n] {2,}[^#]' $(shell git ls-files ':!:poetry.lock' ':!:media' ':!:daq/logs') || true

.PHONY: find_trailing_spaces
find_trailing_spaces:
	@grep -rInHE ' $$' $(shell git ls-files ':!:poetry.lock' ':!:media') || true

# Python linting helper commands

# isort ~ isort:
# flake8 ~ noqa
# ruff ~ noqa
# pylint ~ pylint
# mypy ~ type:
# bandit ~ nosec
# standard ~ eslint
# prettier ~ <!-- prettier-ignore -->
# yamllint ~ yamllint
# markdownlint ~ <!-- markdownlint-disable -->
# detect-secrets ~ pragma: allowlist
# blocklint ~ blocklint: pragma

# mypy 'type:' has to be the first noqa tag in a comment

.PHONY: find_noqa_comments
find_noqa_comments:
	@TMP_FIND_NOQA_COMMENTS=$(shell mktemp /tmp/find_noqa_comments.XXXXXX); \
	# isort, flake8, ruff, pylint, mypy, bandit; \
	grep -rInH 'isort:\|noqa\|pylint\| type:\|nosec' $(shell git ls-files '*.py' '*.ipynb') >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# standard; \
	grep -rInH 'eslint' $(shell git ls-files '*.js') >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# prettier; \
	grep -rInH 'prettier-ignore' $(shell git ls-files '*.html' '*.scss' '*.css' '*.toml') >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# yamllint; \
	grep -rInH 'yamllint [de]' $(shell git ls-files '*.yaml' '*.yml') >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# markdownlint; \
	grep -rInH 'markdownlint-[de]' $(shell git ls-files '*.md') >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# detect-secrets, blocklint; \
	grep -rInH 'pragma: allowlist\|blocklint:.*pragma' $(shell git ls-files) >> $$TMP_FIND_NOQA_COMMENTS || true; \
	# Remove false positive lines from Makefile and .pre-commit-config.yaml; \
	sed -i -r '/^Makefile:[0-9]+:(# detect-secrets ~|# blocklint ~|.*grep)/d' $$TMP_FIND_NOQA_COMMENTS; \
	# CONFIG VIA COMMENTING; \
	# Remove code, keep comments; \
	# perl -pi -E 's/(^.*?:[0-9]+:).*?(#|<\!-- )/\1\2/g' $$TMP_FIND_NOQA_COMMENTS; \
	# Remove filename, line number, and code, keep comments; \
	perl -pi -E 's/(^.*?:[0-9]+:).*?(#|<\!-- )/\2/g' $$TMP_FIND_NOQA_COMMENTS; \
	# Sort and remove duplicates; \
	sort -u -o $$TMP_FIND_NOQA_COMMENTS $$TMP_FIND_NOQA_COMMENTS; \
	# Print results; \
	cat $$TMP_FIND_NOQA_COMMENTS; \
	# Remove temporary file; \
	rm -f $$TMP_FIND_NOQA_COMMENTS; \

.PHONY: find_pymend_placeholders
find_pymend_placeholders:
	@grep -rInH '_description_\|_type_\|_summary_\|__UnknownError__' $(shell git ls-files '*.py' '*.ipynb') || true
