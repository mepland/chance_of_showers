# chance\_of\_showers
Matthew Epland, PhD

TODO

## Opening the Web Dashboard
If `daq.py` is running with `display_web = True`,
the local IP address and port of the dashboard will be logged on startup.

## Setting up cron Jobs
Note that loading a file with `crontab` will overwrite any existing cron jobs, so check first with `crontab -l`!
```bash
crontab -l
crontab cron_jobs.txt
```

You can verify the cron jobs are running as expected with:
```bash
grep CRON /var/log/syslog | grep $LOGNAME
```

## Dev
[![tests](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml/badge.svg)](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mepland/chance_of_showers/blob/main/LICENSE.md)

### Installing Python 3.11 on Raspberry Pi
If `python 3.11` is not available in your release of Raspbian,
you can compile it from source following the instructions [here](https://aruljohn.com/blog/python-raspberrypi),
but will also need to [install the sqlite extensions](https://stackoverflow.com/a/24449632):
```bash
cd /usr/src/
sudo wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz
sudo tar -xzvf Python-3.11.4.tgz
cd Python-3.11.4/
sudo apt update && sudo apt full-upgrade -y
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libsqlite3-dev -y
./configure --enable-optimizations --enable-loadable-sqlite-extensions
sudo make altinstall

# Should be Python 3.11.4 with your compile info
/usr/local/bin/python3.11 -VV

# Link binary
sudo rm /usr/bin/python
sudo rm /usr/bin/python3
sudo ln -s /usr/local/bin/python3.11 /usr/bin/python
sudo ln -s /usr/local/bin/python3.11 /usr/bin/python3

# Should match /usr/local/bin/python3.11 -VV
python -VV
```

### Installing Dependencies with Poetry
Install `poetry` following the [instructions here](https://python-poetry.org/docs/#installation).
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then install the `python` packages needed for this installation. Groups include:
- `daq` for packages needed to run the DAQ script on a Raspberry Pi, optional
- `web` for packages needed to run the live dashboard from the DAQ script, optional
- `ana` for analysis tools, optional
- `dev` for CI and linting tools

```bash
poetry install --with daq,web
```
or
```bash
poetry install --with ana
```

### Setting up pre-commit
You should just need to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts) after installing the `dev` dependencies.
```bash
pre-commit install
```

### Using the Makefile
A `Makefile` is provided convenience,
with commands to `make setupDAQ` or `make setupANA`,
as well run individual CI tests.
