# chance\_of\_showers
Matthew Epland, PhD

TODO

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Cloning the Repository
ssh
```bash
git clone git@github.com:mepland/chance_of_showers.git
```

https
```bash
git clone https://github.com/mepland/chance_of_showers.git
```

## Installing Dependencies with Poetry
Install poetry following the [instructions here](https://python-poetry.org/docs/#installation).
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then install the `python` packages needed for this installation. Groups include:
- `daq` for packages needed to run the DAQ script on a Raspbery Pi, optional
- `web` for packages needed to run the live dashboard from the DAQ script, optional
- `ana` for analysis tools, optional
- `dev` for CI and linting tools

```bash
poetry install --with daq,web,dev
```
or
```bash
poetry install --with ana,dev
```

## Opening the Web Dashboard
If `daq.py` is running with `display_web = True`,
the local IP address and port of the dashboard will be logged on startup.
