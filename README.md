# chance\_of\_showers
Matthew Epland, PhD

TODO Project description

[![healthchecks.io](https://healthchecks.io/badge/63dd8297-b724-4e7d-988b-7eeeca/0nnc0EMy.svg)](https://healthchecks.io)
[![tests](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml/badge.svg)](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mepland/chance_of_showers/blob/main/LICENSE.md)

## Data Acquisition (DAQ)

### Launching the DAQ Script
The provided [`start_daq`](daq/start_daq) bash script will start the [`daq.py`](daq/daq.py) script in a new `tmux` window.
You will need to update the `pkg_path` variable in `start_daq` per your installation location.
```bash
source daq/start_daq
```

### Opening the Web Dashboard
If `daq: {display_web: true}` is set in [`config.yaml`](config.yaml),
the local IP address and port of the dashboard will be logged on DAQ startup.
Open this link in your browser to see the live dashboard, as demonstrated below:

<div align="center">
  <video autoplay loop src="https://github.com/mepland/chance_of_showers/assets/4729931/f3b94d00-fa40-4b0b-8b95-1105d11e7acd" width=100></video>
</div>

### Setting up cron Jobs
Jobs to restart the DAQ on boot and every 30 minutes,
as well as send heartbeat API calls - see below,
are provided in the [`cron_jobs.txt`](daq/cron_jobs.txt) file.
Note that loading this file with `crontab` will overwrite any current cron jobs,
so check your existing settings first with `crontab -l`!
```bash
crontab -l

crontab daq/cron_jobs.txt
```

You can verify the cron jobs are running as expected with:
```bash
grep CRON /var/log/syslog | grep $LOGNAME
```

### Heartbeat Monitoring
You can use the provided [`heartbeat`](daq/heartbeat) bash script to send heartbeat API calls
for the DAQ script to [healthchecks.io](https://healthchecks.io) for monitoring and alerting.
[Configure your alert online at healthchecks.io](https://healthchecks.io/docs/configuring_checks/),
and then run the below commands to setup a `secrets.json` file with your alert's `uuid`.
You will need to update the `pkg_path` variable in `heartbeat` per your installation location.
The provided `cron_jobs.txt` will setup a cron job to send the heartbeat on the 15 and 45 minute of each hour.
```bash
sudo apt install jq
echo -e "{\n\t\"chance_of_showers_heartbeat_uuid\": \"YOUR_UUID_HERE\"\n}" > secrets.json
source daq/heartbeat
```

## Data Analysis

TODO

### Installing CUDA and PyTorch
1. Find the supported CUDA version (`11.8.0`) for the current release of PyTorch (`2.0.1`) [here](https://pytorch.org/get-started/locally/).
2. Install CUDA following the steps for the proper version and target platform [here](https://developer.nvidia.com/cuda-toolkit-archive).
3. Update the poetry `pytorch-gpu-src` source to point to the correct PyTorch version in `pyproject.toml`.
	- This is in place of `pip install --index-url=...` as provided by the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
4. Install the poetry `ana` group, `make setupANA`. This will install `pytorch`, along with the other needed packages.
5. Check that PyTorch and CUDA are correctly configured with the following `python` commands:
```python
import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA IS NOT AVAILABLE!")
```

## Dev

### Installing Python 3.11 on Raspbian
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

### Using the Makefile
A [`Makefile`](Makefile) is provided for convenience,
with commands to `make setupDAQ` or `make setupANA`,
as well run individual CI tests.

### Setting up pre-commit
It is recommended to use [`pre-commit`](https://pre-commit.com) tool to automatically check your commits locally as they are created.
You should just need to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts), see below, after installing the `dev` dependencies. This will run the checks in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) when you create a new commit.
```bash
pre-commit install
```
