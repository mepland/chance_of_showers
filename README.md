# chance\_of\_showers
Matthew Epland, PhD

This project provides live water pressure measurements
via a web dashboard running on a Raspberry Pi,
logs the data, and creates time series forecasts of future water pressure.

[![Prophet](https://img.shields.io/badge/Prophet-3b5998.svg?style=flat&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBlbmFibGUtYmFja2dyb3VuZD0ibmV3IDAgMCA3Mi42IDcyIiB2ZXJzaW9uPSIxLjEiIHZpZXdCb3g9IjAgMCA3Mi42IDcyIiB4bWw6c3BhY2U9InByZXNlcnZlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOiNGRkZGRkY7fQo8L3N0eWxlPgo8cGF0aCBjbGFzcz0ic3QwIiBkPSJtMTAuMiA0OS4xdi0wLjFjMC03LjYgNS40LTEzLjggMTMuMy0xMy44IDEuMiAwIDIuMyAwLjIgMy40IDAuNGw2LjItNy42Yy0yLjgtMS4yLTYtMS44LTkuNS0xLjgtMTMuNy0wLjEtMjMuNiAxMC4zLTIzLjYgMjIuOXYwLjFjMCA1LjIgMS43IDEwLjEgNC43IDEzLjlsNi43LTguM2MtMC44LTEuNy0xLjItMy43LTEuMi01Ljd6Ii8+CjxwYXRoIGNsYXNzPSJzdDAiIGQ9Im0zNi42IDQ2LjFjMC4yIDEgMC4zIDIgMC4zIDN2MC4xYzAgNy42LTUuNCAxMy44LTEzLjMgMTMuOC0yLjMgMC00LjMtMC41LTYuMS0xLjVsLTcuOCA2LjNjMy44IDIuNiA4LjUgNC4yIDEzLjggNC4yIDEzLjcgMCAyMy42LTEwLjMgMjMuNi0yMi45di0wLjFjMC0zLjUtMC44LTYuNy0yLjEtOS42bC04LjQgNi43eiIvPgo8Y2lyY2xlIGNsYXNzPSJzdDAiIGN4PSI1Mi40IiBjeT0iMjAuMiIgcj0iNi45Ii8+CjxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjY4LjciIGN5PSIzLjkiIHI9IjMuOSIvPgo8L3N2Zz4K)](https://github.com/facebook/prophet)
[![Darts](https://img.shields.io/badge/Darts-0023f7?style=flat&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMzQ4IiBoZWlnaHQ9IjM0OCIgdmVyc2lvbj0iMS4xIiB2aWV3Qm94PSIwIDAgMzQ4IDM0OCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3NCwxNzQpIiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iOCI+CiAgPGNpcmNsZSByPSIxNzAiLz4KICA8Y2lyY2xlIGN4PSIyOCIgcj0iMTI0Ii8+CiAgPGNpcmNsZSBjeD0iNTciIHI9Ijc3Ii8+CiAgPGNpcmNsZSBjeD0iODUiIHI9IjMwIi8+CiA8L2c+Cjwvc3ZnPgo=)](https://github.com/unit8co/darts)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)

[![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHJvbGU9ImltZyIgdmVyc2lvbj0iMS4xIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiA8cGF0aCBkPSJtMTIgMGMtNi42MjMgMC0xMiA1LjM3Ny0xMiAxMnM1LjM3NyAxMiAxMiAxMiAxMi01LjM3NyAxMi0xMi01LjM3Ny0xMi0xMi0xMnptMCAwLjI3YzYuNDc0IDAgMTEuNzMgNS4yNTYgMTEuNzMgMTEuNzNzLTUuMjU2IDExLjczLTExLjczIDExLjczLTExLjczLTUuMjU2LTExLjczLTExLjczIDUuMjU2LTExLjczIDExLjczLTExLjczem0wLjAxNiAxMS40OTMtMi4xMjItOS41NTEtMC4wNjYgMC4wMTRhMTAuMDU2IDEwLjA1NiAwIDAgMC00LjIwOCAyLjAzNmwtMC4wNTIgMC4wNDMgNi4xODQgNy41MDEtNy42MDUtMy42NzItMC4wMjggMC4wNjJhOC44MzMgOC44MzMgMCAwIDAtMC42OCA1LjI5OWwwLjAxMiAwLjA2NiA4LjA4Ny0xLjQxMi00LjA2OSAxLjk3MyAwLjAxNyAwLjA1NmMwLjA3NSAwLjIzNCAwLjE2NiAwLjQ2MiAwLjI3NCAwLjY4M2wwLjAzNCAwLjA3IDQuMTAzLTIuNzgtMS4zMjYgNS45NjkgMC4wNjcgMC4wMTRjMC44MSAwLjE2MSAxLjY0NCAwLjE2MSAyLjQ1NCAwbDAuMDY1LTAuMDEzLTEuMDQ0LTUuODcgNi4wNzggNy42NjQgMC4wNTMtMC4wNDJhMTAuMDQgMTAuMDQgMCAwIDAgMS42MjktMS42MjlsMC4wNDItMC4wNTMtNy43MjItNi4xMjRoMi4yOTNsOGUtMyAtMC4wNThhMi41MTMgMi41MTMgMCAwIDAgMC0wLjY5M2wtMC4wMTEtMC4wNzYtMi4zMDIgMC42NCA0LjQ5OS01LjY5LTAuMDU1LTAuMDQyYTcuNDk1IDcuNDk1IDAgMCAwLTIuMDQxLTEuMDg4bC0wLjA2Mi0wLjAyMnoiIGZpbGw9ImJsYWNrIi8+Cjwvc3ZnPgo=)](https://matplotlib.org)
[![Polars](https://img.shields.io/badge/Polars-cd792c.svg?style=flat&logo=Polars&logoColor=white)](https://github.com/pola-rs/polars)
[![Pandas](https://img.shields.io/badge/Pandas-%150458.svg?style=flat&logo=pandas&logoColor=white)](https://github.com/pandas-dev/pandas)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-%238511FA.svg?style=flat&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![Socket.io](https://img.shields.io/badge/Socket.io-black?style=flat&logo=socket.io&badgeColor=010101)](https://socket.io)
[![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=flat&logo=Raspberry-Pi)](https://www.raspberrypi.com)
[![KiCad](https://img.shields.io/badge/KiCad-314CB0.svg?style=flat&logo=KiCad&logoColor=white)](https://www.kicad.org)

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json?style=flat)](https://python-poetry.org)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen?style=flat)](https://github.com/pylint-dev/pylint)
[![linting: flake8](https://img.shields.io/badge/flake8-checked-blueviolet?style=flat)](https://github.com/PyCQA/flake8)
[![checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://github.com/PyCQA/isort)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg?style=flat)](https://github.com/PyCQA/bandit)
[![linting: markdownlint](https://img.shields.io/badge/linting-markdownlint-blue?style=flat)](https://github.com/DavidAnson/markdownlint)
[![linting: html5validator](https://img.shields.io/badge/linting-html5validator-blue?style=flat)](https://github.com/svenkreiss/html5validator)
[![linting: StandardJS](https://img.shields.io/badge/StandardJS-222222.svg?style=flat&logo=StandardJS&logoColor=f3df49)](https://github.com/standard/standard)
[![linting: yamllint](https://img.shields.io/badge/linting-yamllint-blue?style=flat)](https://github.com/adrienverge/yamllint)
[![code style: Prettier](https://img.shields.io/badge/Prettier-222222.svg?style=flat&logo=Prettier&logoColor=f7b93e)](https://prettier.io)
[![linting: checkmake](https://img.shields.io/badge/linting-checkmake-blue?style=flat)](https://github.com/mrtazz/checkmake)
[![linting: shellcheck](https://img.shields.io/badge/linting-shellcheck-blue?style=flat)](https://github.com/koalaman/shellcheck)
[![linting: shfmt](https://img.shields.io/badge/linting-shfmt-blue?style=flat)](https://github.com/mvdan/sh)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat&logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![tests](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml/badge.svg?style=flat)](https://github.com/mepland/chance_of_showers/actions/workflows/tests.yml)
[![healthchecks.io](https://healthchecks.io/badge/63dd8297-b724-4e7d-988b-7eeeca/0nnc0EMy.svg?style=flat)](https://healthchecks.io)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE.md)

## Introduction
Living in a 5th floor walk up in NYC can save you on rent and gym memberships,
but runs the risk of leaving you high and dry when your water pressure gives out!
The pressure delivered from the city's water mains is
_[typically](https://cooperatornews.com/article/dispelling-myths-about-poor-water-pressure)_
sufficient to reach the 6th floor,
with higher buildings needing a booster pump and one of NYC's iconic rooftop
[water towers](https://www.amny.com/real-estate/water-towers-nyc-s-misunderstood-icons-1-6982696).
My building lacks a pump and water tower,
leaving my top floor apartment with _just barely_ satisfactory pressure,
as long as no other units are using water!
As you can see in the data below,
my daytime water pressure is all over the place.
After being stranded soapy and cold halfway through a shower one too many times,
I decided to use my data science and electronics skills to record
the time series of my apartment's hot water pressure
with the goal of forecasting future availability,
and hence `chance_of_showers` was born!

<div align="center">
  <video src="https://github.com/mepland/chance_of_showers/assets/4729931/f3b94d00-fa40-4b0b-8b95-1105d11e7acd"></video>
</div>

## Data Analysis Results
WIP

### Time Series Plots

Below is a sample of the pressure data collected in November 2023.
Clicking the links will open interactive plotly plots, please explore!

#### [Raw analog to digital converter (ADC) values](https://mepland.github.io/chance_of_showers/media/ana_outputs/mean_pressure_value_selected_data.html)

The data acquisition (DAQ) system saves the raw pressure data
from the analog to digital converter (ADC) as an integer between 0 and 65472.
Note that occasionally a water hammer will increase the pressure above its steady state value,
marked by the orange 100% reference line,
with a subsequent decay on the order of 10 minutes.
When water is flowing at the pressure sensor,
the data is shown with an open purple marker.
Using water reduces the pressure slightly under normal conditions,
and abruptly ends overpressure events.

#### [Normalized values](https://mepland.github.io/chance_of_showers/media/ana_outputs/mean_pressure_value_normalized_selected_data.html)

To clean the data before fitting any models,
I rescale the values to 0 and 1 between the steady state extrema.
Any values that are outside the normalization range are capped.

### Overall Pressure Distributions
| ![Mean Pressure Value Density](media/ana_outputs/mean_pressure_value_density.png) | ![Mean Pressure Value Normalized vs Time of Week](media/ana_outputs/mean_pressure_value_normalized_vs_time_of_week.png) | ![Mean Pressure Value Normalized vs Time of Day](media/ana_outputs/mean_pressure_value_normalized_vs_time_of_day.png) |
| :---: | :---: | :---: |

### Prophet Results
| ![Prophet Predict](media/ana_outputs/prophet/prophet_predict.png) | ![Prophet Components](media/ana_outputs/prophet/prophet_components.png) |
| :---: | :---: |
| ![Prophet Components Weekly](media/ana_outputs/prophet/prophet_component_weekly.png) | ![Prophet Components Daily](media/ana_outputs/prophet/prophet_component_daily.png) |

## Hardware

### Bill of Materials
Here is a list of the components I used in my build.
With suitable alterations, the project could definitely be carried out with a wide array of other
sensors, single board computers or microcontrollers, plumbing supplies, etc.

#### Electronics

* [Raspberry Pi 4 Model B 2 GB](https://www.raspberrypi.com/products/raspberry-pi-4-model-b)
  * [USB C Power Supply](https://www.raspberrypi.com/products/type-c-power-supply)
  * [Micro SD Card](https://www.amazon.com/gp/product/B09TQS634Y)
* [8-Channel 10-Bit ADC with SPI Interface - MCP3008](https://www.digikey.com/en/products/detail/microchip-technology/MCP3008-I-P/319422)
* [DFRobot Gravity Water Pressure Sensor - SEN0257](https://wiki.dfrobot.com/Gravity__Water_Pressure_Sensor_SKU__SEN0257)
* [Water Flow Hall Effect Sensor Switch - YWBL-WH](https://www.amazon.com/Interface-Electromagnetic-Flowmeter-Industrial-Accessory/dp/B08B1NG4FZ)
* [1 kΩ and 10 kΩ Resistors](https://www.amazon.com/gp/product/B072BL2VX1)
* 830 Point Breadboard and Dupont Jumper Wires - [Included in GPIO Kit](https://www.amazon.com/gp/product/B08B4SHS18)

#### Plumbing

* [1/2" NPT 3 Way Tee Connector](https://www.amazon.com/Stainless-Diverter-Movable-Flexible-Connector/dp/B09MT39487)
* [Faucet Connector Hose, 3/8" Female Compression Thread to 1/2" Female NPT Thread - B1F09](https://www.amazon.com/gp/product/B000BQWNP8)
* [Adapter, 3/8" Male NPT to 1/2" Female NPT](https://www.amazon.com/gp/product/B07LD3GN4X/ref=ppx_od_dt_b_asin_title_s01)
* [Adapter, 1/2" Male NPT to G1/4" Female - ADT-N12M-G14F](https://koolance.com/threading-adapter-npt-1-2-male-to-g-1-4-female-adt-n12m-g14f)
* [PTFE (Teflon) Thread Seal Tape](https://www.amazon.com/DOPKUSS-Plumbers-Sealant-Waterproof-Inches/dp/B095YCMHNX)

#### Optional Components

* [I2C OLED Display](https://www.amazon.com/dp/B01MRR4LVE)
* [Geekworm Baseplate](https://www.amazon.com/gp/product/B07WCBXFD3)
* Wiring
  * [GPIO Extension Cable Kit](https://www.amazon.com/gp/product/B08B4SHS18)
  * [Breadboard Jumper Wires](https://www.amazon.com/gp/product/B07CJYSL2T)
  * [Clip to Dupont Jumper Wires](https://www.amazon.com/gp/product/B08M5GNY47)
* Cooling
  * [Heatsink - Geekworm P165-B](https://www.amazon.com/gp/product/B08N5VZN8R)
  * [Fan - Noctua NF-A4x20 5V PWM 4-Pin 40x20mm](https://www.amazon.com/gp/product/B071FNHVXN)
  * [2x20 Pin Header Kit](https://www.amazon.com/gp/product/B08GC18NMK) to clear heatsink
  * One [M3 Screw](https://www.amazon.com/gp/product/B01I74TTWU) to attach fan to heatsink
  * Four [M2.5 Screws](https://www.amazon.com/HELIFOUNER-Screws-Washers-Kit-Threaded/dp/B0BKSGC86F)
to attach heatsink to Pi and baseplate

### Circuit Diagram
The circuit diagram for this implementation
is provided as a [KiCad](https://www.kicad.org) schematic
[here](circuit_diagram/circuit_diagram.kicad_sch).

![Circuit Diagram](circuit_diagram/circuit_diagram.svg)

### Photos
| ![Bottom](media/1_bottom.jpg) | ![Left](media/2_left.jpg) | ![Top](media/3_top.jpg) | ![Right](media/4_right.jpg) |
| :---: | :---: | :---: | :---: |
| ![Overhead](media/5_overhead.jpg) | ![Overhead Bottom OLED](media/6_overhead_bottom_oled.jpg) | ![Overhead Middle](media/7_overhead_middle.jpg) | ![Overhead Top GPIO](media/8_overhead_top_gpio.jpg) |
| ![Left Bottom](media/9_left_bottom.jpg) | ![Left Top](media/10_left_top.jpg) | ![Right Top](media/11_right_top.jpg) | ![In Situ](media/12_insitu.jpg) |
| ![In Situ OLED](media/13_insitu_oled.jpg) | ![In Situe OLED (Flash)](media/14_insitu_oled_flash.jpg) | ![Plumbing Front](media/15_plumbing_front.jpg) | ![Plumbing Back](media/16_plumbing_back.jpg) |

## Data Acquisition (DAQ)

The DAQ system recorded 95.4% of possible data points overall,
and 99.870% since implementing the cron job heartbeat monitoring.

### Launching the DAQ Script
The provided [`start_daq`](daq/start_daq) bash script
will start the [`daq.py`](daq/daq.py) and [`fan_control.py`](fan_control/fan_control.py)
scripts in new `tmux` windows.
You will need to update the `pkg_path` variable in `start_daq` per your installation location.

```bash
source daq/start_daq
```

### Opening the Web Dashboard
If `daq: {display_web: true}` is set in [`config.yaml`](config.yaml),
the local IP address and port of the dashboard will be logged on DAQ startup.
Open this link in your browser to see the live dashboard, as shown in the introduction.

### Setting up cron Jobs
Jobs to restart the DAQ on boot and every 30 minutes,
as well as send heartbeat API calls - see below,
are provided in the [`cron_jobs.txt`](daq/cron_jobs.txt) file.
Note that loading this file with `crontab` will overwrite **any** current cron jobs,
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
[Configure your alert online at healthchecks.io](https://healthchecks.io/docs/configuring_checks),
and then run the below commands to setup a `secrets.json` file with your alert's `uuid`.
You will need to update the `pkg_path` variable in `heartbeat` per your installation location.
The provided `cron_jobs.txt` will setup a cron job
to send the heartbeat on the 15 and 45 minute of each hour.

```bash
sudo apt install -y jq
echo -e "{\n\t\"chance_of_showers_heartbeat_uuid\": \"YOUR_UUID_HERE\"\n}" > secrets.json
source daq/heartbeat
```

The `heartbeat` script has also been setup to backup
the `daq/raw_data` and `daq/saved_data` directories to
`backup_path="/media/usb_drive/daq_backup"`.
Please configure `backup_path` to fit your path, or comment out the `rsync` lines to turn them off.
Regular backups of the data to a separate drive are helpful as Raspberry Pis
have been known to corrupt their SD cards due to power loss or excessive writes.

### Combining Raw DAQ Files

Raw CSV files can be combined into convenient Parquet files
prior to analysis with the [`etl.py`](daq/etl.py) script.
If the script crashes, you may need to manually repair
any lines in the CSV files corrupted due to power losses.
Polars should generate error messages indicating
the corrupt datetime to help you locate the problematic file and line.

```bash
python daq/etl.py
```

## Bayesian Optimization

To optimize the many hyperparameters present in this project,
both of the individual forecasting models themselves as well as how the data is prepared,
[Bayesian optimization](https://github.com/mepland/data_science_notes)
was used to efficiently sample the parameter space.
The functions needed to run Bayesian optimization
are located in [`bayesian_opt.py`](utils/bayesian_opt.py).

Unfortunately, actually running the optimization over GPU accelerated models
is not as simple as calling the `run_bayesian_opt()` function.
I have been unable to successfully detach the training of one GPU accelerated model
from the next when training multiple models in a loop.
The second training session will still have access to the tensors of the first,
leading to out of GPU memory errors, even when
[using commands like `gc.collect()` and `torch.cuda.empty_cache()`](https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch).
The `torch` models created by `darts` are very convenient,
but do not provide as much configurability as building your own `torch` model from scratch,
leading me unable to fix this issue in a clean way.

To work around the GPU memory issues, a shell script,
[`start_bayesian_opt`](ana/start_bayesian_opt), is used to repeatedly call `run_bayesian_opt()`
via the [`bayesian_opt_runner.py`](ana/bayesian_opt_runner.py) script.
In this way each model is trained in its own Python session,
totally clearing memory between training iterations.
A signed pickle file is used to quickly load the necessary data and settings on each iteration.
Instructions for running the whole Bayesian optimization workflow are provided below.

### Running Bayesian Optimization

1. Create the input `parent_wrapper.pickle` file for `bayesian_opt_runner.py`
via the `exploratory_ana.py` notebook.
2. Configure the run in `start_bayesian_opt` and `bayesian_opt_runner.py`.
3. Run the shell script, logging outputs to disk via:

```bash
./ana/start_bayesian_opt 2>&1 | tee ana/models/bayesian_optimization/bayesian_opt.log
```

## Dev Notes

### Data Analysis Setup - Installing CUDA and PyTorch

1. Find the supported CUDA version (`11.8.0`) for the current release of PyTorch (`2.0.1`) [here](https://pytorch.org/get-started/locally).
2. Install CUDA following the steps for the proper version and target platform [here](https://developer.nvidia.com/cuda-toolkit-archive).
3. Update the poetry `pytorch-gpu-src` source to point to the correct PyTorch version in [`pyproject.toml`](pyproject.toml).
    * This is in place of `pip install --index-url=...` as provided by the [PyTorch installation instructions](https://pytorch.org/get-started/locally).
4. Install the poetry `ana` group with `make setupANA`.
    * This will install `pytorch`, along with the other necessary packages.
5. Check that PyTorch and CUDA are correctly configured with the following `python` commands:

```python
import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA IS NOT AVAILABLE!")
```

### DAQ Setup - Installing Python 3.11 on Raspbian
If `python 3.11` is not available in your release of Raspbian,
you can compile it from source following the instructions [here](https://aruljohn.com/blog/python-raspberrypi),
but will also need to [install the sqlite extensions](https://stackoverflow.com/a/24449632):

<!-- markdownlint-disable MD013 -->
```bash
cd /usr/src/
sudo wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz
sudo tar -xzvf Python-3.11.4.tgz
cd Python-3.11.4/
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential libbz2-dev libc6-dev libexpat1-dev libffi-dev libgdbm-dev liblzma-dev libncurses5-dev libnss3-dev libsqlite3-dev libssl-dev lzma pkg-config zlib1g-dev
sudo apt autoremove -y
sudo apt update && sudo apt full-upgrade -y
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

#### Additional DAQ Dependencies
To finish setting up the DAQ system you must also:

* Install `tmux`, which is not included in Raspbian by default.
`tmux` is used to control multiple terminal sessions in [`start_daq`](daq/start_daq).
* Install `pigpio`, which is not included in Raspbian Lite, i.e. headless, installations.
`pigpio` is necessary to interface with the GPIO ports and must also be enabled via a [daemon](https://gpiozero.readthedocs.io/en/latest/remote_gpio.html)
* Enable SPI, I2C, and Remote GPIO via `raspi-config`.
* Prevent the [WiFi from powering off](https://desertbot.io/blog/headless-raspberry-pi-4-ssh-wifi-setup-64-bit-mac-windows).
* It is recommended to [install](https://pimylifeup.com/raspberry-pi-log2ram) [`log2ram`](https://github.com/azlux/log2ram) to avoid unnecessary writes to the SD card, prolonging the card's lifetime.

```bash
# Install tmux and pigpio
sudo apt-get install -y tmux pigpio

# Enable SPI, I2C, and Remote GPIO
sudo raspi-config

# Setup pigpio daemon
sudo systemctl enable pigpiod

# Prevent the WiFi from powering off
# Above the line that says exit 0 insert `/sbin/iw wlan0 set power_save off` and save the file
sudo vi /etc/rc.local

# Install log2ram
echo "deb [signed-by=/usr/share/keyrings/azlux-archive-keyring.gpg] http://packages.azlux.fr/debian/ bullseye main" | sudo tee /etc/apt/sources.list.d/azlux.list
sudo wget -O /usr/share/keyrings/azlux-archive-keyring.gpg  https://azlux.fr/repo.gpg
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y log2ram
```
<!-- markdownlint-enable MD013 -->

### Installing Dependencies with Poetry
Install `poetry` following the [instructions here](https://python-poetry.org/docs#installation).

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the `python` packages needed for this installation. Groups include:

* `daq` for packages needed to run the DAQ script on a Raspberry Pi, optional
* `web` for packages needed to run the live dashboard from the DAQ script, optional
* `ana` for analysis tools, optional
* `dev` for continuous integration (CI) and linting tools

```bash
poetry install --with daq,web
```

or

```bash
poetry install --with ana
```

### Setting up pre-commit
It is recommended to use the [`pre-commit`](https://pre-commit.com) tool
to automatically check your commits locally as they are created.
You should just need to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts),
see below, after installing the `dev` dependencies.
This will run the checks in [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
when you create a new commit.

```bash
pre-commit install
```

### Installing Non-Python Based Linters
Markdown is linted using [`markdownlint-cli`](https://github.com/igorshubovych/markdownlint-cli),
JavaScript by [`standard`](https://github.com/standard/standard),
and HTML, SCSS, CSS, and TOML by [`prettier`](https://prettier.io).
You can install these JavaScript-based linters globally with:

```bash
sudo npm install --global markdownlint-cli standard prettier
sudo npm install --global --save-dev --save-exact prettier-plugin-toml
```

Shell files are linted using [`shellcheck`](https://github.com/koalaman/shellcheck)
and [`shfmt`](https://github.com/mvdan/sh).
Follow the linked installation instructions for your system.
On Fedora they are:

```bash
sudo dnf install ShellCheck shfmt
```

### Using the Makefile
A [`Makefile`](Makefile) is provided for convenience,
with commands to setup the DAQ and analysis environments,
`make setupDAQ` and `make setupANA`,
as well run CI and linting tools,
e.g. `make black`, `make pylint`, `make pre-commit`.
