# chance\_of\_showers
Matthew Epland, PhD  

TODO

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
- `ana` for analysis tools, optional
- `dev` for CI and linting tools

```bash
poetry install --with daq
```
or
```bash
poetry install --with ana
```
