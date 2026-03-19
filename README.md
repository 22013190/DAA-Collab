# Multi-Agent System ST Engineering

WIP on agentic workflows between various specialised agents to produce charts, reports and dashboard.

## Installation & Setup

This projects uses [uv](https://github.com/astral-sh/uv) for tooling, workflows and virtual environments.

### git

This project holds submodules of additional projects

First time pulling the repo, run `git submodule update --init --recursive` or `git submodule update --recursive --remote` (git >1.8.2)

Updating submodule to latest changes `git submodule foreach --recursive git pull origin main` or `git submodule update --remote --merge` (git >1.8.2)

### uv

`uv venv --python 3.13` to create virtual environment files

Make sure env is set in your terminal instance correctly e.g. `source .venv/bin/activate.<term>`

`uv sync --refresh --reinstall --all-extras` to make sure dependencies are installed correctly

    Note: Do not have to install all optionals, can specify with `--extra <optional-name>` to install specific group of dependencies.

`uv build --all-packages` to build the entire package and resolve import paths

`uv run pip install dist/<file>.whl` to install package (optional)

### Running Commands

`uv run .\src\agents\analysis\a2a_compliant_service.py` to start the a2a service and `uv run .\src\agents\analysis\helpers\a2a_client.py`

`uv run` `...` to run specific python/pip package commands (similar to `pip run`)

## Dependencies

Please see in `pyproject.toml` for specifics.
