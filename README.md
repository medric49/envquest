# playground
Play with reinforcement learning algorithms.

## Requirements

You need these requirements to run the project:
* python >= 3.9
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* ffmpeg
```shell
# Linux
sudo apt install ffmpeg
# MacOS
brew install ffmpeg
```


## Installation

Install the python dependencies
```shell
uv venv .venv
source .venv/bin/activate
uv sync --frozen
```

Log in to WandB and paste your WandB API key
```shell
wandb login
```

## Usage
Run a random agent in a Gym environment
```shell
python -m scripts.run_random_agent
```