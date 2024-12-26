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

Start a local WandB server to track your experiments
```shell
wandb server start
```

## Usage

### Run a random policy agent ([scripts/run_random_agent.py](scripts/run_random_agent.py))
Run a random agent in a Gym environment 
```shell
python -m scripts.run_random_agent
```

### Train a DQN Agent ([scripts/train_dqn.py](scripts/train_dqn.py))
Train a DQN policy agent in a Gym environment
```shell
python -m scripts.train_dqn
```
