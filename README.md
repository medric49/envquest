# EnvQuest
Play with reinforcement learning algorithms.

## Requirements
You need these requirements to run the project:
* python >= 3.8
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

### Run a random policy agent
Run a random agent in a Gym environment 
```shell
python -m scripts.run_random_agent
```

Read the python script for more information: [scripts/run_random_agent.py](scripts/run_random_agent.py)

### Train Agents

#### DQN Agent
Train a DQN policy agent in a Gym environment
```shell
python -m scripts.train_dqn
```

Read the python script for more information: [scripts/train_dqn.py](scripts/train_dqn.py)


#### Sarsa Agent
Train a Sarsa policy agent in a Gym environment
```shell
python -m scripts.train_sarsa
```

Read the python script for more information: [scripts/train_sarsa.py](scripts/train_sarsa.py)