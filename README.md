# EnvQuest
Train and evaluate your autonomous agents in different environments using a collection of RL algorithms.

## Installation
To install the EnvQuest library, use `pip install envquest`.

## Usage

### Run a simple gym environment
```python
from envquest import envs, agents

# Instantiate an environment
env = envs.gym.GymEnvironment.from_task("LunarLander-v3")

# Instantiate an agent
agent = agents.generics.RandomAgent(env.observation_space, env.action_space)

# Execute an MDP
timestep = env.reset()

while not timestep.last():
    observation = timestep.observation
    action = agent.act(observation=observation)
    timestep = env.step(action)

# Render the environment
frame = env.render(256, 256)
```

### Usage with metaworld
```python
import metaworld
import random

from envquest import envs

ml1 = metaworld.ML1("basketball-v2")
task = random.choice(ml1.train_tasks)
env = ml1.train_classes["basketball-v2"](render_mode="rgb_array")
env.set_task(task)

env = envs.gym.GymEnvironment.from_env(env)
```

### Train a DQN Agent in a gym environment

First, set up a WandB logging environment
```shell
# Install wandb
pip install wandb

# Start a wandb local server
wandb server start
```

Then, train a DQN agent in a gym's CartPole-v1 environment.

```python
from envquest import arguments, envs, agents, trainers

# Define training arguments
args = arguments.TrainingArguments(
    env=arguments.EnvArguments(task="CartPole-v1"),
    agent=arguments.DQNAgentArguments(),
    logging=arguments.LoggingArguments(save_agent_snapshots=False)
)

# Instantiate an environment
env = envs.gym.GymEnvironment.from_task(task=args.env.task, max_episode_length=args.env.max_episode_length)

# Instantiate a DQN Agent
agent = agents.dqn_agents.DiscreteQNetAgent(
    mem_capacity=args.agent.mem_capacity,
    discount=args.agent.discount,
    n_steps=args.agent.n_steps,
    lr=args.agent.lr,
    tau=args.agent.tau,
    eps_start=args.agent.eps_start,
    eps_end=args.agent.eps_end,
    eps_step_duration=args.agent.eps_step_duration,
    eps_decay=args.agent.eps_decay,
    observation_space=env.observation_space,
    action_space=env.action_space,
)

# Instantiate a trainer
trainer = trainers.td_trainers.TDTrainer(env, agent, args)

# Start training
trainer.train()
```

Track the performances of your agent on wandb: http://localhost:8080/

## Examples
See some examples in the [examples](https://github.com/medric49/envquest/tree/master/examples) folder.
