# EnvQuest
Train and evaluate your autonomous agents in different environments using a collection of RL algorithms.

## Installation
To install the EnvQuest library, use `pip install envquest`.

## Usage

### Run a simple gym environment
```python
import envquest as eq

# Instantiate an environment
env = eq.envs.gym.make_env("LunarLander-v3")

# Instantiate an agent
agent = eq.agents.simple.RandomAgent(env.observation_space, env.action_space)

# Execute an MDP
timestep = env.reset()

while not timestep.last():
    observation = timestep.observation
    action = agent.act(observation=observation)
    timestep = env.step(action)

# Render the environment
frame = env.render(256, 256)
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
import envquest as eq

# Training arguments
arguments = eq.arguments.TrainingArguments(
    env=eq.arguments.EnvArguments(task="CartPole-v1"),
    agent=eq.arguments.DQNAgentArguments(), 
    logging=eq.arguments.LoggingArguments(save_agent_snapshots=False)
)

# Instantiate an environment
env = eq.envs.gym.make_env(task=arguments.env.task, max_episode_length=arguments.env.max_episode_length)

# Instantiate a DQN Agent
agent = eq.agents.dqn.DiscreteQNetAgent(
    mem_capacity=arguments.agent.mem_capacity,
    discount=arguments.agent.discount,
    n_steps=arguments.agent.n_steps,
    lr=arguments.agent.lr,
    tau=arguments.agent.tau,
    eps_start=arguments.agent.eps_start,
    eps_end=arguments.agent.eps_end,
    eps_step_duration=arguments.agent.eps_step_duration,
    eps_decay=arguments.agent.eps_decay,
    observation_space=env.observation_space,
    action_space=env.action_space,
)

# Instantiate a trainer
trainer = eq.trainers.Trainer(env, agent, arguments)

# Start training
trainer.train()
```

Track the performances of your agent on wandb: http://localhost:8080/

## Examples
See some examples in the [examples](https://github.com/medric49/envquest/tree/master/examples) folder.
