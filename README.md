# EnvQuest
Train and evaluate your autonomous agents in different environments using a collection of RL algorithms.

## Installation
To install the EnvQuest library, use `pip install envquest`.

## Usage

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