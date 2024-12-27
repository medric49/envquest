import tyro
import gymnasium as gym

import playground as pg


def main():
    arguments = tyro.cli(pg.arguments.TrainingArguments)

    # Define environment
    env = pg.envs.gym.make_env(**arguments.env.__dict__)

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = pg.agents.dqn.DiscreteQNetAgent(
            observation_space=env.observation_space, action_space=env.action_space, **arguments.agent.__dict__
        )
    else:
        raise ValueError(
            f"'[{arguments.env.task}]' task is not a discrete gym environment. This script currently only supports discrete environments."
        )

    # Define trainer
    trainer = pg.trainers.Trainer(env, agent, arguments)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
