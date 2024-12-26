import tyro
import gymnasium as gym

import playground as pg


def main():
    arguments = tyro.cli(pg.arguments.TrainingArguments)

    # Define environment
    env = pg.envs.gym.make_env(arguments.task, arguments.max_episode_length)

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = pg.agents.dqn.DiscreteQNetAgent(
            mem_capacity=arguments.mem_capacity,
            discount=arguments.discount,
            lr=arguments.lr,
            tau=arguments.tau,
            eps_start=arguments.eps_start,
            eps_end=arguments.eps_end,
            eps_step_duration=arguments.eps_step_duration,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        raise ValueError(
            f"'[{arguments.task}]' task is not a discrete gym environment. This script currently only supports discrete environments."
        )

    # Define trainer
    trainer = pg.trainers.Trainer(env, agent, arguments)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
