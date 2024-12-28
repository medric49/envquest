import gymnasium as gym

import playground as pg
from playground.arguments import TrainerArguments, TrainingArguments, SarsaAgentArguments, LoggingArguments


def main():
    # Training arguments
    arguments = TrainingArguments(
        trainer=TrainerArguments(num_updates=1, update_every_steps=1, num_seed_steps=0),
        agent=SarsaAgentArguments(),
        logging=LoggingArguments(save_agent_snapshots=False),
    )

    # Define environment
    env = pg.envs.gym.make_env(**arguments.env.__dict__)

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = pg.agents.sarsa.DiscreteSarsaAgent(
            discount=arguments.agent.discount,
            lr=arguments.agent.lr,
            eps_start=arguments.agent.eps_start,
            eps_end=arguments.agent.eps_end,
            eps_step_duration=arguments.agent.eps_step_duration,
            observation_space=env.observation_space,
            action_space=env.action_space,
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
