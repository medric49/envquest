import tyro

import playground as pg


def main():
    arguments = tyro.cli(pg.arguments.TrainingArguments)

    # Define environment
    env = pg.envs.gym.make_env(arguments.task)

    # Define agent
    agent = pg.agents.DiscreteQNetAgent(
        mem_capacity=arguments.mem_capacity,
        discount=arguments.discount,
        lr=arguments.lr,
        tau=arguments.tau,
        eps_start=arguments.eps_start,
        eps_end=arguments.eps_end,
        eps_duration=arguments.eps_duration,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Define trainer
    trainer = pg.trainers.Trainer(arguments, agent, env)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
