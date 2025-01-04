from dataclasses import asdict

import gymnasium as gym

from envquest import arguments, envs, agents, trainers


def main():
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.TrainerArguments(num_updates=1, update_every_steps=1, num_seed_steps=0),
        agent=arguments.SarsaAgentArguments(),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
    )

    # Define environment
    env = envs.gym.GymEnvironment.from_task(**asdict(args.env))

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = agents.sarsa.DiscreteSarsaAgent(
            discount=args.agent.discount,
            lr=args.agent.lr,
            eps_start=args.agent.eps_start,
            eps_end=args.agent.eps_end,
            eps_step_duration=args.agent.eps_step_duration,
            eps_decay=args.agent.eps_decay,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        raise ValueError(
            f"'[{args.env.task}]' task is not a discrete gym environment. This script currently only supports discrete environments."
        )

    # Define trainer
    trainer = trainers.Trainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
