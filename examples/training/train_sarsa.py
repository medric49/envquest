from dataclasses import asdict

import fire
import gymnasium as gym

from envquest import arguments, envs, agents, trainers


def main(task: str = "CartPole-v1"):
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.TrainerArguments(num_updates=1, update_every_steps=1, num_seed_steps=0),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
        env=arguments.EnvArguments(task=task),
    )

    # Define environment
    env = envs.gym.GymEnvironment.from_task(**asdict(args.env))

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.agent = arguments.SarsaAgentArguments()
        agent = agents.sarsa_agents.DiscreteSarsaAgent(
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
    trainer = trainers.td_trainers.TDTrainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
