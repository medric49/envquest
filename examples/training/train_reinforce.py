from dataclasses import asdict

import fire
import gymnasium as gym

from envquest import arguments, envs, agents, trainers


def main(task: str = "CartPole-v1"):
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.MCTrainerArguments(),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
        env=arguments.EnvArguments(task=task),
    )

    # Define environment
    env = envs.gym.GymEnvironment.from_task(**asdict(args.env))

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.agent = arguments.AgentArguments(class_name="reinforce")
        agent = agents.renforce_agents.DiscreteREINFORCEAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        raise ValueError(
            f"'[{args.env.task}]' task is not a discrete gym environment. This script currently only supports discrete environments."
        )

    # Define trainer
    trainer = trainers.mc_trainers.MCTrainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
