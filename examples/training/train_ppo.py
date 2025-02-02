from dataclasses import asdict

import fire
import gymnasium as gym

from envquest import arguments, envs, agents, trainers


def main():
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.MCTrainerArguments(),
        agent=arguments.PPOAgentArguments(class_name="ppo"),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
        env=arguments.EnvArguments(task="CartPole-v1"),
    )

    # Define environment
    env = envs.gym.GymEnvironment.from_task(**asdict(args.env))

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = agents.ppo_agents.DiscretePPOAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            clip_eps=args.agent.clip_eps,
            num_policy_updates=args.agent.num_policy_updates,
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
