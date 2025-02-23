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
        args.agent = arguments.PPOAgentArguments(class_name="ppo", lr=0.001)
        agent = agents.ppo_agents.DiscretePPOAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            clip_threshold=args.agent.clip_threshold,
            num_policy_updates=args.agent.num_policy_updates,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        args.agent = arguments.ContinuousPPOAgentArguments(class_name="ppo", lr=0.001)
        agent = agents.ppo_agents.ContinuousPPOAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            noise_std_start=args.agent.noise_std_start,
            noise_std_end=args.agent.noise_std_end,
            noise_std_decay=args.agent.noise_std_decay,
            noise_std_step_duration=args.agent.noise_std_step_duration,
            clip_threshold=args.agent.clip_threshold,
            num_policy_updates=args.agent.num_policy_updates,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    # Define trainer
    trainer = trainers.mc_trainers.MCTrainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
