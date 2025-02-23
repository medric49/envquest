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
        args.agent = arguments.AgentArguments(class_name="a2c", lr=0.001)
        agent = agents.a2c_agents.DiscreteA2CAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        args.agent = arguments.ContinuousPGAgentArguments(class_name="a2c", lr=0.001)
        agent = agents.a2c_agents.ContinuousA2CAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            noise_std_start=args.agent.noise_std_start,
            noise_std_end=args.agent.noise_std_end,
            noise_std_decay=args.agent.noise_std_decay,
            noise_std_step_duration=args.agent.noise_std_step_duration,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    # Define trainer
    trainer = trainers.mc_trainers.MCTrainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
