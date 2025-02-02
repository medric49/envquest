from dataclasses import asdict

import fire
import gymnasium as gym

from envquest import arguments, envs, agents, trainers


def main(task: str = "CartPole-v1"):
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.MCTrainerArguments(),
        agent=arguments.AgentArguments(class_name="vanilla_pg", lr=0.001),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
        env=arguments.EnvArguments(task=task),
    )

    # Define environment
    env = envs.gym.GymEnvironment.from_task(**asdict(args.env))

    # Define agent
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent = agents.pg_agents.DiscreteVanillaPGAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    else:
        agent = agents.pg_agents.ContinuousVanillaPGAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    # Define trainer
    trainer = trainers.mc_trainers.MCTrainer(env, agent, args)

    # Start training
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
