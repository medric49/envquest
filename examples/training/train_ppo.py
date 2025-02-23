from dataclasses import asdict

import fire
import gymnasium as gym

from envquest import arguments, envs, agents, trainers
from envquest.agents.common import EpsilonDecay


def main(task: str = "CartPole-v1"):
    # Training arguments
    args = arguments.TrainingArguments(
        trainer=arguments.MCTrainerArguments(),
        agent=arguments.PPOAgentArguments(class_name="ppo", lr=0.001),
        logging=arguments.LoggingArguments(save_agent_snapshots=False),
        env=arguments.EnvArguments(task=task),
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
        agent = agents.ppo_agents.ContinuousPPOAgent(
            mem_capacity=args.agent.mem_capacity,
            discount=args.agent.discount,
            lr=args.agent.lr,
            eps_start=0.4,
            eps_end=0.05,
            eps_decay=EpsilonDecay.LINEAR,
            eps_step_duration=300000,
            clip_eps=args.agent.clip_eps,
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
