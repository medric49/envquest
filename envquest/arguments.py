from dataclasses import dataclass

from envquest.agents.common import DecayType


@dataclass
class EnvArguments:
    task: str = "CartPole-v1"
    max_episode_length: int = 1000


@dataclass
class LoggingArguments:
    wandb_enabled: bool = True
    project_name: str = "envquest"
    exp_id: str = None
    render_width: int = 256
    render_height: int = 256
    save_train_videos: bool = False
    save_eval_videos: bool = True
    log_eval_videos: bool = False
    save_agent_snapshots: bool = True


@dataclass
class TrainerArguments:
    # Training
    batch_size: int = 64
    num_train_steps: int = 1000000
    num_seed_steps: int = 5000
    num_updates: int = 1
    update_every_steps: int = 1

    # Evaluation
    num_eval_episodes: int = 5
    eval_every_steps: int = 10000


@dataclass
class MCTrainerArguments(TrainerArguments):
    num_train_trajectories: int = 5


@dataclass
class AgentArguments:
    class_name: str = None
    discount: float = 0.99
    lr: float = 1e-4
    mem_capacity: int = 1000000


@dataclass
class ContinuousPGAgentArguments(AgentArguments):
    noise_std_start: float = 0.3
    noise_std_end: float = 0.1
    noise_std_decay: str = DecayType.LINEAR
    noise_std_step_duration: int = 1000000


@dataclass
class PPOAgentArguments(AgentArguments):
    class_name: str = "ppo"
    clip_threshold: float = 0.1
    num_policy_updates: int = 5


@dataclass
class ContinuousPPOAgentArguments(PPOAgentArguments, ContinuousPGAgentArguments):
    pass


@dataclass
class DQNAgentArguments(AgentArguments):
    class_name: str = "dqn"
    n_steps: int = 6
    tau: float = 0.005

    greedy_eps_start: float = 0.95
    greedy_eps_end: float = 0.05
    greedy_eps_step_duration: int = 100000
    greedy_eps_decay: str = DecayType.EXPONENTIAL  # "linear" or "exponential"


@dataclass
class SarsaAgentArguments(AgentArguments):
    class_name: str = "sarsa"

    greedy_eps_start: float = 0.95
    greedy_eps_end: float = 0.05
    greedy_eps_step_duration: int = 100000
    greedy_eps_decay: str = DecayType.EXPONENTIAL  # "linear" or "exponential"


@dataclass
class TrainingArguments:
    env: EnvArguments = EnvArguments()
    agent: AgentArguments = None
    trainer: TrainerArguments = TrainerArguments()
    logging: LoggingArguments = LoggingArguments()
