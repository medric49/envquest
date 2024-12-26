from dataclasses import dataclass


@dataclass
class TrainingArguments:
    # General
    task: str = "LunarLander-v3"
    max_episode_length: int = 500

    # Trainer
    num_train_steps: int = 100000
    num_seed_steps: int = 5000
    num_updates: int = 1
    update_every_steps: int = 1

    # Agent
    n_steps: int = 1
    mem_capacity: int = 100000
    discount: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    tau: float = 0.005
    eps_start: float = 0.95
    eps_end: float = 0.05
    eps_step_duration: int = 20000

    # Evaluation
    num_eval_episodes: int = 5
    eval_every_steps: int = 2000

    # Logging
    project_name: str = "playground"
    exp_id: str = None
    render_width: int = 256
    render_height: int = 256
    save_train_videos: bool = False
    log_eval_videos: bool = True
    save_eval_videos: bool = True
