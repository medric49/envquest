from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import wandb

from playground import config, utils
from playground.agents.common import Agent
from playground.arguments import TrainingArguments
from playground.envs.common import Environment
from playground.recorders import EpisodeRecorder


class Trainer:
    def __init__(
        self,
        env: Environment,
        agent: Agent,
        arguments: TrainingArguments,
    ):
        self.env = env
        self.agent = agent
        self.arguments = arguments

        prefix = (
            f"{self.arguments.env.task.replace('/', '-')}"
            if self.arguments.logging.exp_id is None
            else f"{self.arguments.logging.exp_id}"
        )
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_id = self.arguments.agent.class_name + "_" + prefix + "_" + now

        self.exp_id = exp_id
        self.exp_dir = f"{config.EXP_ROOT_DIR}/{exp_id}"
        Path(self.exp_dir).mkdir(parents=True)

        self.train_recorder = EpisodeRecorder(f"{self.exp_dir}/train_video")
        self.eval_recorder = EpisodeRecorder(f"{self.exp_dir}/eval_video")

        self.train_step = 0
        self.train_episode = 0

    def run_eval_episode(self, video_name):
        timestep = self.env.reset()
        agent_return = timestep.reward

        if self.arguments.logging.save_eval_videos:
            self.eval_recorder.start_recording(
                self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
            )

        frames = None
        while not timestep.last():
            action = self.agent.act(observation=timestep.observation)
            timestep = self.env.step(action)
            agent_return += timestep.reward

            if self.arguments.logging.save_eval_videos:
                self.eval_recorder.record(
                    self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
                )
        if self.arguments.logging.save_eval_videos:
            frames = self.eval_recorder.save(video_name)
        return agent_return, frames

    def eval(self):
        agent_return_list = []
        frames_list = []

        for i in tqdm(range(self.arguments.trainer.num_eval_episodes), colour="green"):
            agent_return, frames = self.run_eval_episode(video_name=f"{self.train_episode}_{i}.mp4")
            if frames is not None:
                frames_list.append(frames)
            agent_return_list.append(agent_return)

        agent_return_mean = np.mean(agent_return_list)

        metrics = {
            "eval/return": agent_return_mean,
        }
        wandb.log(metrics, step=self.train_step)

        if self.arguments.logging.log_eval_videos and self.arguments.logging.save_eval_videos:
            frames_list = [np.asarray(frames, dtype=np.uint8).transpose((0, 3, 1, 2)) for frames in frames_list]
            frames_list = np.concatenate(frames_list)
            wandb.log({"eval/demo": wandb.Video(frames_list, fps=20)}, step=self.train_step)

    def save(self):
        artifact_path = self.exp_dir + "/snapshot.pt"
        snapshot = {"agent": self.agent}
        with open(artifact_path, "wb") as f:
            torch.save(snapshot, f)

        wandb_artifact = wandb.Artifact(self.exp_id, type="model")
        wandb_artifact.add_file(artifact_path)

        wandb.log_artifact(wandb_artifact, aliases=["latest", f"step_{self.train_step}"])

    def train(self):
        wandb.init(project=self.arguments.logging.project_name, name=self.exp_id)

        update_every_step = utils.Every(self.arguments.trainer.update_every_steps)
        eval_every_step = utils.Every(self.arguments.trainer.eval_every_steps)
        seed_until_step = utils.Until(self.arguments.trainer.num_seed_steps)
        eval_scheduled = False

        with tqdm(total=self.arguments.trainer.num_train_steps) as pbar:
            while self.train_step < self.arguments.trainer.num_train_steps:
                self.train_episode += 1
                pbar.set_postfix({"Episode": self.train_episode})

                # Start training episode
                timestep = self.env.reset()
                agent_return = timestep.reward

                if self.arguments.logging.save_train_videos:
                    self.train_recorder.start_recording(
                        self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
                    )

                while not timestep.last():
                    self.train_step += 1
                    pbar.update()

                    # Compute action
                    if seed_until_step(self.train_step):
                        action = self.agent.act(observation=timestep.observation, random=True)
                    else:
                        action = self.agent.act(observation=timestep.observation, noisy=True)
                        if hasattr(self.agent, "current_noise"):
                            wandb.log(
                                {"train/noise": self.agent.current_noise},
                                step=self.train_step,
                            )

                    # Execute step
                    prev_timestep = timestep
                    timestep = self.env.step(action)

                    # Memorize step
                    self.agent.memorize(prev_timestep, timestep)

                    # Improve agent
                    if not seed_until_step(self.train_step) and update_every_step(self.train_step):
                        for _ in range(self.arguments.trainer.num_updates):
                            metrics = self.agent.improve(
                                batch_size=self.arguments.trainer.batch_size,
                            )
                        wandb.log(metrics, step=self.train_step)

                    # Compute agent return
                    agent_return += timestep.reward

                    # Record step
                    if self.arguments.logging.save_train_videos:
                        self.train_recorder.record(
                            self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
                        )

                    # Log return and training video
                    if timestep.last():
                        metrics = {"train/eps_length": self.env.episode_length, "train/return": agent_return}
                        wandb.log(metrics, step=self.train_step)

                        if self.arguments.logging.save_train_videos:
                            self.train_recorder.save(
                                f"{self.train_episode}_{int(agent_return)}.mp4",
                            )

                    # Schedule evaluation
                    if eval_every_step(self.train_step) and not seed_until_step(self.train_step):
                        eval_scheduled = True

                # Start evaluation
                if eval_scheduled or self.train_step >= self.arguments.trainer.num_train_steps:
                    self.eval()
                    if self.arguments.logging.save_agent_snapshots:
                        self.save()
                    eval_scheduled = False
