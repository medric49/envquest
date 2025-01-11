import abc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import wandb

from envquest.agents.common import Agent
from envquest.arguments import TrainingArguments
from envquest.envs.common import Environment
from envquest.recorders import EpisodeRecorder


class Trainer(metaclass=abc.ABCMeta):
    def __init__(self, env: Environment, agent: Agent, arguments: TrainingArguments, output_dir=".exp"):
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
        self.exp_dir = f"{output_dir}/{exp_id}"
        Path(self.exp_dir).mkdir(parents=True)

        self.train_recorder = EpisodeRecorder(f"{self.exp_dir}/train_video")
        self.eval_recorder = EpisodeRecorder(f"{self.exp_dir}/eval_video")

        self.train_step = 0
        self.train_episode = 0

    def run_eval_episode(self, video_name):
        timestep = self.env.reset()
        agent_return = timestep.reward

        if self.arguments.logging.save_eval_videos and self.env.is_renderable:
            self.eval_recorder.start_recording(
                self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
            )

        frames = None
        while not timestep.last():
            action = self.agent.act(observation=timestep.observation)
            timestep = self.env.step(action)
            agent_return += timestep.reward

            if self.arguments.logging.save_eval_videos and self.env.is_renderable:
                self.eval_recorder.record(
                    self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
                )
        if self.arguments.logging.save_eval_videos and self.env.is_renderable:
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
        if self.arguments.logging.wandb_enabled:
            wandb.log(metrics, step=self.train_step)

        if self.arguments.logging.log_eval_videos and len(frames_list) > 0:
            frames_list = [np.asarray(frames, dtype=np.uint8).transpose((0, 3, 1, 2)) for frames in frames_list]
            frames_list = np.concatenate(frames_list)

            if self.arguments.logging.wandb_enabled:
                wandb.log({"eval/demo": wandb.Video(frames_list, fps=20)}, step=self.train_step)

    def save(self):
        artifact_path = self.exp_dir + "/snapshot.pt"
        snapshot = {"agent": self.agent}
        with open(artifact_path, "wb") as f:
            torch.save(snapshot, f)

        if self.arguments.logging.wandb_enabled:
            wandb_artifact = wandb.Artifact(self.exp_id, type="model")
            wandb_artifact.add_file(artifact_path)
            wandb.log_artifact(wandb_artifact, aliases=["latest", f"step_{self.train_step}"])

    @abc.abstractmethod
    def train(self):
        pass
