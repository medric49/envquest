from dataclasses import asdict

from tqdm import tqdm
import wandb

from envquest import utils
from envquest.trainers.common import Trainer


class TDTrainer(Trainer):

    def train(self):
        if self.arguments.logging.wandb_enabled:
            wandb.init(project=self.arguments.logging.project_name, name=self.exp_id, config=asdict(self.arguments))

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

                if self.arguments.logging.save_train_videos and self.env.is_renderable:
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
                        if self.arguments.logging.wandb_enabled:
                            wandb.log(metrics, step=self.train_step)

                    # Compute agent return
                    agent_return += timestep.reward

                    # Record step
                    if self.arguments.logging.save_train_videos and self.env.is_renderable:
                        self.train_recorder.record(
                            self.env.render(self.arguments.logging.render_width, self.arguments.logging.render_height)
                        )

                    # Log return and training video
                    if timestep.last():
                        metrics = {"train/eps_length": self.env.episode_length, "train/return": agent_return}
                        if self.arguments.logging.wandb_enabled:
                            wandb.log(metrics, step=self.train_step)

                        if self.arguments.logging.save_train_videos and self.env.is_renderable:
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
