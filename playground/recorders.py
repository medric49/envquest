from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL.Image import Image


class EpisodeRecorder:
    def __init__(self, save_dir, fps=20):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.frames = None
        self.fps = fps

    def start_recording(self, frame: Image):
        self.frames = [frame]

    def record(self, frame: Image):
        self.frames.append(frame)

    def save(self, filename):
        path = self.save_dir / filename

        frames = []
        for frame in self.frames:
            frame = np.asarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        imageio.mimsave(str(path), frames, fps=self.fps)
        return self.frames
