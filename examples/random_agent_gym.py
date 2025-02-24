import time

import cv2
import fire
import numpy as np

from envquest import envs, agents


def main(env_name="LunarLander-v3", im_w=512, im_h=512, fps=30):
    env = envs.gym.GymEnvironment.from_task(env_name)
    agent = agents.generics.RandomAgent(env.observation_space, env.action_space)

    timestep = env.reset()
    while True:
        frame = env.render(im_w, im_h)
        frame = np.asarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", frame)

        time.sleep(1 / fps)
        action = agent.act()
        timestep = env.step(action)
        if cv2.waitKey(1) == ord("q"):
            break

        if timestep.last():
            timestep = env.reset()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(main)
