import time

import cv2
import fire
import numpy as np

from envquest import envs, agents


def main(env_name="LunarLander-v3", im_w=512, im_h=512, fps=30):
    env = envs.gym.make_env(env_name)
    agent = agents.simple.RandomAgent(env.observation_space, env.action_space)

    timestep = env.reset()
    while not timestep.last():
        frame = env.render(im_w, im_h)
        frame = np.asarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", frame)

        time.sleep(1 / fps)
        action = agent.act()
        timestep = env.step(action)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(main)
