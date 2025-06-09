import numpy as np
class Explode():
    def __init__(self, config):
        self.max_step = 1000
        pass

    def get_termination(self, model, data, current_step, additional_data = None):
        obs = np.concatenate([data.qpos, data.qvel]).ravel()
        done = bool(
            np.abs(obs[1]) > 1.0 or
            np.abs(obs[0]) > 89
        )
        if done:
            print("Explode! ", end = "")
        return done
