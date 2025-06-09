import numpy as np
import mujoco

class SphericalBoundaryTermination():
    def __init__(self, config):
        self.target_pos = np.array(config["Match_pos"])
        self.initial_pos = None
        self.center = None
        self.radius = None
        self.initialized = False

    def get_termination(self, model, data, current_step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]  # 절대 위치

        if not self.initialized:
            self.initial_pos = np.copy(site_pos)
            self.center = (self.initial_pos + self.target_pos) / 2
            self.radius = np.linalg.norm(self.initial_pos - self.target_pos) / 2
            self.initialized = True

        dist = np.linalg.norm(site_pos - self.center)
        if dist > self.radius + 0.01:
            print("Out of boundary", end=" ")
            return True

        return False

    def reset(self):
        self.initialized = False
