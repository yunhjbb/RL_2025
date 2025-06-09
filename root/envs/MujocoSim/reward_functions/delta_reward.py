import numpy as np
import mujoco

class DeltaDistanceReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))
        self.prev_distance = None

    def get_reward(self, model, data, step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]
        curr_dist = np.linalg.norm(site_pos - self.target_pos)

        # 첫 스텝은 기준이 없어 보상 없음
        if self.prev_distance is None:
            self.prev_distance = curr_dist
            return 0.0

        # 거리 변화 기반 보상
        delta = self.prev_distance - curr_dist  # +면 가까워짐, -면 멀어짐
        if delta > 0:
            reward = 1
        elif delta < 0:
            reward = -1
        else:
            reward = 0

        self.prev_distance = curr_dist
        return reward

    def reset(self):
        self.prev_distance = None
