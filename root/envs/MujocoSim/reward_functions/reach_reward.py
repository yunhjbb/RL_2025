import numpy as np
import mujoco

class ReachReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0.0, 0.5]))
        self.positive_reward = 5.0
        self.negative_penalty = -5.0

        self.activated = False  # 한 번이라도 도달했는지 여부
        self.tolerance = None   # 초기 거리 절반으로 나중에 설정
        self.init_distance = None

    def get_reward(self, model, data, step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]
        dist = np.linalg.norm(site_pos - self.target_pos)

        # 초기 거리 및 tolerance 설정
        if self.init_distance is None:
            self.init_distance = dist
            self.tolerance = self.init_distance * 0.5
            return 0.0  # 첫 스텝은 보상 없음

        # 아직 한 번도 도달하지 않았으면 보상 없음
        if not self.activated:
            if dist < self.tolerance:
                self.activated = True
                return self.positive_reward
            else:
                return 0.0

        # 한 번 도달한 이후엔 상태에 따라 보상/패널티
        if dist < self.tolerance:
            return self.positive_reward
        else:
            return self.negative_penalty
