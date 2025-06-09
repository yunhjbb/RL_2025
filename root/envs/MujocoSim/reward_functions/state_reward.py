import numpy as np
import mujoco

class StateReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))  # 목표 위치
        self.coeff = None
        self.init_distance = None  # 초기 거리 저장용

    def get_reward(self, model, data, step, additional_data = None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        current_pos = data.site_xpos[site_id]

        # 현재 거리 (타겟까지)
        curr_dist = np.linalg.norm(current_pos - self.target_pos)

        # 최초 1회 초기 거리 저장
        if self.init_distance is None:
            self.init_distance = curr_dist
            self.coeff = 5 * 4 / (self.init_distance ** 3)
            return 0.0

        # 보상: 초기보다 더 가까우면 양의 보상, 멀어지면 음의 보상
        reward = self.coeff * ((self.init_distance - curr_dist) ** 3)
        # print(f"[Step {step}] StateReward: {reward:.4f} (dist: {curr_dist:.4f})")
        return reward
    
    def reset(self):
        self.coeff = None
        self.init_distance = None
