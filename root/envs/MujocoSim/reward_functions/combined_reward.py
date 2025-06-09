import numpy as np
import mujoco

class CombinedReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))
        self.coeff = None
        self.prev_ball_pos = None
        self.init_distance = None

    def get_reward(self, model, data, step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]

        # 현재 거리 계산
        curr_dist = np.linalg.norm(site_pos - self.target_pos)

        # 초기 거리 및 계수 설정 (최초 1회)
        if self.init_distance is None:
            self.init_distance = curr_dist
            self.coeff = 5 * 4 / (self.init_distance ** 3)
            self.prev_ball_pos = np.copy(site_pos)
            return 0.0

        # --- Direction Reward ---
        dir_to_target = self.target_pos - site_pos
        dir_to_target /= (np.linalg.norm(dir_to_target) + 1e-8)

        actual_movement = site_pos - self.prev_ball_pos
        actual_movement /= (np.linalg.norm(actual_movement) + 1e-8)

        direction_reward = np.dot(dir_to_target, actual_movement)

        # --- State Reward Magnitude ---
        state_improvement = (self.init_distance - curr_dist)
        state_reward_magnitude = self.coeff * (state_improvement ** 3)

        # --- Final Combined Reward ---
        reward = direction_reward * state_reward_magnitude

        # 상태 업데이트
        self.prev_ball_pos = np.copy(site_pos)

        return reward
