import numpy as np
import mujoco
from mujoco import mjtObj, mj_objectVelocity

class VelocityReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))  # 목표 위치
        self.target_vel = np.array(config.get("Match_vel", [1, 0, 0])) # 목표 속도
        self.coeff = None
        self.init_distance = None  # 초기 거리 저장용
        self.position_tolerance = 0.05

    def get_reward(self, model, data, step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        current_pos = data.site_xpos[site_id]
        vel_6d = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mjtObj.mjOBJ_SITE, site_id, vel_6d, 0)
        curr_vel = vel_6d[:3]
        curr_angular_vel = vel_6d[3:]

        # 현재 거리 (타겟까지)
        curr_dist = np.linalg.norm(current_pos - self.target_pos)

        if curr_dist < self.position_tolerance:
            # 속도 보상 항: 속도 차이의 음수값, 너무 크면 -1.0로 클리핑
            vel_reward = -np.linalg.norm(curr_vel - self.target_vel) / (np.linalg.norm(self.target_vel) + 1e-8)
            vel_reward = np.clip(vel_reward, -1.0, 0.0)

            curr_vel_norm = np.linalg.norm(curr_vel) + 1e-8
            target_vel_norm = np.linalg.norm(self.target_vel) + 1e-8

            # cosine similarity: [-1, 1] → 클리핑 그대로 사용
            vel_dot = np.dot(curr_vel, self.target_vel) / (curr_vel_norm * target_vel_norm)
            vel_dot = np.clip(vel_dot, -1.0, 1.0)

            # 속도 크기 차이: 너무 크면 패널티 상한
            speed_error = abs(curr_vel_norm - target_vel_norm)
            speed_error = np.clip(speed_error, 0.0, 2.0)
        else:
            vel_reward = 0.0
            vel_dot = 0.0
            speed_error = 0.0

        offset = 7.0
        reward = (vel_reward + vel_dot - speed_error + offset) * 4
        return reward

    def reset(self):
        self.coeff = None
        self.init_distance = None
