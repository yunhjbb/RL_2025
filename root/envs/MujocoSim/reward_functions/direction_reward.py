# envs/MujocoSim/reward_functions/direction_reward.py

import numpy as np
import mujoco

class DirectionReward:
    def __init__(self, config):
        # 목표 위치는 기존 config.yaml의 Match_pos 항목 재사용
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))
        self.prev_pos = None  # 이전 end-effector 위치 저장

    def get_reward(self, model, data, step, additional_data = None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]  # 절대 위치

        if self.prev_pos is None:
            self.prev_pos = np.copy(site_pos)
            return 0.0  # 첫 스텝은 이동 방향 없음

        dir_to_target = self.target_pos - self.prev_pos
        # dir_norm = np.linalg.norm(dir_to_target) + 1e-8
        # dir_to_target /= dir_norm  # 정규화, 타겟 방향으로 향하는지만 고려
        actual_movement = site_pos - self.prev_pos  # 정규화 X, 실제 이동 크기는 반영
        movement_norm = np.linalg.norm(actual_movement) + 1e-8  # 생각해보니 이걸 정규화 목적까지 거리를 남겨놔야
        actual_movement /= movement_norm  # 둘 다 정규화, 방향 맞게 가는지만

        reward = 10 * np.dot(dir_to_target, actual_movement)
        # print(f"[Step {step}] Reward: {reward:.4f}")  # 100배 정도 하면 될듯, 실제 움직임은 10000배

        self.prev_pos = np.copy(site_pos)
        return reward
    
    def reset(self):
        self.prev_pos = None
