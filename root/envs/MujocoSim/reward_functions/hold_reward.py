import numpy as np
import mujoco

class HoldReward():
    def __init__(self, config):
        self.initial_qpos = None

    def get_reward(self, model, data, step, additional_data = None):
        # 자동으로 한 번만 초기 상태 저장
        if self.initial_qpos is None:
            self.initial_qpos = np.array(data.qpos[:7])

        current_qpos = data.qpos[:7]
        deviation = np.linalg.norm(current_qpos - self.initial_qpos)

        return -deviation
