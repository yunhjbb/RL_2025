import numpy as np

class ActionSmoothnessReward():
    def __init__(self, config):
        self.prev_action = None

    def reset(self):
        self.prev_action = None

    def get_reward(self, model, data, step, additional_data = None):
        action = data.ctrl[:].copy()  # 현재 액션 추출
        if self.prev_action is None:
            self.prev_action = np.copy(action)
            return 0.0
        delta = action - self.prev_action
        self.prev_action = np.copy(action)
        return - 0.01 * np.linalg.norm(delta)
