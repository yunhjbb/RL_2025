import numpy as np

class ActionPenaltyReward():
    def __init__(self, config):
        pass

    def get_reward(self, model, data, step, additional_data = None):
        action = data.ctrl[:].copy()  # 현재 액션 추출
        return - 0.01 * np.linalg.norm(action)