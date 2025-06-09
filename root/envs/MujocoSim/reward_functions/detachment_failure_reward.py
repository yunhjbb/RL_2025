import numpy as np


class DetachmentFailureReward():
    
    def __init__(self, config):
        self.max_steps = config['max_steps']

    def get_reward(self, model, data, step, additional_data = None):
        if step == self.max_steps:
            return -1
        return 0

