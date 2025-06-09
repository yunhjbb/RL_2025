import numpy as np
import mujoco

class MultiLevelReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))
        self.init_distance = None
        self.levels = []

        self.relative_levels = [
            (0.05, 16.0, 1000.0),
            (0.10, 8.0, 1000.0),
            (0.25, 4.0, 1000.0),
            (0.50, 2.0, 1000.0),
            (1.00, 1.0, 0.0)
        ]

    def get_reward(self, model, data, step, additional_data=None):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]
        curr_dist = np.linalg.norm(site_pos - self.target_pos)

        if self.init_distance is None:
            self.init_distance = curr_dist
            self.levels = [
                {
                    "thresh": ratio * self.init_distance,
                    "reward": reward,
                    "entry_reward": entry_reward,
                    "activated": False
                }
                for ratio, reward, entry_reward in sorted(self.relative_levels, key=lambda x: x[0])
            ]
            return 0.0

        total_reward = 0.0

        for level in self.levels:
            thresh = level["thresh"]
            is_inside = curr_dist < thresh

            if is_inside:
                if not level["activated"]:
                    level["activated"] = True
                    total_reward += level["entry_reward"]
                else:
                    total_reward += level["reward"]
            # else:
            #     if level["activated"]:
            #         total_reward -= level["reward"]

        return total_reward

    def reset(self):
        self.init_distance = None
        self.levels = []
