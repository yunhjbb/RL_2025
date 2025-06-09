import numpy as np
import mujoco

class TargetReached():
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))
        self.tolerance = config.get("target_tolerance", 0.01)
        self.init_distance = None  # 초기 거리 저장용

    def get_termination(self, model, data, current_step, additional_data = None):
        done = False
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        site_pos = data.site_xpos[site_id]
        dist = np.linalg.norm(site_pos - self.target_pos)

        # 초기 거리 저장
        if self.init_distance is None:
            self.init_distance = dist

        # 목표에 도달하거나, 너무 멀어지면 종료
        if dist < self.tolerance:
            print(f"Target reached! Distance: {dist:.6f} < {self.tolerance} ", end="")
            done = True
        # elif dist > 1.5 * self.init_distance:
        #     print(f"Far from target! Distance: {dist:.6f} > 1.5 × {self.init_distance:.6f} ", end="")
        #     done = True

        return done
