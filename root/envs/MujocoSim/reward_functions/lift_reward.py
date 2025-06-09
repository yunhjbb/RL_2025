import numpy as np
import mujoco

class LiftReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))  # 목표 위치
        self.coeff = None
        self.init_distance = None  # 초기 거리 저장용

    def get_reward(self, model, data, step, additional_data = None):
        reward = 0
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        ee_pos = data.site_xpos[ee_site_id]
        rot_mat = data.site_xmat[ee_site_id].reshape(3, 3)

        # 법선 벡터 (plate의 z축)
        normal_vec = rot_mat[:, 2]
        normal_unit = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)

        # ball 위치
        ball_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        ball_pos = data.site_xpos[ball_site_id]

        # plate → ball 벡터
        vec_to_ball = ball_pos - ee_pos


        vel_ee = np.zeros(6)
        vel_ball = np.zeros(6)

        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ee_site_id, vel_ee, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ball_site_id, vel_ball, 0)

        # 상대 선속도
        relative_vel = vel_ball[:3] - vel_ee[:3]

        # plate 법선 방향으로의 상대 속도 성분
        normal_component = np.dot(relative_vel, normal_unit)

        if normal_component < -0.01:  # 임계값은 민감도에 따라 조정
            reward += 2  # 혹은 비례 보상: reward += -0.5 * normal_component
        return reward
    
    def reset(self):
        self.coeff = None
        self.init_distance = None
