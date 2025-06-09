import numpy as np
import mujoco
import sys
import os

# B/ 디렉토리를 포함하는 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from termination_conditions.contactoff import ContactOff
class ShootZReward():
    
    def __init__(self, config):
        self.match_vel_bin = config.get("Match_vel_bin", False)
        self.match_vel = config.get("Match_vel", [0,0,0])
        self.match_pos_bin = config.get("Match_pos_bin", False)
        self.match_pos = config.get("Match_pos", [0,0,0])
        self.tolerance = config.get("target_tolerance", 0.01)  # 허용 오차
        pass

    def get_reward(self, model, data, step, additional_data = None):
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        ee_pos = data.site_xpos[ee_site_id]
        rot_mat = data.site_xmat[ee_site_id].reshape(3, 3)

        # 법선 벡터 (plate의 z축)
        normal_vec = rot_mat[:, 2]
        normal_unit = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)

        # local 좌표축들
        x_axis = rot_mat[:, 0]  # plate의 local x축
        y_axis = rot_mat[:, 1]  # plate의 local y축

        # ball 위치
        ball_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        ball_pos = data.site_xpos[ball_site_id]
        ball_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ball_site_id, ball_vel, 0)
        ball_vel_world = ball_vel[:3]
        # plate → ball 벡터
        vec_to_ball = ball_pos - ee_pos

        # ball의 반지름 가져오기
        ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        ball_radius = model.geom_size[ball_geom_id][0]

        # 수직 거리 계산 (중심 간 거리)
        normal_distance = np.abs(np.dot(vec_to_ball, normal_unit))



        obj = ContactOff(None)
        if obj.get_termination(model, data, step, additional_data):
            return np.clip(ball_vel_world[2], -5, 10) * 3


        return 0

