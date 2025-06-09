import numpy as np
import mujoco
import sys
import os

# B/ 디렉토리를 포함하는 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from termination_conditions.contactoff import ContactOff
class MatchReward():
    
    def __init__(self, config):
        self.match_vel_bin = config.get("Match_vel_bin", False)
        self.match_vel = config.get("Match_vel", [0,0,0])
        self.match_pos_bin = config.get("Match_pos_bin", False)
        self.match_pos = config.get("Match_pos", [0,0,0])
        self.tolerance = config.get("target_tolerance", 0.01)  # 허용 오차
        pass

    def get_reward(self, model, data, step, additional_data = None):
        # ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        # ball_qvel_adr = model.body_dofadr[ball_body_id]  # 자유도 인덱스 시작점
        # ball_pos = data.xpos[ball_body_id]  # (3,) array: x, y, z 위치
        # lin_vel = data.qvel[ball_qvel_adr : ball_qvel_adr+3]
        # flag = False # 지금 공은 판과 붙어 있는가?
        # for i in range(data.ncon):  # 현재 발생한 접촉 개수만큼 반복
        #     contact = data.contact[i]
        #     g1 = contact.geom1
        #     g2 = contact.geom2

        #     name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        #     name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

        #     if ("ball" in (name1, name2)) and ("plate" in (name1, name2)):
        #         flag = True

        '''
        if flag == True:
            dist = 0.0
            if self.match_vel_bin:
                dist += np.linalg.norm(lin_vel - np.array(self.match_vel))
            if self.match_pos_bin:
                dist += np.linalg.norm(ball_pos - np.array(self.match_pos))

            # reward = 10 * np.exp(- 1 * dist)  # 가까울수록 보상 1, 멀면 급격히 감소
            reward = 10 / (1 + 10 * dist)
            return reward
        ''' # direction reward로 대체
        total_dist = 0
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
        if self.match_vel_bin:
            total_dist += np.linalg.norm(ball_vel_world - self.match_vel)
        if self.match_pos_bin:
            total_dist += np.linalg.norm(ball_pos - self.match_pos)
        reward = np.exp(-total_dist/0.5) * 6
        #print(reward)
        return reward

