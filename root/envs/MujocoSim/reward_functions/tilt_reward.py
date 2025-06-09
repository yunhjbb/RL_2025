import numpy as np
import mujoco

class TiltReward:
    def __init__(self, config):
        self.target_pos = np.array(config.get("Match_pos", [0.6, 0, 0.5]))  # 목표 위치
        self.coeff = None
        self.init_distance = None  # 초기 거리 저장용
        self.reward_scale = 1

    def get_reward(self, model, data, step, additional_data = None):
        # # -------------------------------------
        # # ▶ Plate 회전 및 위치 정보
        # reward = 0
        # plate_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        # plate_center = data.site_xpos[plate_site_id]
        # rot_mat = data.site_xmat[plate_site_id].reshape(3, 3)
        # x_axis = rot_mat[:, 0]
        # y_axis = rot_mat[:, 1]
        # z_axis = rot_mat[:, 2]

        # # ▶ 방향 안전성 보정 (법선이 아래를 향하고 있다면 뒤집기)
        # if np.dot(z_axis, np.array([0, 0, 1])) < 0:
        #     z_axis *= -1
        #     x_axis *= -1
        #     y_axis *= -1

        # # -------------------------------------
        # # ▶ Ball 위치 및 속도
        # ball_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        # ball_pos = data.site_xpos[ball_site_id]
        # vec_to_ball = ball_pos - plate_center

        # # ▶ Ball 속도 (siteVelocity)
        # vel_ball = np.zeros(6)

        # # plate normal이 얼마나 world x/y 방향으로 기울어졌는가
        # tilt_x = np.dot(z_axis, np.array([1, 0, 0]))
        # tilt_y = np.dot(z_axis, np.array([0, 1, 0]))

        # # 공이 world x/y 방향으로 얼마나 치우쳐 있는가
        # offset_vec = ball_pos - plate_center
        # offset_x = np.dot(offset_vec, np.array([1, 0, 0]))
        # offset_y = np.dot(offset_vec, np.array([0, 1, 0]))

        # # 보상 조건: 기울기가 공의 치우친 방향과 반대라면 → 중심으로 돌아오려는 유도
        # if offset_x * tilt_x < 0:
        #     reward += 1
        # if offset_y * tilt_y < 0:
        #     reward += 1


        # return reward
        # ========== Plate pose & local frame ==========
        plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        plate_pos = data.site_xpos[plate_id]
        rot_mat = data.site_xmat[plate_id].reshape(3, 3)
        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]

        # ========== Ball pose ==========
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        ball_pos = data.site_xpos[ball_id]

        # ========== Ball velocity ==========
        ball_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ball_id, ball_vel, 0)
        ball_vel_world = ball_vel[:3]

        # ========== Compute relative info ==========
        vec_to_ball_world = ball_pos - plate_pos
        vec_to_ball_local = rot_mat.T @ vec_to_ball_world
        ball_vel_local = rot_mat.T @ ball_vel_world

        # ========== Escape check ==========
        # → 중심에서 바깥 방향으로 가는 중인지 (dot > 0)
        is_escaping = np.dot(vec_to_ball_local[:2], ball_vel_local[:2]) > 0

        reward = 0.0

        if is_escaping:
            # 단위 이탈 방향
            escape_dir = vec_to_ball_local[:2]
            escape_dir /= (np.linalg.norm(escape_dir) + 1e-8)

            # plate 기울기 방향 (world z축이 local frame에서 보이는 x/y방향)
            world_z = np.array([0, 0, 1])
            world_z_in_plate = rot_mat.T @ world_z
            tilt_vector = world_z_in_plate[:2]

            alignment = np.dot(escape_dir, tilt_vector)

            if alignment < 0:
                reward += self.reward_scale  # 역틸트 → 보상
            elif alignment > 0:
                reward -= self.reward_scale  # 탈출 촉진 → 감점

        return reward
    
    def reset(self):
        self.coeff = None
        self.init_distance = None
