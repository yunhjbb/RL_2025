import numpy as np
import mujoco
class ContactOff():
    def __init__(self, config):
        self.config = config
        pass

    def get_termination(self, model, data, current_step, additional_data = None):
        # flag = False # 지금 공은 판과 붙어 있는가?
        # done = False
        # for i in range(data.ncon):  # 현재 발생한 접촉 개수만큼 반복
        #     contact = data.contact[i]
        #     g1 = contact.geom1
        #     g2 = contact.geom2

        #     name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        #     name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        #     if ("ball" in (name1, name2)) and ("plate" in (name1, name2)):
        #         flag = True

        # if flag == False:
        #     print("Contact Off! ", end = "")
        #     done = True
            # plate 중심 및 회전 정보
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

        # plate → ball 벡터
        vec_to_ball = ball_pos - ee_pos

        # ball의 반지름 가져오기
        ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        ball_radius = model.geom_size[ball_geom_id][0]

        # 수직 거리 계산 (중심 간 거리)
        normal_distance = np.abs(np.dot(vec_to_ball, normal_unit))

        # 판정 임계값: 반지름 + epsilon
        CONTACT_EPSILON = 0.07  # 추가 허용 거리 (3mm 등)
        CONTACT_THRESHOLD = ball_radius + CONTACT_EPSILON

        # top-view 위에 있는지 판단
        x_proj = np.dot(vec_to_ball, x_axis)
        y_proj = np.dot(vec_to_ball, y_axis)
        TOPVIEW_EPSILON = 0.03
        in_topview = (-0.06 - ball_radius -TOPVIEW_EPSILON <= x_proj <= 0.06 + ball_radius+TOPVIEW_EPSILON) and (-0.06 - ball_radius-TOPVIEW_EPSILON<= y_proj <= 0.06 + ball_radius+TOPVIEW_EPSILON)

        # 접촉 유사 상태인지 판단
        contact_like = (0 <= normal_distance <= CONTACT_THRESHOLD) and in_topview

        if contact_like:
            done = False
        else:
            if self.config != None:
                print("Contact Off! ", end = "")
            done = True
        return done
