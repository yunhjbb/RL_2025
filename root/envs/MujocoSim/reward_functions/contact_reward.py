import mujoco

class ContactReward():
    def __init__(self, config):
        pass

    def get_reward(self, model, data, step, additional_data = None):
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            if ("ball" in (name1, name2)) and ("plate" in (name1, name2)):
                return 10.0  # plate와 접촉 중이면 보상

        return -1  # 아니면 보상 없음
