import mujoco
class RobotFloorContact():
    def __init__(self, config):
        self.max_step = config.get("max_steps", 10000)
        pass

    def get_termination(self, model, data, current_step, additional_data = None):
        flag = False # 지금 공은 판과 붙어 있는가?
        done = False
        for i in range(data.ncon):  # 현재 발생한 접촉 개수만큼 반복
            contact = data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            if 'floor' in [name1, name2] and 'ball' not in [name1, name2]:
                flag = True

        if flag == True:
            print("Fall in Floor ", end = "")
            done = True
        return done
