from mujoco import MjModel, MjData, viewer, mj_step

# 1. 모델 로드
model = MjModel.from_xml_path("panda_mujoco-master/world.xml")  # actuator 없는 모델
data = MjData(model)
print("qpos dim:", model.nq)
print("qvel dim:", model.nv)
print("ctrl dim:", model.nu)
def get_names(model, adr_array):
    names = []
    for i in range(len(adr_array)):
        start = adr_array[i]
        end = model.names.find(b'\0', start)
        names.append(model.names[start:end].decode('utf-8'))
    return names

# 예시:
actuator_names = get_names(model, model.name_actuatoradr)
joint_names = get_names(model, model.name_jntadr)
site_names = get_names(model, model.name_siteadr)
print("Actuator names:", actuator_names)
print("Joint names : ", joint_names)
print("Site names : ", site_names)
# 2. 단순히 보기만
with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # 아무 제어 입력 없이 한 스텝 진행
        mj_step(model, data)
