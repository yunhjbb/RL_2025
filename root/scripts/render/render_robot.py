import sys
import os
target_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(target_path)
from runner.mujoco_runner import SimpleRunner  
from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer
from envs.MujocoSim.envs.robot_env import SimpleMuJoCoEnv
from config import get_config
from mujoco import MjModel, MjData, mj_step, viewer
import traceback
import torch
import numpy as np
import mujoco
import yaml
import random

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print(all_args)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda")  
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    # env init
    config = load_config(all_args.env_config_path)
    print(config)
    xml_path = all_args.xml_path
    max_steps = all_args.max_steps
    model = MjModel.from_xml_path(xml_path)
    data = MjData(model)


    use_custom_obs = config.get("use_custom_obs", False)
    custom_obs_idx = config.get("custom_obs_idx", 1)
    use_custom_act = config.get("use_custom_act", False)
    custom_act_idx = config.get("custom_act_idx", [1,2,3,4,5,6,7])

    # custom_obs 적용 / obs length 가변화
    if use_custom_obs:
        obs_sample = SimpleMuJoCoEnv._get_custom_obs(model, data, custom_obs_idx, config)
        obs_length = obs_sample.shape[0]
        print(f"USING CUSTOM OBSERVATION SPACE!, idx : {custom_obs_idx}")
    else:
        obs_length = np.size(data.qpos) + np.size(data.qvel)
    print(f"OBS LENGTH : {obs_length}")
    
    # custom_act 적용 / act length 가변화
    if use_custom_act:
        act_shape = len(custom_act_idx)
        print(f"USING CUSTOM OBSERVATION SPACE!, idx : {custom_act_idx}")
    else:
        act_shape = model.nu
    
    # custom_pose 초기화 적용
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "custom_pose")
    # 등록
    target_vel = config.get("Match_vel", None)
    use_vel = config.get("Match_vel_bin", False)

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)  # ← 이 줄 추가 (반드시 reset 다음에 와야 함)
        print("[INFO] Applied keyframe 'custom_pose'")
    else:
        print("[WARNING] Keyframe 'custom_pose' not found.")
        
    print("PRESS CTRL+C at TERMINAL TO EXIT...")
    if all_args.act_by_model == True:
        from algorithms.ppo.ppo_policy import PPOActor
        from algorithms.ddpg.ddpg_actor import DDPGActor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if all_args.algorithm_name == 'ppo':
            policy = PPOActor(obs_length, act_shape, hidden_sizes=all_args.hidden_size_actor).to(device)
        elif all_args.algorithm_name == 'ddpg':
            print(obs_length)
            print(act_shape)
            print(all_args.hidden_size_actor)
            policy = DDPGActor(obs_length, act_shape, hidden_sizes=all_args.hidden_size_actor).to(device)
        policy.load_state_dict(torch.load(all_args.render_model_path, map_location=device, weights_only=True))
        policy.eval()
        with viewer.launch_passive(model, data) as v:
            t = 0
            detach = False
            floor_robot_contact = False
            last_time = data.time
            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
            use_ball = (ball_body_id != -1 and ball_geom_id != -1)
            while True:
                # Reset 감지: time이 줄어들었거나 0으로 돌아가면
                if data.time < last_time:
                    t = 0
                    detach = False

                last_time = data.time
                # ball_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ball")
                # ball_pos = data.site_xpos[ball_site_id]
                # ball_vel = np.zeros(6)
                # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, ball_site_id, ball_vel, 0)
                # ball_vel_world = ball_vel[:3]
                # # if t % 10 == 0:
                # #     print(f"vel : {ball_vel_world[2]}, pos : {ball_pos}")
                # # 초기 접촉 여부 확인
                # if t == 0 and use_ball:
                #     contact_found = False
                #     for i in range(data.ncon):
                #         contact = data.contact[i]
                #         name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                #         name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                #         if 'ball' in [name1, name2] and 'floor' not in [name1, name2]:
                #             contact_found = True
                #             break

                with torch.no_grad():
                    if use_custom_obs:
                        obs = SimpleMuJoCoEnv._get_custom_obs(model, data, custom_obs_idx, config)
                    else:
                        obs = np.concatenate([data.qpos, data.qvel], axis=0)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    if all_args.algorithm_name == 'ppo':
                        action, _ = policy(obs_tensor, deterministic = all_args.deterministic)
                    elif all_args.algorithm_name == 'ddpg':
                        action = policy(obs_tensor)
                    action = action.cpu().numpy().squeeze(0)
                        # actuator 범위 가져오기 (MuJoCo에 세팅된 (xml 파일에서 정의한) ctrlrange)
                ctrlrange = model.actuator_ctrlrange  # shape: (nu, 2)
                action_low = ctrlrange[:, 0]
                action_high = ctrlrange[:, 1]
                # 정규화된 [-1, 1] 행동(actor의 출력)을 실제 물리 범위로 스케일링

                if use_custom_act:
                    new_action = np.zeros(model.nu)
                    for idx in custom_act_idx:
                        new_action[idx-1] = action[idx-1] # 잠기지 않은 부분만 action에 집어넣음.
                    scaled_action = action_low + 0.5 * (new_action + 1.0) * (action_high - action_low)
                else:
                    scaled_action = action_low + 0.5 * (action + 1.0) * (action_high - action_low)


                if t > 0:
                    data.ctrl[:] = scaled_action
                    mj_step(model, data)
                else:
                    data.ctrl[:] = np.zeros_like(data.ctrl)
                    mj_step(model, data)
                if use_ball:
                    # 접촉 여부 판단
                    flag = False
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                        if 'ball' in [name1, name2] and 'floor' not in [name1, name2]:
                            flag = False
                        if 'floor' in [name1, name2] and 'ball' not in [name1, name2] and not floor_robot_contact:
                            floor_robot_contact = True
                            other = name1 if name2 == 'floor' else name2
                    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                    ball_qvel_adr = model.body_dofadr[ball_body_id]
                    ball_pos = data.xpos[ball_body_id]
                    lin_vel = data.qvel[ball_qvel_adr : ball_qvel_adr+3]

                    if not detach and not flag and t != 0:
                        detach = True

                v.sync()
                t += 1

    
    else:
        print("NO MODEL LOADED")
        with viewer.launch_passive(model, data) as v:
            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
            use_ball = (ball_body_id != -1 and ball_geom_id != -1)           
            while(True):
                for _ in range(1000):
                    mj_step(model, data)
                    flag = False
                    if use_ball:
                        for i in range(data.ncon):  # 현재 발생한 접촉 개수만큼 반복
                            contact = data.contact[i]
                            g1 = contact.geom1
                            g2 = contact.geom2
                            ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
                            floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                            ball_qvel_adr = model.body_dofadr[ball_body_id]  # 자유도 인덱스 시작점

                            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                            if name1== 'ball' or name2 == 'ball':
                                if name1 != 'floor' and name2 != 'floor':
                                    flag = True

                        if flag == False:
                            lin_vel = data.qvel[ball_qvel_adr : ball_qvel_adr+3]
                            ang_vel = data.qvel[ball_qvel_adr+3 : ball_qvel_adr+6]
                            # print(f"Linear velocity:  {lin_vel}")
                    v.sync()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])