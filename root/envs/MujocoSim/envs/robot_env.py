import numpy as np
from typing import Tuple, Dict
from gym import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step
import importlib
import sys, os

# project 루트 경로를 PYTHONPATH에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(f"PRJ ROOT, {PROJECT_ROOT}")
def resolve_class(name: str, module_path: str):
    module = importlib.import_module(module_path)
    return getattr(module, name)

class SimpleMuJoCoEnv:
    def __init__(self, xml_path: str, config):
        """
        단일 에이전트 MuJoCo 환경 초기화

        Args:
            xml_path (str): MuJoCo XML 파일 경로
            config: 리워드 및 종료 조건에 전달할 설정 객체
            max_steps (int): 에피소드 최대 스텝 수
        """
        self.config = config
        self.max_steps = config.get("max_steps", 100000)
        self.current_step = 0
        self.use_custom_obs = config.get("use_custom_obs", False)
        self.custom_obs_idx = config.get("custom_obs_idx", 1)
        self.use_custom_act = config.get("use_custom_act", False)
        self.custom_act_idx = config.get("custom_act_idx", [1,2,3,4,5,6,7])
        if self.use_custom_obs:
            print(f"USING CUSTOM OBSERVATION SPACE!, idx : {self.custom_obs_idx}")
        if self.use_custom_act:
            print(f"USING CUSTOM ACTION SPACE!")

        # MuJoCo 모델 및 데이터 초기화
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        print("actuator control range")
        print(self.model.actuator_ctrlrange)
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_id = self.model.jnt_qposadr[i]
            qvel_id = self.model.jnt_dofadr[i]
            

        # reward나 done을 판단할 때 model과 data만으로 부족한 경우 쓰세요.
        self.additional_data = None

        # 관찰 및 행동 공간 설정
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_sample.shape, dtype=np.float32
        )
        # action 커스텀
        if self.use_custom_act:
            action_shape = len(self.custom_act_idx)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(action_shape,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32
        )
        # 리워드 구성 (가중치 포함)
        self.reward_functions = []
        for name, weight in config.get("rewards", {}).items():
            cls = resolve_class(name, "envs.MujocoSim.reward_functions")
            self.reward_functions.append((weight, cls(config)))

        # 종료 조건 구성
        self.termination_conditions = []
        for name, enabled in config.get("termination_conditions", {}).items():
            if enabled:
                cls = resolve_class(name, "envs.MujocoSim.termination_conditions")
                self.termination_conditions.append(cls(config))

    def reset(self) -> Tuple[np.ndarray]:
        """
        환경을 초기화하고 초기 관찰 반환
        Returns:
            obs, share_obs: 둘 다 동일하게 qpos+qvel
        """
        self.current_step = 0
        self.data = MjData(self.model)  # 데이터 리셋
        # mj_step(self.model, self.data)  # 초기 상태 반영
        if self.use_custom_obs:
            # custom_pose로 초기화
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "custom_pose")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            mujoco.mj_forward(self.model, self.data)

            # initial distance 계산 (terminal reward에 사용하기 때문에)
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
            site_pos = self.data.site_xpos[site_id]
            target_pos = np.array(self.config.get("Match_pos", [0.6, 0.0, 0.5]))
            self.initial_target_distance = np.linalg.norm(site_pos - target_pos)

        for _, r in self.reward_functions:  # 보상함수에 저장된 값들을 초기화 해야함
            if hasattr(r, "reset"):
                r.reset()
        
        for cond in self.termination_conditions:  # 터미널널함수에 저장된 값들을 초기화 해야함
            if hasattr(cond, "reset"):
                cond.reset()

        
        obs = self._get_obs()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        환경을 한 스텝 진행

        Args:
            action: 에이전트의 행동

        Returns:
            obs, share_obs, reward, done, info
        """
        self.current_step += 1

        # actuator 범위 가져오기 (MuJoCo에 세팅된 (xml 파일에서 정의한) ctrlrange)
        ctrlrange = self.model.actuator_ctrlrange  # shape: (nu, 2)
        action_low = ctrlrange[:, 0]
        action_high = ctrlrange[:, 1]
        if self.use_custom_act:
            new_action = np.zeros(self.model.nu)
            for idx in self.custom_act_idx:
                new_action[idx-1] = action[idx-1] # 잠기지 않은 부분만 action에 집어넣음.
            scaled_action = action_low + 0.5 * (new_action + 1.0) * (action_high - action_low)
        else:
            scaled_action = action_low + 0.5 * (action + 1.0) * (action_high - action_low)


        # MuJoCo에 적용
        self.data.ctrl[:] = scaled_action
        mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()                
        info = {"current_step": self.current_step}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """관찰값 생성: qpos + qvel"""
        if self.use_custom_obs == True:
            return self._get_custom_obs(self.model, self.data, self.custom_obs_idx, self.config)
        else:
            return np.concatenate([self.data.qpos, self.data.qvel], axis=0)

    
    @staticmethod
    def _get_custom_obs(model, data, obs_idx, custom_config = {"custom_data" : [0.6, 0.0, 0.5]}, ):
        if obs_idx == 1:
            ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
            ee_pos = data.site_xpos[ee_site_id]
            body_id = model.site_bodyid[ee_site_id]  # site가 속한 body의 선속도 사용
            ee_vel = data.cvel[body_id][:3]
            target_pos = np.array(custom_config.get("custom_data", [0.6, 0.0, 0.5]))
            return np.concatenate([data.qpos, data.qvel, ee_pos, ee_vel, target_pos], axis=0)
        elif obs_idx == 2:
            obs = np.concatenate([data.qpos[:7], data.qvel[:7]], axis=0)
            return obs
        elif obs_idx == 3:
            robot_pos = data.qpos[:7]
            ball_pos = data.qpos[7:]
            robot_vel = data.qvel[:7]
            ball_vel = data.qvel[7:]
            ball_vel = ball_vel * (1.0/100) # scaling
            return np.concatenate([robot_pos, ball_pos, robot_vel, ball_vel], axis=0)
        elif obs_idx == 4:
            new_qvel = data.qvel * (1.0/10) # scaling
            return np.concatenate([data.qpos[:7], new_qvel[:7]], axis=0) # scaling, 그리고 robot 커팅
        else:
            # 2번, 3번 custom_obs가 필요한 경우 여기에 작성하면 됩니다.
            return NotImplementedError

    def _compute_reward(self) -> float:
        """모든 리워드 함수의 합산"""
        return sum(weight * r.get_reward(self.model, self.data, self.current_step, self.additional_data)
               for weight, r in self.reward_functions)

    def _check_done(self) -> bool:
        """종료 조건 만족 여부 판단"""
        return any(cond.get_termination(self.model, self.data, self.current_step, self.additional_data) for cond in self.termination_conditions)
