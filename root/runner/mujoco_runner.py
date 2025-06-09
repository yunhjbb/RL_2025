import time
import torch
import numpy as np
import logging
import os
import csv

class SimpleRunner:
    def __init__(self, env, args, device=torch.device("cpu")):
        self.env = env
        self.args = args
        self.device = device

        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.hidden_shape_actor = self.args.hidden_size_actor
        self.hidden_shape_critic = self.args.hidden_size_critic


        self.eval_interval = getattr(args, "eval_interval")
        self.total_steps = 0
        if self.args.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
            self.policy = Policy(self.obs_shape[0], self.act_shape[0], self.args, device=self.device)
            self.trainer = Trainer(self.policy, args)
        elif self.args.algorithm_name == "ddpg":
            from algorithms.ddpg.ddpg_actor import DDPGActor
            from algorithms.ddpg.ddpg_critic import DDPGCritic
            from algorithms.ddpg.ddpg_trainer import DDPGTrainer as Trainer
            from algorithms.ddpg.ddpg_policy import DDPGPolicy as Policy
            from algorithms.ddpg.noise import OUNoise
            from algorithms.ddpg.ddpg_replaybuffer import ReplayBuffer
            self.policy = Policy(
                obs_dim=self.obs_shape[0],
                act_dim=self.act_shape[0],
                actor_cls=DDPGActor,
                critic_cls=DDPGCritic,
                args=self.args,
                device=self.device,
            )
            self.trainer = Trainer(
                policy=self.policy,
                args=self.args,
                device=self.device,
            )
            self.noise = OUNoise(self.act_shape[0])
            self.replay_buffer = ReplayBuffer(capacity = 10000)
        else:
            raise NotImplementedError
        
        if self.args.continual_learning == True:
            print("CONTINUAL LEARNING")
            print(f"ACTOR : {self.args.load_actor_path}")
            print(f"CRITIC : {self.args.load_critic_path}")
            self.policy.actor.load_state_dict(torch.load(self.args.load_actor_path, map_location=device, weights_only=True))
            self.policy.critic.load_state_dict(torch.load(self.args.load_critic_path, map_location=device, weights_only=True))


    def train(self, policy, obs, action):
        self.policy.eval()
        train_infos = self.trainer.train(obs, action)
        return train_infos
    def run(self):
        if self.args.algorithm_name == 'ddpg':
            self.run_ddpg()
        elif self.args.algorithm_name == 'ppo':
            self.run_ppo()  
    def run_ddpg(self):
        self.episode_rewards = []
        obs = self.env.reset()
        for episode in range(self.args.num_episodes):
            episode_reward = 0
            done = False
            self.noise.reset()  # exploration noise 초기화

            while not done:
                self.policy.eval()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy.get_action(obs_tensor)  # deterministic + noise 내부 적용
                action_np = action.cpu().numpy().squeeze(0)

                next_obs, reward, done, info = self.env.step(action_np)

                # Transition 저장
                self.replay_buffer.push(obs, action_np, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward
                self.total_steps += 1

                # 일정 스텝마다 학습
                if len(self.replay_buffer) >= self.args.batch_size and self.total_steps % self.args.update_every == 0:
                    result = self.trainer.train(self.replay_buffer)
                    #print(f"Step {self.total_steps}: loss_actor = {result['actor_loss']:.3f}, loss_critic = {result['critic_loss']:.3f}")

            self.episode_rewards.append(episode_reward)
            obs = self.env.reset()
            print(f"Episode {episode + 1} | done at step {info['current_step']}: reward {episode_reward:.2f}")
                # save model
            if (episode + 1) % self.args.save_every == 0:
                self.save(episode)
                try:
                    with open("numbers_temp.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["index", "value"])
                        writer.writerows(enumerate(self.episode_rewards))
                except PermissionError:
                    pass
    def run_ppo(self):
        obs_buffer, action_buffer, log_prob_buffer = [], [], []
        value_buffer, reward_buffer, return_buffer, adv_buffer = [], [], [], []
        self.episode_rewards = []
        self.full_rewards = []
        for episode in range(self.args.num_episodes):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            ep_obs, ep_actions, ep_log_probs = [], [], []
            ep_values, ep_rewards = [], []
            done = False
            episode_reward = 0

            while not done:
                self.policy.eval()
                with torch.no_grad():
                    value, action, log_prob = self.policy.get_actions(obs_tensor, deterministic = self.args.deterministic)

                action_np = action.cpu().numpy().squeeze(0)
                obs_next, reward, done, info = self.env.step(action_np)
                ep_obs.append(obs_tensor.squeeze(0))
                ep_actions.append(action.squeeze(0))
                ep_log_probs.append(log_prob.squeeze(0))
                ep_values.append(value.squeeze())
                ep_rewards.append(reward)

                episode_reward += reward
                obs_tensor = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0).to(self.device)
                self.total_steps += 1

            # Episode finished: compute returns/advantages
            R = 0
            returns, advantages = [], []
            for r, v in zip(reversed(ep_rewards), reversed(ep_values)):
                R = r + self.args.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            values = torch.stack(ep_values).detach()
            advantages = returns - values

            # Append episode data to buffers
            obs_buffer += ep_obs
            action_buffer += ep_actions
            log_prob_buffer += ep_log_probs
            value_buffer += ep_values
            reward_buffer += ep_rewards
            return_buffer += returns
            adv_buffer += advantages
            self.episode_rewards.append(episode_reward)

            if episode == 0:
                with open("episode0_timesteps.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestep", "observation", "action", "reward", "value", "log_prob"])
                    for t in range(len(ep_rewards)):
                        writer.writerow([
                            t,
                            ep_obs[t].cpu().numpy().tolist(),
                            ep_actions[t].cpu().numpy().tolist(),
                            ep_rewards[t],
                            ep_values[t].item(),
                            ep_log_probs[t].item()
                        ])
            # === Perform training after collecting enough episodes ===
            print(f"Episode {episode + 1} | done at step {info['current_step']}: reward {episode_reward:.2f}")
            if (episode + 1) % self.args.update_every == 0:
                self.trainer.train(
                    torch.stack(obs_buffer),
                    torch.stack(action_buffer),
                    torch.stack(log_prob_buffer),
                    torch.stack(return_buffer),
                    torch.stack(adv_buffer)
                )
                obs_buffer, action_buffer, log_prob_buffer = [], [], []
                value_buffer, reward_buffer, return_buffer, adv_buffer = [], [], [], []
                print(f"Training at episode {episode + 1}")

            # save model
            if (episode + 1) % self.args.save_every == 0:
                self.save(episode)
                try:
                    with open("numbers_temp.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["index", "value"])
                        writer.writerows(enumerate(self.episode_rewards))
                except PermissionError:
                    pass

    def save(self, episode):
        # 기본 경로들
        base_save_dir = str(self.args.save_dir)
        exp_subdir = os.path.join(base_save_dir, str(self.args.experiment_name))
        os.makedirs(exp_subdir, exist_ok=True)

        policy_actor = self.policy.actor
        policy_critic = self.policy.critic

        # 1. latest 파일 → 항상 최상위 save_dir에 저장
        torch.save(policy_actor.state_dict(), os.path.join(base_save_dir, "actor_latest.pt"))
        torch.save(policy_critic.state_dict(), os.path.join(base_save_dir, "critic_latest.pt"))

        # 2. 버전 저장 파일 → experiment_name 디렉토리 하위에 episode 번호 포함
        torch.save(policy_actor.state_dict(), os.path.join(exp_subdir, f"actor_ep{episode+1}.pt"))
        torch.save(policy_critic.state_dict(), os.path.join(exp_subdir, f"critic_ep{episode+1}.pt"))

        # 3. best_actor.pt 저장 (현재 에피소드 리워드 기준)
        if not hasattr(self, "best_return"):
            self.best_return = -float("inf")
        if len(self.episode_rewards) > 0:
            recent_reward = self.episode_rewards[-1]
            if recent_reward > self.best_return:
                self.best_return = recent_reward
                torch.save(policy_actor.state_dict(), os.path.join(base_save_dir, "best_actor.pt"))
                torch.save(policy_critic.state_dict(), os.path.join(base_save_dir, "best_critic.pt"))


