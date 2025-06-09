# RL_2025
7-DOF Robotic Arm Non-Prehensile Throwing Control via Reinforcement Learning
This project trains a 7-DOF robotic manipulator to throw objects towards a target using deep reinforcement learning.

# Dependencies
Install required packages:

```bash
pip install torch numpy gym mujoco yaml
```
 Please notify us if you encounter version mismatches or installation issues.

```graphql
root/
├── algorithms/       # PPO, actor-critic modules
├── envs/             # Custom MuJoCo environment
├── panda_mujoco-master/           # XML configs for our robot
├── scripts/          # Run & evaluation scripts

# Quick Start
