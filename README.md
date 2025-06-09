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
├── algorithms/               # PPO, actor-critic modules
├── envs/                     # Custom MuJoCo environment
├── panda_mujoco-master/      # XML configs for our robot
├── scripts/                  # Run & evaluation scripts
```
# Quick Start
To train the policy:
```Powershell
scripts/train_mujoco.ps1
```
You can also visualize the trained policy:
```Powershell
scripts/render_mujoco.ps1
```
You can re-start demo by clicking "Load Key" button on Mujoco Viewer.
These scripts are designed for Windows PowerShell. To use on Linux or macOS, you can easily adapt them into shell scripts (.sh).

# Demo
The agent is trained to match a target velocity and position at the moment of object release ("contact off"), and the target landing point is calculated from these parameters.
The target landing point will be visualized as "green point" at the viewer.
Demonstration videos and visualizations are available.
