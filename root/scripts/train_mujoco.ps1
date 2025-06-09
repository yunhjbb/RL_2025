# train_mujoco.ps1
$env = "SimpleMuJoCoEnv"
$algo = "ddpg"
$exp = "v1"
$seed = 1557
# $xml = "..\panda_mujoco-master\world_new.xml"
$xml = "..\panda_mujoco-master\world_new.xml"

# Optional arguments with defaults
$max_steps = 10000
$num_episodes = 10000
$eval_interval = 50
$update_every = 10
$save_every = 5
$save_dir = "save"
$env_config_path = "..\env_config.yaml"
$load_actor_path = 'save/actor_latest.pt'
$load_critic_path = 'save/critic_latest.pt'
$continual_learning = $false
$continual_learning_flag = ""
if ($continual_learning) { $continual_learning_flag = "--continual-learning" }

# PPO hyperparams
$gamma = 0.99
$lam = 0.95
# $lr = 3e-4 actor, critic 학습률 분리
$lr_actor = 1e-4
$lr_critic = 3e-4  # 더 작게 설정
$clip_param = 0.2
$ppo_epoch = 5
$batch_size = 64
$value_loss_coef = 0.5
$entropy_coef = 0.005
$max_grad_norm = 0.5


# NN arch
$hidden_size_actor = '64 64'
$hidden_size_critic = '64 64'
$arch_actor = "MLP"
$arch_critic = "MLP"
$deterministic = $false
$deterministic_flag = ""
if ($deterministic) { $deterministic_flag = "--deterministic" }

Write-Host "env is $env, algo is $algo, exp is $exp, seed is $seed"

$env:CUDA_VISIBLE_DEVICES = "0"

python train/train_robot.py `
    --env-name $env `
    --algorithm-name $algo `
    --experiment-name $exp `
    --seed $seed `
    --xml-path $xml `
    --max-steps $max_steps `
    --num-episodes $num_episodes `
    --eval-interval $eval_interval `
    --update-every $update_every `
    --save-every $save_every `
    --save-dir $save_dir `
    --env-config-path $env_config_path `
    --load-actor-path $load_actor_path `
    --load-critic-path $load_critic_path `
    $continual_learning_flag `
    --gamma $gamma `
    --lam $lam `
    --lr-actor $lr_actor `
    --lr-critic $lr_critic `
    --clip-param $clip_param `
    --ppo-epoch $ppo_epoch `
    --batch-size $batch_size `
    --value-loss-coef $value_loss_coef `
    --entropy-coef $entropy_coef `
    --max-grad-norm $max_grad_norm `
    --hidden-size-actor $hidden_size_actor.Split(" ") `
    --hidden-size-critic $hidden_size_critic.Split(" ") `
    --arch-actor $arch_actor `
    --arch-critic $arch_critic `
    --deterministic_flag
