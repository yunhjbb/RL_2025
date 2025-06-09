# render_mujoco.ps1
$algo = "ddpg"
$xml = "..\panda_mujoco-master\world_new.xml"
$max_steps = 10000
$hidden_size_actor = '64 64'
$use_cuda = $false
$render_model_path = "save/actor_latest.pt"
$act_by_model = $true
$env_config_path = "..\env_config.yaml"
$deterministic = $true # render 시에는 랜덤 샘플 끄기

$env:CUDA_VISIBLE_DEVICES = "0"

# 조건부 인자 구성
$cuda_flag = ""
if ($use_cuda) { $cuda_flag = "--cuda" }

$act_flag = ""
if ($act_by_model) { $act_flag = "--act-by-model" }

$deterministic_flag = ""
if ($deterministic) { $deterministic_flag = "--deterministic" }

python render/render_robot.py `
    --algorithm-name $algo `
    --xml-path $xml `
    --max-steps $max_steps `
    --render-model-path $render_model_path `
    --hidden-size-actor $hidden_size_actor.Split(" ") `
    --env-config-path $env_config_path `
    $cuda_flag `
    $act_flag `
    $deterministic_flag
    
