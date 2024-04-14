cd ./scripts

exp_dir="your experiment directory"

#### bedrooms
config="../config/text/diffusion_bedrooms_instancond_lat32_v_bert.yaml"
exp_name="diffusion_bedrooms_instancond_lat32_v_bert"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger

#### diningrooms
config="../config/uncond/diffusion_diningrooms_instancond_lat32_v_bert.yaml"
exp_name="diffusion_diningrooms_instancond_lat32_v_bert"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger

#### livingrooms
config="../config/uncond/diffusion_livingrooms_instancond_lat32_v_bert.yaml"
exp_name="diffusion_livingrooms_instancond_lat32_v_bert"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger