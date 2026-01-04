CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch eval_rewarded_soups.py \
    --reward_names 'faithful,deberta' \
    --base_model_path1 './logs_ppo_summary/rs4faithful/batch_153' \
    --base_model_path2 './logs_ppo_summary/rs4deberta/batch_153' \
    --exp_type 'summary' \
    --wandb_name 'eval_pposoups_faithful_deberta'