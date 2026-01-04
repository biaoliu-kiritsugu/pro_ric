CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch evaluation_all.py \
    --reward_names 'harmless,helpful' \
    --exp_type 'assistant' \
    --wandb_name ours_assistant_3d+pre\
    --base_model_name './logs_trl/ours_assistant_3d+pre/model_iter0'\
    --reward_stats_path '/data/xuwenzhe/RiC/ric/datasets/assistant_all.hf/all_reward_stat.npy' \
    --reward_indices '0,1'\

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch evaluation_all.py \
#    --reward_names 'harmless,helpful' \
#    --exp_type 'assistant' \
#    --wandb_name ours_assistant_2d_offline\
#    --base_model_name './logs_trl/ours_assistant_2d/model_iter0'\
#    --reward_stats_path '/data/xuwenzhe/RiC/ric/datasets/assistant_all.hf/all_reward_stat.npy' \
#    --reward_indices '0,1'\