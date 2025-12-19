CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch evaluation.py \
    --reward_names 'summary,faithful' \
    --exp_type 'summary' \
    --wandb_name ours_summary_3_add\
    --base_model_name './logs_trl/ours_summary_3_add/model_iter0'\