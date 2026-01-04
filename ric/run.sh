export SWANLAB_API_KEY="rpxuDFlpEUHJjd1zR41yl"
export SWANLAB_PROJ_NAME="add_ric"
export CUDA_VISIBLE_DEVICES=4,5,6,7

#accelerate launch main.py \
#    --base_model_name '/data/xuwenzhe/models/qwen_3_0.6B' \
#    --train_dataset_path '/data/xuwenzhe/RiC/ric/datasets/summary_pref1faithful.hf' \
#    --save_directory './logs_trl/' \
#    --learning_rate 1e-4 \
#    --batch_size 4 \
#    --training_epochs 0 \
#    --online_training_epochs 3 \
#    --num_online_iterations 2 \
#    --num_generation_samples 20000 \
#    --num_origin_samples 10000 \
#    --load_in_8bit False \
#    --bf16 True \
#    --use_lora False \
#    --wandb_name 'summary_pref1faithful' \
#    --reward_names 'summary,faithful' \
#    --exp_type 'summary' \

python evaluation_vllm.py \
    --base_model_name 'logs_trl/summary_pref1faithful/model_iter0' \
    --reward_names 'summary,faithful' \
    --exp_type 'summary' \
    --tensor_parallel_size 4 \
    --wandb_name 'ric_offline_summary_pref1faithful_test' \
    --reward_stats_path '/data/xuwenzhe/RiC/ric/datasets/summary_pref1faithful.hf/all_reward_stat.npy' \

python evaluation_vllm.py \
    --base_model_name 'logs_trl/summary_pref1faithful/model_iter2' \
    --reward_names 'summary,faithful' \
    --exp_type 'summary' \
    --tensor_parallel_size 4 \
    --wandb_name 'ric_offline_summary_pref1faithful_test' \
    --reward_stats_path '/data/xuwenzhe/RiC/ric/datasets/summary_pref1faithful.hf/all_reward_stat.npy' \