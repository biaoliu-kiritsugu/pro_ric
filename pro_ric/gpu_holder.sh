export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7

# summary
# accelerate launch prepare_dataset_with_rewards.py \
#     --reward_names 'summary,faithful,deberta' \
#     --exp_type 'summary' \
#     --save_directory './datasets/summary_pref1faithfuldeberta.hf'

# python prepare_messages.py \
#     --dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
#     --num_objects 3 \
#     --exp_type 'summary' \
#     --save_path './datasets/summary_pref1faithfuldeberta_messages.json'

# summary task of offline training
accelerate launch main.py \
    --base_model_name 'Qwen/Qwen3-0.6B' \
    --train_dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
    --save_directory './logs_gpu_holder/' \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --training_epochs 30000 \
    --num_online_iterations 0 \
    --load_in_8bit False \
    --bf16 True \
    --use_lora False \
    --wandb_name 'summary_pref1faithfuldeberta_offline_gpu_holder' \
    --reward_names 'summary,faithful,deberta' \
    --exp_type 'summary' \