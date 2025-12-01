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

# offline training for summary task test
accelerate launch main.py \
    --base_model_name 'Qwen/Qwen3-0.6B' \
    --train_dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
    --save_directory './logs_trl_test/' \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --training_epochs 1 \
    --num_online_iterations 0 \
    --online_training_steps 10 \
    --num_generation_samples 16 \
    --num_origin_samples 8 \
    --max_train_samples 8 \
    --load_in_8bit False \
    --bf16 True \
    --use_lora False \
    --wandb_name 'summary_pref1faithfuldeberta_offline' \
    --reward_names 'summary,faithful,deberta' \
    --exp_type 'summary' \
