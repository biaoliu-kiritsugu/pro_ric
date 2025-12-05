export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2,3

# summary task of offline training
# accelerate launch main.py \
#     --base_model_name 'Qwen/Qwen3-0.6B' \
#     --train_dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
#     --save_directory './logs_test_generation/' \
#     --learning_rate 1e-4 \
#     --batch_size 1 \
#     --training_epochs 0 \
#     --num_online_iterations 1 \
#     --num_generation_samples 4 \
#     --num_origin_samples 4 \
#     --max_train_samples 16 \
#     --load_in_8bit False \
#     --bf16 True \
#     --use_lora False \
#     --wandb_name 'summary_pref1faithfuldeberta_test_generation' \
#     --reward_names 'summary,faithful,deberta' \
#     --exp_type 'summary' \


accelerate launch main.py \
    --base_model_name 'Qwen/Qwen3-0.6B' \
    --train_dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
    --save_directory './logs_trl_test/' \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --training_epochs 1 \
    --num_online_iterations 2 \
    --num_generation_samples 16 \
    --num_origin_samples 8 \
    --max_train_samples 16 \
    --load_in_8bit False \
    --bf16 True \
    --use_lora False \
    --wandb_name 'summary_pref1faithfuldeberta_offline_test_all' \
    --reward_names 'summary,faithful,deberta' \
    --exp_type 'summary' \