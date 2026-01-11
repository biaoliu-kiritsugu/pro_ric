export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# assistant task
# accelerate launch prepare_dataset_with_rewards.py \
#     --reward_names 'harmless,helpful,humor' \
#     --exp_type 'assistant' \
#     --save_directory './datasets/assistant_harmhelphumor.hf'

# python prepare_messages.py \
#     --dataset_path './datasets/assistant_harmhelphumor.hf' \
#     --num_objects 3 \
#     --exp_type 'assistant' \
#     --save_path './datasets/assistant_harmhelphumor_messages.hf'

# train pro for assistant task
# accelerate launch train_pro.py \
#     --model_name Qwen/Qwen3-0.6B \
#     --dataset_path ./datasets/assistant_harmhelphumor_messages.hf \
#     --output_dir ./pro/pro_assistant \
#     --num_epochs 3 \
#     --batch_size 16 \
#     --learning_rate 1e-5 \
#     --temperature 1 \
#     --exp_type 'assistant' \
#     --max_length 512 \

# accelerate launch train_pro.py \
#   --model_name Qwen/Qwen3-0.6B \
#   --dataset_path ./datasets/summary_pref1faithfuldeberta.hf \
#   --output_dir ./pro_summary \
#   --num_epochs 3 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --temperature 1 \
#   --exp_type summary \
#   --max_length 512 \
#   --monitor_outputs \
#   --monitor_every_n_steps 20

# python add_messages2dataset.py \
#     --dataset_path ./datasets/assistant_harmhelphumor.hf \
#     --exp_type 'assistant' \
#     --save_path ./datasets/assistant_harmhelphumor_messages.hf \
#     --num_rewards 3 \

# accelerate launch main.py \
#     --base_model_name 'Qwen/Qwen3-4B' \
#     --train_dataset_path './datasets/assistant_harmhelphumor_messages.hf' \
#     --save_directory './logs_pro/' \
#     --learning_rate 1e-4 \
#     --batch_size 4 \
#     --training_epochs 3 \
#     --score_temperature 0.5 \
#     --score_rate 10 \
#     --score_shift 2 \
#     --pro_path './pro/pro_assistant' \
#     --online_training_epochs 3 \
#     --num_online_iterations 0 \
#     --num_generation_samples 60000 \
#     --num_origin_samples 0 \
#     --load_in_8bit False \
#     --bf16 True \
#     --use_lora False \
#     --wandb_name 'assistant_harmhelphumor_online_pro_t0.5_rate10_gen60000_shift2_iter4' \
#     --reward_names 'harmless,helpful,humor' \
#     --exp_type 'assistant' \

accelerate launch main.py \
    --base_model_name 'Qwen/Qwen3-4B' \
    --train_dataset_path './datasets/assistant_harmhelphumor_messages.hf' \
    --save_directory './logs_pro/' \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --training_epochs 3 \
    --score_temperature 0.5 \
    --score_rate 10 \
    --score_shift 0 \
    --pro_path './pro/pro_assistant' \
    --online_training_epochs 3 \
    --num_online_iterations 4 \
    --num_generation_samples 60000 \
    --num_origin_samples 0 \
    --load_in_8bit False \
    --bf16 True \
    --use_lora False \
    --wandb_name 'assistant_harmhelphumor_online_pro_t0.5_rate10_gen60000_shift2_iter4' \
    --reward_names 'harmless,helpful,humor' \
    --exp_type 'assistant' \

# # summary and faithful rewards
# offline
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter0' \
#     --reward_names 'summary,faithful' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'ric_offline_summary_pref1faithful_pro_t0.5_rate10' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,1' \
#     --score_rate 10

# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter0' \
#     --reward_names 'faithful,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'ric_offline_summary_faithfuldeberta_pro_t0.5_rate10' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '1,2' \
#     --score_rate 10

# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter0' \
#     --reward_names 'summary,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'ric_offline_summary_pref1deberta_pro_t0.5_rate10' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,2' \
#     --score_rate 10

# online
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter4' \
#     --reward_names 'summary,faithful' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'pro_online_summary_pref1faithful_pro_t0.5_rate10_gen60000_shift2_iter4' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,1' \
#     --score_rate 10

# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter4' \
#     --reward_names 'faithful,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'pro_online_summary_faithfuldeberta_pro_t0.5_rate10_gen60000_shift2_iter4' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '1,2' \
#     --score_rate 10

# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_pro/summary_pref1faithfuldeberta_online_pro_t0.5_rate10_gen60000_iter4/model_iter4' \
#     --reward_names 'summary,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 4 \
#     --wandb_name 'pro_online_summary_pref1deberta_pro_t0.5_rate10_gen60000_shift2_iter4' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,2' \
#     --score_rate 10

# # parameters search
# temperature_list=(0.7 1.0 1.5 2.0)
# rate_list=(1 5)
# for temperature in ${temperature_list[@]}; do
#     for rate in ${rate_list[@]}; do
#         accelerate launch main.py \
#             --base_model_name 'Qwen/Qwen3-0.6B' \
#             --train_dataset_path './datasets/summary_pref1faithfuldeberta_messages.hf' \
#             --save_directory './logs_trl_pro/' \
#             --learning_rate 1e-4 \
#             --batch_size 8 \
#             --training_epochs 3 \
#             --score_temperature ${temperature} \
#             --score_rate ${rate} \
#             --online_training_epochs 0 \
#             --num_online_iterations 0 \
#             --num_generation_samples 20000 \
#             --num_origin_samples 10000 \
#             --load_in_8bit False \
#             --bf16 True \
#             --use_lora False \
#             --wandb_name "summary_pref1faithfuldeberta_offline_pro_t${temperature}_rate${rate}" \
#             --reward_names 'summary,faithful,deberta' \
#             --exp_type 'summary' \

#         # summary and faithful rewards
#         python evaluation_vllm_all_rewards.py \
#             --base_model_name "logs_trl_pro/summary_pref1faithfuldeberta_offline_pro_t${temperature}_rate${rate}/model_iter0" \
#             --reward_names 'summary,faithful' \
#             --exp_type 'summary' \
#             --tensor_parallel_size 2 \
#             --wandb_name "ric_offline_summary_pref1faithful_pro_t${temperature}_rate${rate}" \
#             --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#             --reward_indices '0,1'
#     done
# done