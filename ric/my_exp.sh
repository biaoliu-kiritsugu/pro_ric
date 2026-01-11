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
# accelerate launch main.py \
#     --base_model_name 'Qwen/Qwen3-0.6B' \
#     --train_dataset_path './datasets/summary_pref1faithfuldeberta.hf' \
#     --save_directory './logs_trl/' \
#     --learning_rate 1e-4 \
#     --batch_size 8 \
#     --training_epochs 0 \
#     --online_training_epochs 3 \
#     --num_online_iterations 2 \
#     --num_generation_samples 20000 \
#     --num_origin_samples 10000 \
#     --load_in_8bit False \
#     --bf16 True \
#     --use_lora False \
#     --wandb_name 'summary_pref1faithfuldeberta_offline' \
#     --reward_names 'summary,faithful,deberta' \
#     --exp_type 'summary' \

# export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=2,3
# # python evaluation_vllm.py \
# #     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter2' \
# #     --reward_names 'summary,faithful' \
# #     --exp_type 'summary' \
# #     --tensor_parallel_size 2 \
# #     --wandb_name 'ric_online_summary_pref1faithful_offline_test' \
# #     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy'

# # python evaluation_vllm.py \
# #     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter0' \
# #     --reward_names 'summary,faithful' \
# #     --exp_type 'summary' \
# #     --tensor_parallel_size 2 \
# #     --wandb_name 'ric_offline_summary_pref1faithful_offline_test' \
# #     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy'

# # summary and faithful rewards
# # offline
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter0' \
#     --reward_names 'summary,faithful' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_offline_summary_pref1faithful_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,1'

# # online
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter2' \
#     --reward_names 'summary,faithful' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_online_summary_pref1faithful_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,1'

# # summary and deberta rewards
# # offline
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter0' \
#     --reward_names 'summary,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_offline_summary_pref1deberta_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,2'

# # online
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter2' \
#     --reward_names 'summary,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_online_summary_pref1deberta_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '0,2'

# # faithful and deberta rewards
# # offline
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter0' \
#     --reward_names 'faithful,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_offline_summary_faithful_deberta_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '1,2'

# # online
# python evaluation_vllm_all_rewards.py \
#     --base_model_name 'logs_trl/summary_pref1faithfuldeberta_offline/model_iter2' \
#     --reward_names 'faithful,deberta' \
#     --exp_type 'summary' \
#     --tensor_parallel_size 2 \
#     --wandb_name 'ric_online_summary_faithful_deberta_test' \
#     --reward_stats_path 'datasets/summary_pref1faithfuldeberta.hf/all_reward_stat.npy' \
#     --reward_indices '1,2'

# assistant task
# prepare dataset
# accelerate launch prepare_dataset_with_rewards.py \
#     --reward_names 'harmless,helpful,humor' \
#     --exp_type 'assistant' \
#     --save_directory './datasets/assistant_harmhelphumor.hf'

# python prepare_messages.py \
#     --dataset_path './datasets/assistant_harmhelphumor.hf' \
#     --num_objects 3 \
#     --exp_type 'assistant' \
#     --save_path './datasets/assistant_harmhelphumor_messages.json'

# offline training
accelerate launch main.py \
    --base_model_name 'Qwen/Qwen3-4B' \
    --train_dataset_path './datasets/assistant_harmhelphumor.hf' \
    --save_directory './logs_trl/' \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --training_epochs 0 \
    --online_training_epochs 3 \
    --num_online_iterations 2 \
    --num_generation_samples 20000 \
    --num_origin_samples 10000 \
    --load_in_8bit False \
    --bf16 True \
    --use_lora False \
    --wandb_name 'assistant_harmlesshelpfulhumor' \
    --reward_names 'harmless,helpful,humor' \
    --exp_type 'assistant' \

# online training
# accelerate launch main.py \
#     --base_model_name 'Qwen/Qwen3-4B' \
#     --train_dataset_path './datasets/assistant_harmhelphumor.hf' \
#     --save_directory './logs_trl/' \
#     --learning_rate 1e-4 \
#     --batch_size 8 \
#     --training_epochs 0 \
#     --online_training_epochs 3 \
#     --num_online_iterations 2 \
#     --num_generation_samples 200 \
#     --num_origin_samples 50 \
#     --max_train_samples 100 \
#     --load_in_8bit False \
#     --bf16 True \
#     --use_lora False \
#     --wandb_name 'assistant_harmlesshelpfulhumor_offline' \
#     --reward_names 'harmless,helpful,humor' \
#     --exp_type 'assistant' \