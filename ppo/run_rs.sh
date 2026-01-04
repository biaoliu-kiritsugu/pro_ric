export WANDB_API_KEY=c41332624f6064acf85bcbb3b356a98165424329
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch ppo.py \
    --reward_name 'deberta' \
    --base_model_name '/data/xuwenzhe/pro_ric/sft/logs_trl/sft/model' \
    --exp_type 'summary' \
    --wandb_name 'rs4deberta' \
    --init_kl_coef 0.2 \
    --learning_rate 5e-6 \
    #--mini_batch_size 4 \
    #--target 6