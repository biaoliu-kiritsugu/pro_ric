export WANDB_API_KEY=c41332624f6064acf85bcbb3b356a98165424329
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch sft.py \
    --base_model_name '/data/xuwenzhe/models/qwen_3_0.6B' \
    --exp_type 'summary' \
    --wandb_name 'sft' \
    --batch_size 4