export SWANLAB_API_KEY="rpxuDFlpEUHJjd1zR41yl"
export SWANLAB_PROJ_NAME="add_ric"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py \
    --train_dataset_path '/data/xuwenzhe/RiC/ric/datasets/assistant_all.hf' \
    --exp_type 'assistant' \
    --reward_names 'harmless,helpful,humor' \
    --training_epochs 3 \
    --online_training_epochs 0 \
    --num_online_iterations 0 \
    --wandb_name 'ours_assistant_3d+pre' \
    --batch_size 2 \
    --classifier_path '/data/xuwenzhe/my_pro/score_classifier/assistant_all_freeze/best_model.pt' \
    --base_model_name '/data/xuwenzhe/models/qwen_3_4B_add_3d+head' \

#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch main.py \
    #--train_dataset_path '/data/xuwenzhe/RiC/ric/datasets/summary_pref1faithful.hf' \
    #--exp_type 'summary' \
    #--reward_names 'summary,faithful' \
    #--training_epochs 0 \
    #--online_training_epochs 3 \
    #--num_online_iterations 2 \
    #--wandb_name 'ours_summary_3_add' \
    #--batch_size 4 \
    #--classifier_path '/data/xuwenzhe/my_pro/score_classifier/summary/best_model.pt' \
    #--base_model_name '/data/xuwenzhe/models/qwen_3_0.6B_add' \