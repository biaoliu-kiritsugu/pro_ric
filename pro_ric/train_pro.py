'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_pro.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path ./datasets/summary_pref1faithfuldeberta.hf \
    --output_dir ./pro_summary \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --temperature 0.1 \
    --exp_type 'summary' \
    --max_length 512 \
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import argparse
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
import torch.nn.functional as F
import swanlab

from utils import Instructions_n, Instructions_summary_n, add_messages

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Name of the pretrained model to use")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset with rewards")
    parser.add_argument("--output_dir", type=str, default="./pro",
                        help="Directory to save the model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--exp_type", type=str, default='summary',
                        help="Experiment type, 'assistant' or 'summary'")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax normalization")
    return parser.parse_args()

def train(args):
    # Initialize accelerator
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        swanlab.init(
            experiment_name="score_classifier",
            description="Training score classifier with soft targets",
            config=vars(args)
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.select(range(len(dataset) // 2))
    score_cols = [col for col in dataset.column_names if col.startswith('score')]
    score_cols.sort()
    num_rewards = len(score_cols)
    if accelerator.is_main_process:
        print(f"Detected {num_rewards} reward columns: {score_cols}")

    instructions = Instructions_summary_n(num_rewards) if args.exp_type == 'summary' else Instructions_n(num_rewards)
    # dataset = dataset.select(range(200))
    if 'messages' not in dataset.column_names:
        dataset = dataset.map(lambda x: add_messages(x, instructions), batched=False, num_proc=20)
    
    # Keep messages and score columns
    dataset = dataset.select_columns(["messages"] + score_cols)
    # print(dataset[0:3])
    # import sys
    # sys.exit()

    # Load reward statistics if available (optional)
    stats_path = os.path.join(args.dataset_path, 'all_reward_stat.npy')
    if os.path.exists(stats_path) and accelerator.is_main_process:
        reward_stats = np.load(stats_path)
        print("Loaded reward statistics:", reward_stats)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_rewards,
        trust_remote_code=True,
        problem_type="single_label_classification"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create dataloader
    def collate_fn(batch):
        # Process inputs
        messages_list = [item['messages'] for item in batch]
        texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in messages_list]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        
        # Process targets
        scores = []
        for item in batch:
            item_scores = [item[f'score{i+1}'] for i in range(num_rewards)]
            scores.append(item_scores)
        
        scores_tensor = torch.tensor(scores, dtype=torch.float)
        # Softmax normalization
        temperature = args.temperature
        targets = F.softmax(scores_tensor / temperature, dim=1)
        
        return inputs, targets
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Training loop
    best_loss = float('inf')
    batch_losses = []  # List to store loss for each batch
    global_step = 0  # Counter for total number of batches processed
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                          disable=not accelerator.is_main_process)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Cross Entropy with soft targets (KL divergence equivalent for optimization)
            loss = nn.CrossEntropyLoss()(logits, targets)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            loss_value = loss.item()
            total_loss += loss_value
            batch_losses.append(loss_value)
            global_step += 1
            
            if accelerator.is_main_process:
                progress_bar.set_postfix({'loss': loss_value})
                swanlab.log({
                    "train/loss": loss_value,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (batch_idx + 1) / len(train_dataloader)
                })
                
                # Plot and save the loss curve every 100 batches
                # if global_step % 100 == 0:
                #     plt.figure(figsize=(15, 6))
                #     plt.plot(range(1, len(batch_losses) + 1), batch_losses)
                #     plt.title('Training Loss Curve (per batch)')
                #     plt.xlabel('Batch')
                #     plt.ylabel('Loss')
                #     plt.grid(True)
                #     plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
                #     plt.close()
        
        # Calculate average loss
        accelerator.wait_for_everyone()
        avg_loss = total_loss / len(train_dataloader)
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            swanlab.log({"train/epoch_avg_loss": avg_loss})
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
    # Final plot of the complete loss curve
    # if accelerator.is_main_process:
    #     plt.figure(figsize=(15, 6))
    #     plt.plot(range(1, len(batch_losses) + 1), batch_losses)
    #     plt.title('Training Loss Curve (per batch)')
    #     plt.xlabel('Batch')
    #     plt.ylabel('Loss')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(args.output_dir, 'loss_curve_final.png'))
    #     plt.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)
