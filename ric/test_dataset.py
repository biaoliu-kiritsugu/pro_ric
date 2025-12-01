"""
Simple script to visualize dataset samples.
"""
import argparse
from datasets import load_from_disk
import json

def format_sample(sample, index):
    """Format a single sample for printing."""
    print(f"\n{'='*80}")
    print(f"Sample {index + 1}")
    print(f"{'='*80}\n")
    
    # Print prompt and response
    if 'prompt' in sample:
        prompt = sample['prompt']
        print(f"Prompt: {prompt}")
        print()
    
    if 'response' in sample:
        response = sample['response']
        print(f"Response: {response}")
        print()
    
    # Print prompt_with_score if exists
    if 'prompt_with_score' in sample:
        prompt_ws = sample['prompt_with_score']
        print(f"Prompt with Score: {prompt_ws}")
        print()
    
    # Print scores
    score_keys = [k for k in sample.keys() if k.startswith('score')]
    if score_keys:
        print("Reward Scores:")
        for key in sorted(score_keys):
            score = sample[key]
            print(f"  {key}: {score:.4f}")
        print()
    
    # Print input_ids info
    if 'input_ids' in sample:
        input_ids = sample['input_ids']
        print(f"Input IDs (length: {len(input_ids)}):")
        print(f"  First 20 tokens: {input_ids[:20]}")
        if len(input_ids) > 20:
            print(f"  Last 20 tokens: {input_ids[-20:]}")
        print()
    
    # Print query if exists
    if 'query' in sample:
        query = sample['query']
        print(f"Query: {query}")
        print()
    
    # Print prompt_with_score_ids if exists
    if 'prompt_with_score_ids' in sample:
        prompt_ids = sample['prompt_with_score_ids']
        print(f"Prompt with Score IDs (length: {len(prompt_ids)}):")
        print(f"  First 20 tokens: {prompt_ids[:20]}")
        print()
    
    # Print all other fields
    other_keys = [k for k in sample.keys() 
                  if k not in ['prompt', 'response', 'prompt_with_score', 
                              'input_ids', 'query', 'prompt_with_score_ids'] 
                  and not k.startswith('score')]
    if other_keys:
        print("Other Fields:")
        for key in other_keys:
            value = sample[key]
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")
            elif hasattr(value, 'shape'):  # tensor/array
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset samples')
    parser.add_argument('--dataset_path', type=str, 
                       default='./datasets/summary_pref1faithfuldeberta.hf',
                       help='Path to the dataset directory')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to print')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for samples')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # Convert to python format if needed
    if dataset.format['type'] == 'torch':
        dataset.set_format(type='python')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    print(f"\nPrinting {args.num_samples} samples starting from index {args.start_idx}...")
    
    # Print samples
    for i in range(args.start_idx, min(args.start_idx + args.num_samples, len(dataset))):
        sample = dataset[i]
        format_sample(sample, i - args.start_idx)
    
    print(f"\n{'='*80}")
    print("Done!")

if __name__ == '__main__':
    main()

