from datasets import load_from_disk
import argparse

from utils import Instructions_n, Instructions_summary_n, add_messages_without_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset with rewards")
    parser.add_argument("--exp_type", type=str, default='summary',
                        help="Experiment type, 'assistant' or 'summary'")
    parser.add_argument("--normalize", type=bool, default=False,
                        help="Normalize the rewards")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax normalization")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the dataset with messages")
    parser.add_argument("--num_rewards", type=int, default=3,
                        help="Number of rewards")
    
    return parser.parse_args()

def main(args):
    args = parse_args()
    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    instructions = Instructions_summary_n(args.num_rewards) if args.exp_type == 'summary' else Instructions_n(args.num_rewards)
    # dataset = dataset.select(range(6))
    if 'messages' not in dataset.column_names:
        dataset = dataset.map(lambda x: add_messages_without_score(x, instructions), batched=False, num_proc=20)
    scores_cols = [col for col in dataset.column_names if col.startswith('score')]
    dataset = dataset.select_columns(["messages"] + scores_cols)
    # dataset.to_json(args.save_path)
    dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
