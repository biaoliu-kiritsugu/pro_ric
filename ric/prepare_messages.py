from utils import Instructions_n, Instructions_summary_n
from datasets import load_from_disk
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(default='', metadata={"help": "path to the dataset"})
    num_objects: Optional[int] = field(default=3, metadata={"help": "number of objects"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    save_path: Optional[str] = field(default=None, metadata={"help": "path to save the processed dataset"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.exp_type == 'assistant':
    instructions = Instructions_n(script_args.num_objects)
    dataset = load_from_disk(script_args.dataset_path)
else:
    instructions = Instructions_summary_n(script_args.num_objects)
    dataset = load_from_disk(script_args.dataset_path)

#dataset = dataset.select(range(10))

def add_messages(sample):
    """
    Add messages field to each sample.
    Format: [{"role": "user", "content": "prompt_with_score"}, {"role": "assistant", "content": "response"}]
    """
    sample['messages'] = [
        # {"role": "user", "content": sample['prompt_with_score']},
        {"role": "user", "content": instructions.get_post_with_score(sample['query'])},
        {"role": "assistant", "content": sample['response']}
    ]
    return sample

# Add messages field to the dataset
print("Adding messages field to dataset...")
dataset = dataset.map(add_messages, batched=False, num_proc=20)

# Save the processed dataset
if script_args.save_path:
    print(f"Saving dataset to {script_args.save_path}...")
    dataset.to_json(script_args.save_path)
    print("Dataset saved successfully!")
else:
    raise ValueError("save_path not provided. Dataset not saved.")
