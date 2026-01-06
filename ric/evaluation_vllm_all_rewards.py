import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from utils import (
    get_clean_data, 
    Instructions_n, 
    build_dataset_with_preference_n, 
    load_main_tokenizer, 
    Instructions_summary_n,
    save_configs, 
    map_rewards_from_preference, 
    build_summary_dataset_with_preference_n, 
    clean_gpu_memory,
    add_messages,
    build_summary_dataset_with_preference_n_messages
)
from multi_reward_models import RewardModels

tqdm.pandas()

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

@dataclass
class ScriptArguments:
    num_prefer_points: Optional[int] = field(default=10)
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_trl_eval/')
    wandb_name: Optional[str] = field(default='test', metadata={"help": "Name for this experiment"})
    reward_names: Optional[str] = field(default='harmless,helpful') 
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
    reward_stats_path: Optional[str] = field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    tensor_parallel_size: Optional[int] = field(default=1, metadata={"help": "tensor parallel size"})
    reward_indices: Optional[str] = field(default=None, metadata={"help": "indices of rewards to evaluate, e.g. '0,1'"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenier_name = script_args.base_model_name
reward_stats_path = script_args.reward_stats_path if len(script_args.reward_stats_path) else None
print('base model: ', base_model_name)


reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)

if script_args.reward_indices is not None:
    active_reward_indices = [int(x) for x in script_args.reward_indices.split(',')]
else:
    active_reward_indices = list(range(len(reward_names)))

reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

save_info = {
    'base_model_name': base_model_name,
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))

# Initialize vLLM
# Note: We assume the environment has enough memory or the user configures CUDA_VISIBLE_DEVICES appropriately.
# We set gpu_memory_utilization to leave space for reward models if they are loaded on the same GPU.
print("Loading vLLM model...")
llm = LLM(
    model=base_model_name,
    tokenizer=tokenier_name,
    trust_remote_code=True,
    tensor_parallel_size=script_args.tensor_parallel_size,
    gpu_memory_utilization=0.6 # Adjust as needed to fit RMs
)
tokenizer = llm.get_tokenizer()
tokenizer.padding_side = "left"

# Load reward models
# We use the first GPU by default or rely on device_map="auto" inside RewardModels if implemented, 
# but RewardModels takes gpu_id. 
# We'll assume gpu_id=0 for single GPU setup or let accelerator handle it if we were using it.
# Here we just use 0.
gpu_id = 0 
print("Loading Reward Models...")
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path, active_reward_indices)
num_rewards = len(reward_model_path_list)
total_dims = 3
instructions = Instructions_n(total_dims) if exp_type == 'assistant' else Instructions_summary_n(total_dims)

rm_tokenizers = []
for i in range(num_rewards):
    rm_tokenizers.append(AutoTokenizer.from_pretrained(rm_tokenizer_path_list[i]))

# Preference settings
if reward_models.num_rewards == 2:
    N = script_args.num_prefer_points
    preferences = np.zeros((N+1, 3))
    
    idx1, idx2 = active_reward_indices[0], active_reward_indices[1]
    preferences[:, idx1] = np.arange(0,1 + 1/N, 1/N)
    preferences[:, idx2] = 1 - preferences[:, idx1]
    preferences = np.round(preferences, 1)
elif reward_models.num_rewards == 3:
    preferences = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.4, 0.4],
        [0.2, 0.6, 0.2],
        [0.33, 0.33, 0.33],
        [0.4, 0.4, 0.2],
        [0.4, 0.2, 0.4], 
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1], 
        [1.0, 0.0, 0.0], 
        ])
else: 
    raise NotImplementedError

# using Gaussian rewards as reference
rewards_reference_list = [np.random.randn(50000) for _ in range(len(preferences[0]))]

def evaluate_model_vllm(
    llm,
    reward_models,
    tokenizer,
    target_rewards,
    instructions,
    gpu_id
):
    if exp_type == 'assistant':
        valid_dataset = build_dataset_with_preference_n(hhrlhf_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test') 
    else:
        valid_dataset = build_summary_dataset_with_preference_n_messages(summary_dataset_path, tokenizer, rm_tokenizers, target_rewards, active_reward_indices, split='test') 
    
    print(f"Size of the validation set: {len(valid_dataset)}")
    
    valid_dataset = valid_dataset.remove_columns('messages')
    valid_dataset = valid_dataset.rename_column('messages_prompt_with_score', 'messages')
    messages = valid_dataset['messages']
    # Prepare prompts
    prompts = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(prompts[1])
    sampling_params = SamplingParams(
        max_tokens=128 if exp_type == 'assistant' else 48,
        top_p=0.9,
        temperature=1.0, # do_sample=True implies temp > 0. Default was do_sample=True.
        top_k=-1, # default is -1 (all) in vllm, evaluation.py had top_k=0.0 which means all? Transformers top_k=0 usually means all.
    )
    
    print("Generating with vLLM...")
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    full_responses = [instructions.get_full_response(message[0]['content'], assistant_content) for message, assistant_content in zip(messages, responses)]
    queries_responses = [(instructions.get_input(text),  instructions.get_response(text)) for text in full_responses]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    desired_rewards_list = [[] for _ in range(reward_models.num_rewards)]
    for text in full_responses:
        desired_scores = instructions.get_scores(text)
        for i in range(reward_models.num_rewards):
            desired_rewards_list[i].append(float(desired_scores[i]))

    return rewards_list, desired_rewards_list, prompts, responses


if __name__ == '__main__':

    print('evaluation........')

    for k in range(len(preferences)): 
        preference = preferences[k]
        target_rewards = map_rewards_from_preference(rewards_reference_list, preference, method='l2').reshape(-1)
        print(k, target_rewards, preference)
        
        all_rewards, all_desired_rewards, all_full_prompts, all_full_responses = evaluate_model_vllm(
            llm, reward_models, tokenizer, target_rewards, instructions, gpu_id
        )
        
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            evaluation_result['desired_score{}'.format(i+1)] = all_desired_rewards[i]

            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            print('total average desired score {}: {}'.format(i+1, np.mean(evaluation_result['desired_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        # if len(preference) == 3:
        #     dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))
        # else:
        #     dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))

        if len(active_reward_indices) == 3:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[active_reward_indices[0]], preference[active_reward_indices[1]])))


