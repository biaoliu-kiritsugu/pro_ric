import os
from accelerate import Accelerator
import torch
from datasets import load_from_disk, disable_caching,Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, set_seed
from peft import PeftModel
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, \
                 reset_score_in_dataset, Instructions_n, clean_gpu_memory, Instructions_summary_n,load_dataset_with_message,build_full_responses,build_message
from multi_reward_models import RewardModels
tqdm.pandas()
disable_caching()

def generate_data(
    model_path,
    reward_model_path_list,
    dataset,
    tokenizer_name,
    rm_tokenizer_path_list,
    save_path,
    reward_stats_path,
    iter=0,
    peft_name=None,
    args=None,
    exp_type='assistant',
    score_classifier=None,
    batch_size=1
):
    set_seed(8888 + iter)
    print('Generating ...')
    process_id = Accelerator().local_process_index 
    gpu_id = process_id
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
    tokenizer = load_main_tokenizer(tokenizer_name)
    ## load model 
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,  # fast inference
        device_map=gpu_id, 
    )
    model.resize_token_embeddings(len(tokenizer))
    #if peft_name is not None:
        #model = PeftModel.from_pretrained(model, peft_name)
    #if hasattr(model, 'merge_and_unload'):
        #model = model.merge_and_unload()

    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        'temperature': 0.7,
        "do_sample": True,
    }
    ### for evaluation
    print('generation........')
    tokenizer.padding_side = "left"

    if type(dataset) == str:
        #dataset = load_from_disk(dataset)
        dataset = load_dataset_with_message(dataset,tokenizer,generate=True,exp=exp_type)
    select_index = np.random.randint(0, len(dataset), args.num_generation_samples)
    selected_dataset = dataset.select(select_index)
    scores_name_list = []
    #for key in dataset.column_names:
        #if key.startswith('score'):
            #scores_name_list.append(key)

    #selected_dataset = reset_score_in_dataset(selected_dataset, tokenizer, exp_type=exp_type) #, [dataset[key] for key in scores_name_list])
    print(f"Size of the selected dataset: {len(selected_dataset)}")

    batch_size = batch_size # batch size > 1 will pad on the left and will negatively influence generation
    remove_columns = []
    selected_dataset = selected_dataset.add_column("sample_id", range(len(selected_dataset)))
    origin_dataset=selected_dataset
    for name in ['prompt_message', 'prompt', 'text', 'response', 'query', 'prompt_with_score','message','prompt_with_score_ids','input'] + scores_name_list:
        if name in selected_dataset.column_names:
            remove_columns.append(name)
    selected_dataset = selected_dataset.remove_columns(remove_columns)
    #selected_dataset = selected_dataset.rename_column('prompt_with_score_ids', 'input_ids')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(selected_dataset, batch_size=batch_size, drop_last=True, collate_fn=data_collator)
    
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)

    full_response_tensors = []
    full_prompts = []
    scores=[]

    pbar = tqdm(total=len(selected_dataset) // batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            sample_ids=batch['sample_id'].detach().cpu().tolist()
            prompts=origin_dataset[sample_ids]['prompt']
            pref_vec=score_classifier(prompts)
            input_len = batch['input_ids'].shape[1]
            response_tensors = accelerator.unwrap_model(model).generate(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"],pref_vec=pref_vec,**generation_kwargs)
            #full_response_tensors.extend(response_tensors)
            #full_prompts.extend(batch['input_ids'])
            full_prompts.extend(prompts)
            full_response_tensors.extend(response_tensors[:,input_len:])
            for s in pref_vec:
                scores.append(s.detach().cpu().tolist())    
            #print(tokenizer.decode(batch['input_ids'][0]))
            pbar.update(1)

    ## decoding
    #full_prompts = tokenizer.batch_decode(full_prompts)
    full_responses = tokenizer.batch_decode(full_response_tensors,skip_special_tokens=True)
    #full_prompts, full_responses = get_clean_data(full_responses, full_prompts, remove_bad=True)
    for i in range(len(full_responses)):
        clean_response=full_responses[i].replace('user\n','')
        clean_response=clean_response.replace('<think>\n\n</think>\n\n','')
        clean_response=clean_response.replace('assistant\n','')
        full_responses[i]=clean_response
    #print(full_responses[0])
    ## del model and clear gpu 
    del model, data_loader
    clean_gpu_memory()

    # Compute score
    reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path)
    if exp_type == 'assistant':
        instructions = Instructions_n(reward_models.num_rewards)
    else:
        instructions = Instructions_summary_n(reward_models.num_rewards)
    format_responses = [build_full_responses(prompt,score,response,exp='assistant') for prompt,response,score in zip(full_prompts,full_responses,scores)]
    #full_responses=new_full_responses
    #print(format_responses[0])
    queries_responses = [(instructions.get_input(text),  instructions.get_response(text)) for text in format_responses]
    #print(queries_responses[0])
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    desired_rewards_list = [[] for _ in range(reward_models.num_rewards)]
    for text in format_responses:
        try:
            desired_scores = instructions.get_scores(text)
            for i in range(reward_models.num_rewards):
                desired_rewards_list[i].append(float(desired_scores[i]))
        except:
            print(text)

    ### merge data
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    all_desired_rewards = []
    for i in range(len(rewards_list)):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_desired_rewards.append(accelerator.gather_for_metrics(desired_rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)

    if process_id == 0:
        full_messages=[build_message(p,r,exp=exp_type) for p,r in zip(all_full_prompts,all_full_responses)]
        evaluation_result = {
            'message': full_messages,
        }
        for i in range(len(rewards_list)):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            evaluation_result['desired_score{}'.format(i+1)] = all_desired_rewards[i]

            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            print('total average desired score {}: {}'.format(i+1, np.mean(evaluation_result['desired_score{}'.format(i+1)])))

        print("length of the datasets: {}".format(len(evaluation_result['message'])))
        # save as json dataset
        dataset = Dataset.from_dict(evaluation_result)
        dataset.to_json(os.path.join(save_path,'data.json'))
        # 格式化保存json数据集
        dataset.to_json(os.path.join(save_path,'formatted_data.json'), indent=2)
        # dataset.save_to_disk(os.path.join(save_path,'data.json'))

        # dataframe = pd.DataFrame(evaluation_result)
        # dataframe.to_csv(os.path.join(save_path,'data.csv'))
    # wait for the main process
    print("*"*100)
    accelerator.wait_for_everyone()


