import os
import json
import time
import tempfile
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
import torch
from datasets import load_from_disk, disable_caching, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, set_seed
from peft import PeftModel
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, \
                 reset_score_in_dataset_chat_template, Instructions_n, clean_gpu_memory, Instructions_summary_n
from multi_reward_models import RewardModels
from vllm import LLM, SamplingParams
import sys
import gc
import multiprocessing
from multiprocessing import Process, Queue
import socket
tqdm.pandas()
disable_caching()

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _generate_in_subprocess(model_path, local_inputs, sampling_params_dict, process_id, result_file_path, env_vars):
    """
    在子进程中运行 vLLM 模型初始化和生成
    
    Args:
        model_path: 模型路径
        local_inputs: 输入文本列表
        sampling_params_dict: 生成参数字典（用于重建 SamplingParams）
        process_id: 进程ID
        result_file_path: 结果文件保存路径
        env_vars: 从主进程传递的所有环境变量字典
    """
    try:
        # 设置所有环境变量，确保 vLLM 的 external_launcher 能正确初始化
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = str(value)

        # 让每个子进程“独立启动自己的 vLLM / process group”
        # - 避免继承 accelerate 的 RANK/WORLD_SIZE 后，子进程之间互相 rendezvous 卡住
        # - 同时确保每个子进程仍然用“父进程对应的 GPU”
        parent_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_cvd:
            cvd_list = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
            if len(cvd_list) >= 1:
                chosen_idx = min(max(parent_local_rank, 0), len(cvd_list) - 1)
                # 缩到单卡：子进程内部只看到 1 张 GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd_list[chosen_idx]
                os.environ["LOCAL_RANK"] = "0"

        # 单进程独立组：不要继承 accelerate 的 rank/world_size
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        # 仅用于兼容某些依赖会读取该变量（本文件后面也会打印它）
        os.environ["MASTER_PORT"] = str(get_free_port())
        
        # 打印关键环境变量用于调试
        print(f"Subprocess {process_id} env vars: RANK={os.environ.get('RANK')}, "
              f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
              f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
              f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
              f"MASTER_PORT={os.environ.get('MASTER_PORT')}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # 设置 CUDA 设备（如果指定了 LOCAL_RANK）
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
            print(f"Subprocess {process_id}: Set CUDA device to {local_rank}")
        
        # 初始化 torch.distributed（如果还没有初始化）
        # external_launcher 需要 torch.distributed 已经初始化
        # 对于 tensor_parallel_size=1，每个子进程需要独立的单进程分布式环境
        import torch.distributed as dist
        if not dist.is_initialized():
            # 用 file:// 初始化，完全避免端口不一致/占用导致的 hang
            init_file = os.path.join(
                tempfile.gettempdir(),
                f"vllm_dist_init_{process_id}_{os.getpid()}_{int(time.time() * 1e6)}"
            )
            init_method = f"file://{init_file}"
            print(f"Subprocess {process_id}: Initializing torch.distributed with {init_method}")
            print(
                f"Subprocess {process_id}: "
                f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE'])
            )

        print(
            f"Subprocess {process_id}: Initialized torch.distributed "
            f"(rank={os.environ['RANK']}, world_size={os.environ['WORLD_SIZE']})"
        )
        
        print("Loading vLLM model for process {} in subprocess".format(process_id))
        llm = LLM(model=model_path,
            gpu_memory_utilization=0.6,
            tensor_parallel_size=1,
            trust_remote_code=True,
            distributed_executor_backend="external_launcher",
        )
        
        # 在子进程中重建 SamplingParams 对象
        generation_kwargs = SamplingParams(**sampling_params_dict)
        
        print("Generating responses for process {} in subprocess".format(process_id))
        outputs = llm.generate(local_inputs, sampling_params=generation_kwargs)
        local_responses = [output.outputs[0].text for output in outputs]
        print(f"Process {process_id}: Generated {len(local_responses)} responses in subprocess...")
        
        # 将结果保存到文件
        print(f"Subprocess {process_id}: Saving results to file {result_file_path}...")
        result_data = {
            'success': True,
            'responses': local_responses,
            'process_id': process_id
        }
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"Subprocess {process_id}: Results saved to file successfully")
        # close the distributed process group
        dist.destroy_process_group()
        # from vllm.distributed.parallel_state import destroy_model_parallel
        # destroy_model_parallel()
        # # del llm.llm_engine.model_executor.driver_worker
        # del llm
        # gc.collect()
        # torch.cuda.empty_cache()
        os._exit(0)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in subprocess {process_id}: {error_msg}")
        # 即使出错也保存错误信息到文件
        result_data = {
            'success': False,
            'error': error_msg,
            'process_id': process_id
        }
        try:
            with open(result_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
        except:
            pass
        os._exit(1)


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
):
    set_seed(8888 + iter)
    print('Generating ...')
    accelerator = Accelerator()
    process_id = Accelerator().local_process_index 
    gpu_id = process_id
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
    tokenizer = load_main_tokenizer(tokenizer_name)
    # ## load model 
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, 
    #     torch_dtype=torch.bfloat16,  # fast inference
    #     device_map=gpu_id, 
    # )
    # model.resize_token_embeddings(len(tokenizer))
    # if peft_name is not None:
    #     model = PeftModel.from_pretrained(model, peft_name)
    # if hasattr(model, 'merge_and_unload'):
    #     model = model.merge_and_unload()

    generation_kwargs = SamplingParams(
        max_tokens=128 if exp_type == 'assistant' else 48,
        top_k=0,
        top_p=0.9,
        temperature=0.7,
    )

    tokenizer.padding_side = "left"

    if type(dataset) == str:
        dataset = load_from_disk(dataset)

    select_index = None    
    if accelerator.is_main_process:
        select_index = np.random.randint(0, len(dataset), min(args.num_generation_samples, len(dataset)))
        print(f"Selected {len(select_index)} samples")
    accelerator.wait_for_everyone()
    obj_list = [select_index]
    select_index = broadcast_object_list(obj_list, from_process=0)[0]
    print(f"Selected {len(select_index)} samples")

    selected_dataset = dataset.select(select_index)
    scores_name_list = []
    for key in selected_dataset.column_names:
        if key.startswith('score'):
            scores_name_list.append(key)
    
    local_dataset = selected_dataset.shard(num_shards=accelerator.num_processes, index=process_id)
    # print(local_dataset[0])
    local_dataset = reset_score_in_dataset_chat_template(local_dataset, exp_type=exp_type)

    remove_columns = []
    for name in ['input_ids', 'prompt', 'text', 'response', 'query', 'prompt_with_score', 'messages'] + scores_name_list:
        if name in local_dataset.column_names:
            remove_columns.append(name)
    local_dataset = local_dataset.remove_columns(remove_columns)
    local_dataset = local_dataset.rename_column('messages_prompt_reset_score', 'messages')
    # print(local_dataset[0])

    # if len(local_dataset) > 0:
    local_messages = local_dataset['messages']
    local_inputs = tokenizer.apply_chat_template(
        local_messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    # 使用子进程运行 vLLM 生成
    # 将 SamplingParams 转换为字典以便序列化传递
    sampling_params_dict = {
        'max_tokens': generation_kwargs.max_tokens,
        'top_k': generation_kwargs.top_k,
        'top_p': generation_kwargs.top_p,
        'temperature': generation_kwargs.temperature,
    }
    
    # 收集所有环境变量并传递给子进程
    # 这对于 vLLM 的 external_launcher 是必需的
    all_env_vars = dict(os.environ)
    
    # 创建临时文件用于传递结果
    temp_dir = tempfile.gettempdir()
    result_file_path = os.path.join(temp_dir, f'vllm_result_{process_id}_{int(time.time() * 1000000)}.json')
    
    # 使用 spawn 方式创建子进程以支持 CUDA
    ctx = multiprocessing.get_context('spawn')
    subprocess = ctx.Process(
        target=_generate_in_subprocess,
        args=(model_path, local_inputs, sampling_params_dict, process_id, result_file_path, all_env_vars)
    )
    
    print(f"Starting subprocess for process {process_id}...")
    subprocess.start()
    print(f"Process {process_id}: Waiting for subprocess to complete...")
    subprocess.join()
    
    # 如果子进程还在运行，强制终止
    if subprocess.is_alive():
        print(f"Warning: Subprocess {process_id} is still alive after getting results, terminating...")
        subprocess.terminate()
    subprocess.close()

    # 从文件读取结果
    print(f"Process {process_id}: Reading results from file {result_file_path}...")
    with open(result_file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    os.remove(result_file_path)
    if not result['success']:
        raise RuntimeError(f"Subprocess failed for process {process_id}: {result.get('error', 'Unknown error')}")
    
    local_responses = result['responses']
    print(f"Process {process_id}: Received {len(local_responses)} responses from subprocess...")

    accelerator.wait_for_everyone()

    # Compute score
    reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path)
    instructions = Instructions_summary_n(reward_models.num_rewards) if exp_type == 'summary' else Instructions_n(reward_models.num_rewards)
    full_local_responses = [instructions.get_full_response(message[0]['content'], assistant_content) for message, assistant_content in zip(local_messages, local_responses)]
    # print(full_local_responses[0])
    queries_responses = [(instructions.get_input(text),  instructions.get_response(text)) for text in full_local_responses]
    # print(queries_responses[0])

    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    desired_rewards_list = [[] for _ in range(reward_models.num_rewards)]
    for text in full_local_responses:
        desired_scores = instructions.get_scores(text)
        for i in range(reward_models.num_rewards):
            desired_rewards_list[i].append(float(desired_scores[i]))

    ### merge data
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    all_desired_rewards = []
    for i in range(len(rewards_list)):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_desired_rewards.append(accelerator.gather_for_metrics(desired_rewards_list[i]))
    all_full_messages = accelerator.gather_for_metrics(local_messages)
    all_full_responses = accelerator.gather_for_metrics(local_responses)
    # print(all_full_messages[0])
    # print(all_full_responses[0])
    # sys.exit()
    all_messages_with_responses = [
        message + [{"role": "assistant", "content": response}] for message, response in zip(all_full_messages, all_full_responses)
    ]
    print(all_messages_with_responses[0])
    if process_id == 0:
        evaluation_result = {
            'messages': all_messages_with_responses,
        }
        for i in range(len(rewards_list)):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            evaluation_result['desired_score{}'.format(i+1)] = all_desired_rewards[i]

            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            print('total average desired score {}: {}'.format(i+1, np.mean(evaluation_result['desired_score{}'.format(i+1)])))

        print("length of the datasets: {}".format(len(evaluation_result['messages'])))
        # save as json dataset
        dataset = Dataset.from_dict(evaluation_result)
        dataset.to_json(os.path.join(save_path,'data.json'))
        # dataset.save_to_disk(os.path.join(save_path,'data.json'))

        # dataframe = pd.DataFrame(evaluation_result)
        # dataframe.to_csv(os.path.join(save_path,'data.csv'))
    # wait for the main process
    print("*"*100)
    accelerator.wait_for_everyone()


