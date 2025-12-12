import os
from accelerate import Accelerator
import torch
from datasets import load_from_disk, disable_caching
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed
from trl import SFTTrainer, SFTConfig
import numpy as np
from utils import Instructions_n, add_chat_template_kwargs, load_main_tokenizer, save_configs, Instructions_summary_n, print_trainable_parameters, add_score4messaegs
disable_caching()


def train_model(
    base_model_name,
    reward_model_path_list,
    train_dataset,
    save_path,
    tokenizer_name=None,
    rm_tokenizer_path_list=None,
    peft_name=None,
    generated_dataset=None,
    training_epochs=1,
    training_steps=None,
    learning_rate=None,
    iter=0,
    lr_scheduler_type='linear',
    args=None,
    exp_type='assistant',
    use_lora=False,
    max_train_samples=None,
    score_temperature=0.1,
    score_rate=10,
):
    set_seed(8888 + iter)
    print('base model: ', base_model_name)
    training_args = SFTConfig(
            num_train_epochs=training_epochs,
            max_steps=training_steps if training_steps is not None else -1,
            output_dir=os.path.join(args.save_directory, args.wandb_name),
            dataloader_drop_last=True,
            do_eval=False,
            save_strategy='no', 
            # save_strategy='steps', 
            # save_steps=1000000,
            logging_steps=10,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=0,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            weight_decay=0.01,
            bf16=True if args.bf16 else False,
            run_name=args.wandb_name + '_iter' + str(iter),
            report_to='swanlab',
            ddp_find_unused_parameters=False,
            max_length=4096,
        )
    
    # # save training args
    save_configs(training_args, save_path)
    accelerator = Accelerator()
    process_id = Accelerator().local_process_index 
    gpu_id = process_id
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

    # if use_lora:
    #     lora_config = LoraConfig(
    #         r=64, 
    #         lora_alpha=128,
    #         lora_dropout=0.05,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    tokenizer = load_main_tokenizer(tokenizer_name)

    ### load dataset when input a path
    if type(train_dataset) == str:
        train_dataset = load_from_disk(train_dataset)

    train_dataset = train_dataset.select(range(max_train_samples)) if max_train_samples is not None else train_dataset
    num_rewards = len(reward_model_path_list)
    dataset = train_dataset.map(lambda x: add_score4messaegs(x, num_rewards, score_temperature, score_rate), batched=False, num_proc=20)
    # print(train_dataset[0:3])
    selected_index = np.arange(0, len(dataset))
    np.random.shuffle(selected_index)
    dataset = dataset.select(selected_index)
    dataset = dataset.map(add_chat_template_kwargs, batched=False, num_proc=20)
    print(f"Size of the train set: {len(dataset)}")

    #### training 
    if training_epochs > 0 or (training_steps is not None and training_steps > 0):
        if args.load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                load_in_8bit=True, device_map=gpu_id)
        else: # load in bf 16
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                torch_dtype=torch.bfloat16, device_map=gpu_id)

        model.resize_token_embeddings(len(tokenizer))
        # if peft_name is not None:
        #     model = PeftModel.from_pretrained(model, peft_name, is_trainable=True)

        print_trainable_parameters(model)
        # if exp_type == 'assistant':
        #     response_template_ids = tokenizer.encode(Instructions_n.response_split, add_special_tokens=False)[1:]  
        # else:
        #     response_template_ids = tokenizer.encode(Instructions_summary_n.response_split, add_special_tokens=False)[1:]  
        # collator = DataCollatorForCompletionOnlyLM(
        #                 response_template=response_template_ids, 
        #                 tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            # peft_config=lora_config if use_lora else None,
            # packing=False,
            # dataset_text_field="query",
            # data_collator=collator,
        )
        trainer.train()
        if process_id == 0:
            print("Saving last checkpoint of the model")
            trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    
    # wait for the main process
    accelerator.wait_for_everyone()
    return train_dataset

