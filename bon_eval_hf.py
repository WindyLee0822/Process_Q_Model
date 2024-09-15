import pandas as pd
import numpy as np
import json
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from value_model import AutoModelForCausalLMWithValueHead
import torch
from bon_eval_utils import eval_gsm8k, eval_math_prm
import sys, os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import re
import deepspeed
from copy import deepcopy
from safetensors import safe_open
from collections import Counter


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def instruction_format(s):
#     return f'[INST] {s} [/INST]'
def instruction_format(s):

    # return f'[INST] {s} [/INST]'
    # messages = [
    #     {"role": "user",
    #      "content": s+'\n'+r"Please reason step by step, and put your final answer within \boxed{}."}
    # ]
    # messages = [
    #     {"role": "user",
    #      "content": s}
    # ]
    #
    # return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{s}\n\n### Response: Let's think step by step"


def split_query(completions, n, N=16): # extract top-n logprob completion for each query
    splitted_completions = []
    for idx in range(int(len(completions) / N)):
        samples = [sample for sample in completions if sample["idx"] == idx]
        samples = sorted(samples, key=lambda x: x["logprobs"], reverse=True)
        splitted_completions.append(samples[:n])
    return splitted_completions

def best_of_n(splitted_completions):
    selected_completions = []
    for n_completions_per_query in splitted_completions:
        n_completions_per_query = sorted(n_completions_per_query, key=lambda x: x["reward"], reverse=True)
        assert all([n_completions_per_query[0]["reward"] >= completion["reward"] for completion in n_completions_per_query])
        selected_completions.append(n_completions_per_query[0])
    return selected_completions


def compute_metrics(dataset_name, scored_results):
    metrics = {}
    sample_nums = [1, 8, 16, 32, 64, 128]

    if dataset_name == 'gsm8k':
        original_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
    else:
        path = './MATH500.jsonl'
        with open(path) as f:
            original_dataset = [json.loads(line) for line in f]

    for n in sample_nums:
        splitted_completions = split_query(scored_results, n, max(sample_nums))
        if not args.baseline and not args.combine:

            selected_completions = best_of_n(splitted_completions)
            assert len(original_dataset) == len(selected_completions)
            if dataset_name == 'math':
                # acc, _, _ = eval_math_prm([{'response':query['response']} for query in selected_completions],all_problems=[{'solution':data['question']['ground_truth_answer'],'question':data['question']['problem']} for data in original_dataset],is_extract=True)
                acc, _, _ = eval_math_prm([{'response': query['response']} for query in selected_completions],
                                          all_problems=[{'solution': data['solution'], 'question': data['problem']} for
                                                        data in original_dataset], is_extract=False)
            else:
                acc, _, _ = eval_gsm8k([{'response': query['response']} for query in selected_completions],
                                       answers=[data['answer'] for data in original_dataset],is_extract=True)
            metrics[n] = acc
            if accelerator.is_local_main_process:
                print('*********')
                print(n, acc)
                print('*********')
        else:
            selected_completions = []
            for comps in splitted_completions:
                selected_completions += comps
            if dataset_name == 'math':
                # acc, _, _ = eval_math_prm([{'response':query['response']} for query in selected_completions],all_problems=[{'solution':data['question']['ground_truth_answer'],'question':data['question']['problem']} for data in original_dataset],is_extract=True)
                acc, acc_list, output_list = eval_math_prm([{'response': query['response']} for query in selected_completions],
                                          all_problems=[{'solution': data['solution'], 'question': data['problem']} for
                                                        data in original_dataset for _ in range(n)], is_extract=False)
            else:
                acc, acc_list, output_list = eval_gsm8k([{'response': query['response']} for query in selected_completions],
                                       answers=[data['answer'] for data in original_dataset for _ in range(n)],is_extract=True)
            total_index = int(len(acc_list) / n)
            if args.baseline:
                pass_k = sum([1 for ii in range(total_index) if True in acc_list[ii*n:(ii+1)*n]])/total_index
                consistent_outputs = [Counter(output_list[ii*n:(ii+1)*n]).most_common(1)[0][0] for ii in range(total_index)]  # (num_instructions, )
                position_of_consistent_outputs = [output_list[ii*n:(ii+1)*n].index(consistent_outputs[ii]) for ii in range(total_index)]  # (num_instructions, )
                acc_of_consistency = [acc_list[ii*n:(ii+1)*n][idx_of_split] for ii, idx_of_split in enumerate(position_of_consistent_outputs)]
                sc = sum(acc_of_consistency)/total_index
                if accelerator.is_local_main_process:
                    print('*********')
                    print(n,pass_k,sc)
                    print('*********')
            else:
                correct,sumv = 0,0
                for ii in range(total_index):
                    answer_dict = {k:0 for k in set(output_list[ii*n:(ii+1)*n])}
                    reward_list = [ele['reward'] for ele in selected_completions[ii*n:(ii+1)*n]]
                    for ele,reward in zip(output_list[ii*n:(ii+1)*n],reward_list):
                        answer_dict[ele]+=torch.sigmoid(torch.tensor(reward)).item()
                    select_answer = sorted(answer_dict.items(),key=lambda x:x[1],reverse=True)[0][0]
                    correct += acc_list[ii*n:(ii+1)*n][output_list[ii * n:(ii + 1) * n].index(select_answer)]
                    sumv+=1
                if accelerator.is_local_main_process:
                    print('*********')
                    print(n,correct/sumv)
                    print('*********')
    return metrics


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--combine", type=int, default=0)
    parser.add_argument("--orm", type=int, default=0)
    parser.add_argument("--backbone-path", type=str, default="/nobackup2/prm_checkpoints/alldata-zeta-4/checkpoint-532/model.safetensors")
    parser.add_argument("--model-path", type=str, default="/nobackup2/prm_checkpoints/alldata-zeta-4/checkpoint-532/model.safetensors")
    parser.add_argument("--data-name", type=str,choices=['math','gsm8k'])
    parser.add_argument("--data-file", type=str,required=True)
    parser.add_argument("--save-file", type=str,default="./prm-data.json")

    args = parser.parse_args()
    print(args)


    seed_everything(0)
    accelerator = Accelerator()
    if not args.baseline:
        prm_token = '[PRM]'
        model_path = args.backbone_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16)
        tokenizer.add_special_tokens({'additional_special_tokens':[prm_token]})
        prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]
        model.resize_token_embeddings(len(tokenizer))
        model = AutoModelForCausalLMWithValueHead(model)
        if '.safetensor' in args.model_path:
            state_dict = {}
            with safe_open(args.model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(args.model_path)

        model.load_state_dict(state_dict)

        ds_engine = deepspeed.init_inference(model,
                                             tensor_parallel={"tp_size": 2},
                                             dtype=torch.bfloat16)

        model = ds_engine.module
        model.eval()

        def data_collator(example, tokenizer=tokenizer):
            inputs = []
            special_ids = []
            step_labels = []
            orm_ids = []
            idx,reward_idx = [],[]
            template = '{query}\n{answer}'
            for d in example:
                input_ids = tokenizer.encode(template.format(query=d['query'],answer=d['answer']),
                                              add_special_tokens=False)
                inputs.append(torch.tensor(input_ids))

                cur_special_ids = []
                for ii,id in enumerate(input_ids):
                    if id==prm_token_id:
                        cur_special_ids.append(ii)
                # assert len(cur_special_ids)==len(d['labels'])
                special_ids.append(torch.tensor(cur_special_ids))
                orm_ids.append(cur_special_ids[-1])
                # step_labels.append(torch.tensor(d['labels']))
                idx.append(d['idx'])
                reward_idx.append(d['reward_idx'])

            inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
            attention_mask = (inputs!=tokenizer.pad_token_id)
            special_ids = pad_sequence(special_ids, padding_value=-100, batch_first=True)
            # step_labels = pad_sequence(step_labels, padding_value=-100, batch_first=True)

            return {
                'input_ids': inputs.int().to(accelerator.device),
                'attention_mask': attention_mask.int().to(accelerator.device),
                'special_tokens':special_ids.to(accelerator.device),
                'orm_tokens': torch.tensor(orm_ids).to(accelerator.device),
                'idx':torch.tensor(idx).to(accelerator.device),
                'reward_idx':torch.tensor(reward_idx).to(accelerator.device)
            }
    data_name = args.data_name

    if data_name == 'gsm8k':
        file_list = [
            args.data_file,
        ]
        queries = []
        cur_queries = []
        origin_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
        for file_name in file_list:
            cur_data = json.load(open(file_name))
            if len(cur_queries) == len(cur_data):
                for cur_q, cur_d in zip(cur_queries, cur_data):
                    cur_q['responses'].extend(cur_d['responses'])
            else:
                cur_queries = deepcopy(cur_data)

        assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
        for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
            assert data['question'] == ori['question']
            assert len(data['responses']) == 128
            for response_dict in data['responses']:
                queries.append({
                    'idx': idx,
                    'prompt': data['question'],
                    'response': response_dict['text'],
                    'solution': ori['answer'],
                    'logprobs': 0,
                })
    elif data_name == 'math':
        file_list = [
            args.data_file,
        ]
        queries = []
        cur_queries = []
        path = './MATH500.jsonl'
        with open(path) as f:
            origin_dataset = [json.loads(line) for line in f]
        for file_name in file_list:
            cur_data = json.load(open(file_name))
            if len(cur_queries) == len(cur_data):
                for cur_q, cur_d in zip(cur_queries, cur_data):
                    cur_q['responses'].extend(cur_d['responses'])
            else:
                cur_queries = deepcopy(cur_data)

        assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
        for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
            assert data['question'] == ori['problem']
            assert len(data['responses']) % 128==0
            for response_dict in data['responses']:
                queries.append({
                    'idx': idx,
                    'prompt': data['question'],
                    'response': response_dict['text'],
                    'solution': ori['solution'],
                    'logprobs': 0,
                })

    if not args.baseline:
        for idx, data in enumerate(queries):
            data['reward_idx'] = idx
            data["query"] = instruction_format(data["prompt"])
            steps = re.split('Step \d+:', data['response'])
            steps = [f'Step {id + 1}: ' + step.strip() for id, step in enumerate(steps) if step.strip()!='']
            data["answer"] = f" {prm_token}\n".join(steps) + f" {prm_token}"


        dataset = Dataset.from_pandas(pd.DataFrame.from_records(queries))
        dataloader = DataLoader(dataset,batch_size=4,shuffle=False,collate_fn=data_collator)


        for inputs in tqdm(dataloader):
            with torch.no_grad():
                _, _, rewards = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            cur_index = torch.where(inputs['special_tokens']==-100,0,inputs['special_tokens'])
            if not args.orm:
                rewards = rewards.gather(dim=-1, index=cur_index)
                final_rewards = torch.where(inputs['special_tokens']==-100,1e5,rewards).min(-1).values
            else:
                rewards = rewards.gather(dim=-1, index=inputs['orm_tokens'][...,None])
                final_rewards = rewards.squeeze(1)
            for step_reward,final_reward,reward_idx in zip(rewards.tolist(),final_rewards.tolist(),inputs['reward_idx'].tolist()):
                queries[int(reward_idx)]['reward'] = final_reward
                queries[int(reward_idx)]['step_reward'] = [r for r in step_reward if r!=1e5]

    with open(args.save_file,'w') as f:
        json.dump(queries,f)
    # queries = json.load(open('temp.json'))
    compute_metrics(data_name, queries)





