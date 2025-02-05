# This file is DEPRECATED because of the vllm empty string bug,
# use hf one by one evaluate "hf_obo_evaluate.py" instead.

import torch, os
import json
from time import time
import re
# import bmtrain as bmt
import os
from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk,load_dataset


# from fastchat import conv_template


def load_generator(checkpoint, tokenizer_path):
    print("start loading generator")
    cur_time = time()
    dtype = "auto"
    gpu_memory_utilization = 0.9
    model = LLM(
        checkpoint,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=1,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype=dtype,
        tokenizer=tokenizer_path
    )
    print("check tokenizer length:")
    # print(len(model.get_tokenizer()))
    print("generator loaded")
    elapsed_time = time() - cur_time
    print("loading time: {:.2f}min".format(elapsed_time / 60))
    return model


@torch.no_grad()
def generate(prompts, repeat_num=1):
    with torch.inference_mode():
        cur_time = time()
        sampling_params = SamplingParams(
            n=1,
            temperature=0.5,
            top_p=1,
            stop=[tokenizer.eos_token, '<|eot_id|>'],
            max_tokens=2048,
        )
        responses = generator.generate(prompts, sampling_params)
        elapsed_time = time() - cur_time
        print("generation elapsed time: {:.2f}min".format(elapsed_time / 60))
        extracted_responses = []
        for response in responses:
            for i in range(repeat_num):
                extracted_responses.append(response.outputs[i].text.strip().rstrip("</s>").strip())

        elapsed_time = time() - cur_time
        print("generation elapsed time: {:.2f}min".format(elapsed_time / 60))
    return extracted_responses

def get_ms_question():
    ds = load_from_disk("/home/test/test05/lwd/hf-dataset-download/Math-Shepherd")['train']
    ds = [d for d in ds]
    queries = []
    statistic = [0, 0, 0]
    for d in ds[len(ds)//2:]:
        input_text = d['input']
        steps = re.split('Step \d+:', input_text)
        steps = [s for s in steps if s.strip() != '']
        if len(steps) == 1:
            continue
        question = steps[0]
        steps = [f'Step {i + 1}: ' + step.strip().replace('ки', '').strip() for i, step in enumerate(steps[1:]) if
                 step.strip() != '']
        label_steps = re.split('Step \d+:', d['label'])
        label_steps = [s.strip() for s in label_steps[1:]]
        try:
            for s in label_steps:
                assert s[-1] in ['+', '-'], (label_steps)
        except:
            continue
        step_labels = [1 if l[-1] == '+' else 0 for l in label_steps]
        try:
            assert len(steps) == len(step_labels)
        except:
            continue
        queries.append({
            "query": question,
        })


    return [q['query'] for q in queries]


def annotate_steps(args):

    if args.dataset == 'gsm8k-plus':
        dataset = load_dataset('qintongli/GSM-Plus')['testmini']
        print('length', len(dataset))
        questions = [data['question'] for data in dataset]
    else:
        path = '/path/to/MATH500.jsonl'
        with open(path) as f:
            dataset = [json.loads(line) for line in f]
        questions = [d['problem'] for d in dataset]

    prompts = [template.replace('{question}', question) for question in questions]

    print("first prompt:\n{}".format(prompts[0]))
    print("first prompt:\n{}".format(prompts[-1]))
    print(" prompts: {} x {} = {}".format(len(prompts), args.repeat_num, len(prompts) * args.repeat_num))
    response_list = []
    for i in range(args.repeat_num):
        responses = generate(prompts)
        assert len(prompts) == len(responses)
        response_list.append(responses)

    final_data = [{'question': question} for question in questions]
    for i, data in enumerate(final_data):
        data['responses'] = [{'text': responses[i], 'model_name': args.model_name} for responses in response_list]

    with open(args.save_path, 'w') as f:
        json.dump(final_data, f)


if __name__ == "__main__":
    import argparse
    # import deepspeed
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--model-name", type=str, default="Llama-3-70B-Instruct")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--save-path", type=str, default="/path/to/save.json")
    parser.add_argument("--repeat-num", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    generator = load_generator(args.model_path, args.tokenizer_path)

    if 'metamath' in args.model_path.lower() or 'muggle' in args.model_path.lower():
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response: Let's think step by step. Step 1:"

    elif 'llama' in args.model_path.lower() or 'zephyr' in args.model_path.lower() or 'deepseek' in args.model_path.lower() or 'llemma' in args.model_path.lower() or 'qwen' in args.model_path.lower():
        template = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Solve the following math problem step-by-step.\nDo not use any code, just use the chain of thought.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n{question}'}], tokenize=False,
                                                 add_generation_prompt=True) + ' Step 1:'

    annotate_steps(args)





