import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torch
import deepspeed
from torch.nn.utils.rnn import pad_sequence
import os
import re
import argparse
from datasets import load_from_disk, Dataset
import numpy as np
import pandas as pd
from grader import math_equal
import random
import signal
import contextlib
import textwrap
import jsonlines
import subprocess
import math
from normalizer import extract_math_answer_new

random.seed(2)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def eval_gsm8k(scored_results, print_acc=False, answers=None, is_extract=False):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    def extract_answer_hf(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return eval(match_str)
        else:
            return INVALID_ANS

    def extract_answer(completion):
        try:
            last_number = re.findall(r'\d+\.\d+|\d+', completion)[-1]
            return eval(last_number)
        except:
            return INVALID_ANS

    def is_correct(completion, answer, is_extract):
        if is_extract:
            try:
                gold = eval(answer)
            except:
                print(answer)
                gold=answer
        else:
            gold = extract_answer_hf(answer)
        assert gold != INVALID_ANS, f"No ground truth answer found in the document:{answer}"
        return extract_answer(completion) == gold

    completions = [result["response"] for result in scored_results]
    correct_pass = []

    if answers == None:
        test = load_from_disk(os.path.join('/mnt/data/user/tc_agi/ylf/eval_data/gsm8k', "test"))
        answers = [d['solution'] for d in test]
        # test = pd.DataFrame.from_dict({'answer':answers})

    # test = test.add_column("completion", completions)

    acc_list = [is_correct(completion, answer, is_extract) for completion, answer in zip(completions, answers)]
    acc = 100 * sum(acc_list) / len(acc_list)
    if print_acc:
        print("Accuracy:", acc)
    return acc, acc_list, [extract_answer(completion) for completion in completions]


def eval_math(scored_results, print_acc=False):
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def _clean_numbers(string):
        """
        Clean Numbers in the given string

        >>> _clean_numbers(None, "Hello 123")
        'Hello 123'
        >>> _clean_numbers(None, "Hello 1234")
        'Hello 1,234'
        >>> _clean_numbers(None, "Hello 1234324asdasd")
        'Hello 1,234,324asdasd'
        """
        num_prev_digits = 0
        new_string = ""
        for i, c in enumerate(string):
            # isdigit() doesnt work here because of weird unicode chars.
            if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
                num_prev_digits += 1
            else:
                if num_prev_digits > 3:
                    # Some fixing
                    string_number = new_string[-num_prev_digits:]
                    new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
                num_prev_digits = 0
            new_string += c

        if num_prev_digits > 3:
            # Some fixing
            string_number = new_string[-num_prev_digits:]
            new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

        return new_string

    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            return s[len(left):-1]
        except:
            return None

    def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1

        if left_brace_idx is None or right_brace_idx is None:
            return None

        return string[left_brace_idx + 1: right_brace_idx].strip()

    def match_answer(response):
        is_matched = False
        ans_marker = 'answer:\n'
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

        ans_marker = 'answer: '
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

        # Find boxed
        ans_boxed = _last_boxed_only_string(response)
        if ans_boxed:
            is_matched = True
            response = ans_boxed

        # Grade
        return is_matched, response

    path = '/mnt/data/user/tc_agi/ylf/eval_data/math/test/math_test_cleaned.json'
    all_problems = pd.read_json(path).to_dict(orient="records")  # [:len(scored_results)]

    completions = []
    random_completions = []
    outputs = []
    answers = []
    types = []
    levels = []
    matches = []
    fnames_list = []
    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = []
    total = 0
    random_correct = 0

    completions = [result["response"] for result in scored_results]

    for problem_data, model_output in zip(all_problems, completions):
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None
        answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
        levels.append(prob_level)
        types.append(prob_type)
        is_matched, model_output = match_answer(model_output)
        matches.append(is_matched)
        outputs.append(model_output)
        answers.append(answer)

        try:
            # equiv = is_equiv(model_output, answer)
            equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False
        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        correct.append(equiv)

    total = len(all_problems)
    acc = math.fsum(correct) / total * 100
    if print_acc:
        print("Overall Accuracy = {}/{} = {:.4f}".format(math.fsum(correct), total, acc))
    return acc, correct, outputs


def eval_math_prm(scored_results, print_acc=False, all_problems=None, is_extract=False):
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def _clean_numbers(string):
        """
        Clean Numbers in the given string

        >>> _clean_numbers(None, "Hello 123")
        'Hello 123'
        >>> _clean_numbers(None, "Hello 1234")
        'Hello 1,234'
        >>> _clean_numbers(None, "Hello 1234324asdasd")
        'Hello 1,234,324asdasd'
        """
        num_prev_digits = 0
        new_string = ""
        for i, c in enumerate(string):
            # isdigit() doesnt work here because of weird unicode chars.
            if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
                num_prev_digits += 1
            else:
                if num_prev_digits > 3:
                    # Some fixing
                    string_number = new_string[-num_prev_digits:]
                    new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
                num_prev_digits = 0
            new_string += c

        if num_prev_digits > 3:
            # Some fixing
            string_number = new_string[-num_prev_digits:]
            new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

        return new_string

    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            return s[len(left):-1]
        except:
            return None

    def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1

        if left_brace_idx is None or right_brace_idx is None:
            return None

        return string[left_brace_idx + 1: right_brace_idx].strip()

    def match_answer(response):
        is_matched = False
        ans_marker = 'answer:\n'
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]


        ans_marker = 'answer:'
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

        ans_marker = 'the answer is: '
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("."):
                response = response[:-1]

        # ans_marker = 'the answer is '
        # ans_idx = response.lower().rfind(ans_marker)
        # if ans_idx != -1:
        #     is_matched = True
        #     response = response[ans_idx + len(ans_marker):].strip()
        #     if response.endswith("."):
        #         response = response[:-1]

        ans_marker = 'the final answer is '
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            response = response.replace('I hope it is correct.','').strip()
            if response.endswith("."):
                response = response[:-1]

        # Find boxed
        ans_boxed = _last_boxed_only_string(response)
        if ans_boxed:
            is_matched = True
            response = ans_boxed

        # Grade
        return is_matched, response

    if not all_problems:
        path = '/mnt/data/user/tc_agi/user/lwd/mcts-data/math_train_filter_check.json'
        all_problems = pd.read_json(path).to_dict(orient="records")  # [:len(scored_results)]

    completions = []
    random_completions = []
    outputs = []
    answers = []
    types = []
    levels = []
    matches = []
    fnames_list = []
    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = []
    total = 0
    random_correct = 0

    completions = [result["response"] for result in scored_results]
    write_data = []
    assert len(all_problems) == len(completions), f"{len(all_problems)}\n{len(completions)}"
    for problem_data, model_output in zip(all_problems, completions):

        try:
            prob_level = problem_data["level"]
            prob_type = problem_data["type"]
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None
            prob_type = None


        try:
            answer = extract_math_answer_new(problem_data['question'], problem_data["solution"], is_extract)
            is_matched, model_output = match_answer(model_output)
        except:
            is_matched = False
            model_output = None
            answer = None


        levels.append(prob_level)
        types.append(prob_type)
        matches.append(is_matched)
        outputs.append(model_output)
        answers.append(answer)

        try:
            # equiv = is_equiv(model_output, answer)
            equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False
        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        correct.append(equiv)
        if not equiv:
            write_data.append(problem_data)

    total = len(all_problems)
    acc = math.fsum(correct) / total * 100
    if print_acc:
        print("Overall Accuracy = {}/{} = {:.4f}".format(math.fsum(correct), total, acc))
    return acc, correct, outputs


def eval_mbpp(scored_results, print_acc=False):
    class TimeoutException(Exception):
        pass

    @contextlib.contextmanager
    def time_limit(seconds: float):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")

        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def exec_helper(code):
        with time_limit(3):
            exec(compile(code, filename="mbpp", mode='exec'), globals())

    def evaluate(dataset):
        correct = []
        format_error = 0
        exec_error = 0
        for i, example in enumerate(iter(dataset)):
            correct.append(False)
            completion = example["completion"]
            # remove texts
            code = completion.split("\n")
            code_ = []
            for c in code:
                if len(c.lstrip()) == len(c) and not c.startswith("def"):
                    continue
                code_.append(c)
            code = "\n".join(code_)
            function = code
            test_cases = "\n".join(example["test_list"]).replace("\/", "/")
            test_run = "\n".join([
                function,
                test_cases,
            ])
            # define function
            try:
                exec_helper(function)
            except Exception as e:
                format_error += 1
                continue

            try:
                # run test case
                exec_helper(test_cases)
                exec_helper(test_run)
            except:
                exec_error += 1
                continue
            else:
                correct[-1] = True
        return 100 * math.fsum(correct) / len(dataset), 100 * exec_error / len(dataset), 100 * format_error / len(
            dataset), correct

    completions = []
    random_completions = []
    all_solution = []
    import random
    random.seed(2)
    dataset = Dataset.from_json('/mnt/data/user/tc_agi/sbj/code/new_mbpp.json')

    completions = [result["response"] for result in scored_results]

    dataset = dataset.add_column("completion", completions)
    # dataset = dataset.add_column("completion_list", all_solution)

    accuracy, exec_error, format_error, acc_list = evaluate(dataset)
    if print_acc:
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error})
    return accuracy, acc_list, completions


def eval_humaneval(scored_results, print_acc=False):
    output_list = []
    random_output_list = []
    dataset_in = []
    from typing import Iterable, Dict
    import gzip
    def stream_jsonl(filename: str) -> Iterable[Dict]:
        if filename.endswith(".gz"):
            with open(filename, "rb") as gzfp:
                with gzip.open(gzfp, 'rt') as fp:
                    for line in fp:
                        if any(not x.isspace() for x in line):
                            yield json.loads(line)
        else:
            with open(filename, "r") as fp:
                return json.load(fp)
                # for line in fp:
                #     if any(not x.isspace() for x in line):
                #         yield json.loads(line)

    def extract_code(text, entry_point):
        # 正则表达式匹配代码块
        code_block_pattern = re.compile(
            rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
        if code_block is None:
            code_block_pattern = re.compile(
                rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
            )
            code_block = code_block_pattern.search(text)
        if code_block is None:
            code_block_pattern = re.compile(
                r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
            )
            code_block = code_block_pattern.search(text)

        if code_block is not None:
            return code_block.group(1)

        # if no code block is found, assume the LM is simply filling the code
        return textwrap.indent(text, " " * 4)

    for example in stream_jsonl("/mnt/data/user/tc_agi/ylf/eval_data/humaneval/HumanEval.jsonl.gz"):
        signature = re.search(
            rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
        ).group(1)
        description = "\n".join(
            [
                line.strip()
                for line in re.search(
                rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
            )
                .group(1)
                .split("\n")
            ]
        )
        prompt = (
            f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
            f"{description}\n"
            f"{example['prompt']}"
        )
        dataset_in.append((example["task_id"], prompt, example["entry_point"]))

    # dataset_in = dataset_in
    output_filepath = os.path.join('/data/results', "samples.jsonl")
    f_output = jsonlines.Writer(open(output_filepath, "w", encoding="utf-8"))

    for _, result in zip(dataset_in, scored_results):
        # use humanevalpack prompt
        task_id, prompt, entry_point = _
        # print(result)
        answer = result["response"]
        for cur_response in [answer]:
            # cur_response = cur_response.split("```")[0]
            # answer = extract_code(cur_response, entry_point)
            # gen_jobjs = {"task_id": task_id, "completion": answer, "response": cur_response}
            gen_jobjs = {"task_id": task_id, "completion": answer, "response": cur_response}
            # print(gen_jobjs)
            # print("\n"*2)
            output_list.append(gen_jobjs)
    # print(output_list[0])
    # print("---------------------------------------------")
    # print(output_list[1])
    assert len(output_list) == 164

    from human_eval.evaluate_functional_correctness import entry_point
    result, acc_list = entry_point(output_list)

    if print_acc:
        print("accuracy:", result)
    return result["pass@1"] * 100, acc_list, [result["response"] for result in scored_results]

def eval_theoremqa(scored_results, print_acc=False, all_problems=None):
    from theorem_qa_utils import match_answer,postprocess_number,TheoremqaTask

    completions = [result["response"] for result in scored_results]
    correct = []
    assert len(all_problems) == len(completions), f"{len(all_problems)}\n{len(completions)}"
    for example, model_output in zip(all_problems, completions):
        _, prediction = match_answer(model_output)
        prediction = postprocess_number(prediction)
        verifier = TheoremqaTask(id=example["id"],
                                 prompt=example["Question"],
                                 reference=example["Answer"],
                                 answer_type=example["Answer_type"])
        acc = verifier.success(prediction)
        correct.append(acc)
    total = len(all_problems)
    acc = math.fsum(correct) / total * 100
    if print_acc:
        print("Overall Accuracy = {}/{} = {:.4f}".format(math.fsum(correct), total, acc))
    return acc, correct, None


def main(ckpt_path, local_rank):
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/user/tc_agi/user/lwd/llemma-7b')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['[POSITIVE]', '[NEGATIVE]', '[PLACE]', '[REQUEST]']})
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    # model = model.resize_token_embeddings(len(tokenizer))
    # model = AutoModelForCausalLM.from_pretrained('/data/mistral-math',torch_dtype=torch.bfloat16)
    ds_engine = deepspeed.init_inference(model,
                                         mp_size=8,
                                         dtype=torch.bfloat16,
                                         checkpoint=None)
    model = ds_engine.module
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    rating2label = {1: tokenizer.get_vocab()[tokenizer.tokenize('[POSITIVE]')[0]],
                    0: tokenizer.get_vocab()[tokenizer.tokenize('[NEGATIVE]')[0]],
                    -1: tokenizer.get_vocab()[tokenizer.tokenize('[NEGATIVE]')[0]], }

    file_name_list = ['math', 'gsm8k', 'humaneval', 'mbpp']
    for file_name in file_name_list:
        if local_rank == 0:
            print('PRM ', file_name, '...')
        if file_name == 'math':
            dataset1 = json.load(open('gen_data/_math_all1.json'))
            dataset2 = json.load(open('gen_data/_math_all2.json'))
            dataset = dataset1 + dataset2
        else:
            dataset = json.load(open(f'gen_data/_{file_name}.json'))

        process_list = []
        iii = 2
        is_print = 1
        for data in dataset:

            # problem = data['prompt'].replace("<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",'').replace('[/INST]  </s>','').strip(' ')
            if file_name in ['gsm8k', 'math']:
                problem = data['prompt'].replace(
                    '[INST] You are a mathematician, you are supposed to answer the given question.\nQuestion:',
                    '').replace("\n[/INST]Let's think step by step. Step 1:", '').strip(' ')  # math
            elif file_name == 'mbpp':
                problem = data['prompt'].replace(
                    '[INST]You are a helpful, respectful and honest assistant. Present code in ```python```\n',
                    '').replace('[/INST] Step 1:', '').strip(' ')  # mbpp
            else:
                problem = data['prompt'].replace('[INST]', '').replace('[/INST] Step 1:', '').strip(' ')  # humaneval

            instruct = f"Problem:\n{problem}\nSolution:\nStep 1:"
            # instruct = problem

            inputs = tokenizer(instruct, add_special_tokens=False)
            index = [0] * len(inputs['input_ids'])

            response = data['response']
            if file_name in ['mbpp', 'humaneval']:
                responss_id = response.find("```python\n")
                response = response[responss_id:]
            f = re.finditer('Step|# Step', response)
            scan_list = []
            for it in f:
                scan_list.append(it.span()[0])
            if scan_list != []:
                res_steps = []
                for idx, scan in enumerate(scan_list):
                    if idx == 0:
                        res_steps.append(response[:scan])
                        if len(scan_list) > 1:
                            res_steps.append(response[scan:scan_list[idx + 1]])
                        else:
                            res_steps.append(response[scan:])
                    elif idx == len(scan_list) - 1:
                        res_steps.append(response[scan:])
                    else:
                        res_steps.append(response[scan:scan_list[idx + 1]])
                res_steps = [s for s in res_steps if s != '']
            else:
                res_steps = [response]

            if is_print:
                is_print = 0
                print(res_steps)

            if iii > 0:
                print(res_steps)
                iii -= 1
            is_first = 1
            for idx, step in enumerate(res_steps):
                if is_first:
                    is_first = 0
                    step_inputs = tokenizer.encode(step.strip(),
                                                   add_special_tokens=False) + tokenizer.convert_tokens_to_ids(
                        ['[REQUEST]'])
                else:
                    step_inputs = tokenizer.convert_tokens_to_ids(['[PLACE]']) + tokenizer.encode(step.strip(),
                                                                                                  add_special_tokens=False) + tokenizer.convert_tokens_to_ids(
                        ['[REQUEST]'])
                index += [0] * len(step_inputs)
                index[-1] = 1
                inputs['input_ids'] += step_inputs
                inputs['attention_mask'] += [1] * len(step_inputs)

            if len(inputs['input_ids']) > 1024:
                continue
            inputs['index'] = index
            inputs['problem'] = problem
            inputs['response'] = data['response']
            process_list.append(inputs)

        dataset = {
            'input_ids': [i['input_ids'] for i in process_list],
            'attention_mask': [i['attention_mask'] for i in process_list],
            'index': [i['index'] for i in process_list],
            'prompt': [i['problem'] for i in process_list],
            'response': [i['response'] for i in process_list],
        }

        dataset = Dataset.from_dict(dataset)

        def collator_fn(features):
            batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
            batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
            batch_index = [torch.LongTensor(feature["index"]) for feature in features]

            batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
            batch_index = pad_sequence(batch_index, batch_first=True, padding_value=0)
            # assert batch_input_ids.shape[-1]==batch_labels.shape[-1]
            # if batch_input_ids.shape[-1]>1024:
            #     batch_input_ids = batch_input_ids[:,:1024]
            #     batch_attention_mask = batch_attention_mask[:, :1024]
            #     batch_index = batch_index[:, :1024]
            #     print('***************')
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "index": batch_index,
                "prompt": [feature['prompt'] for feature in features],
                "response": [feature['response'] for feature in features]
            }

        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, 4, collate_fn=collator_fn)

        results = {
            'prompt': [],
            'response': [],
            'scores': [],
            'step_scores': []
        }

        for data in dataloader:
            with torch.no_grad():
                logits = model(input_ids=data['input_ids'].cuda(), attention_mask=data['attention_mask'].cuda()).logits
            index = data['index'].cuda()
            # cur_index = torch.where(index==-100,0,index)
            # select_logits = torch.gather(logits,cur_index,dim=1)
            # mask = torch.ones_like(logits)
            # mask[:,:,list(rating2label.values())] = 0
            # mask *= -1e6
            # logits += mask
            logits = logits[:, :, list(rating2label.values())[:2]]
            probs = torch.softmax(logits, dim=-1)
            probs = probs[:, :, 0]
            # scores = torch.log(probs)
            scores = probs
            # print(scores.shape,index.shape)

            scores = torch.where(index == 0, 1e4, scores)
            final_scores = scores.min(-1).values
            results['prompt'].extend(data['prompt'])
            results['response'].extend(data['response'])
            results['scores'].extend(final_scores.tolist())
            results['step_scores'].extend([[ss for ss in score if ss != 1e4] for score in scores.tolist()])
            # results['step_scores'].extend([[c for c in s if c!=1e5] for s in scores.tolist()])

        if local_rank == 0:
            if file_name == 'gsm8k':
                print('Eval gsm8k...')
                eval_gsm8k(results)
            elif file_name == 'math':
                print('Eval math...')
                eval_math(results)
            elif file_name == 'mbpp':
                print('Eval mbpp...')
                eval_mbpp(results)
            elif file_name == 'humaneval':
                print('Eval humaneval...')
                eval_humaneval(results)
            else:
                raise NotImplementedError

        # with open(ckpt_path + ".json", "w") as f:
        #     json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--world_info", type=str, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("--enable_each_rank_log", type=bool, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    print(args.ckpt_path)
    main(args.ckpt_path, args.local_rank)
