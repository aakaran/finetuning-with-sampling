import os, json, time
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


from contextlib import nullcontext
from glob import glob
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets
from pathlib import Path


import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt


from constants import *

import re

from grader_utils.sci_grader import parse_answer_gpqa, grade_answer, same_balanced_equation, same_balanced_equation_olmo

def format_prompt(text, model_type, tokenizer):
    if model_type == "base":
        format_str = text
    elif model_type == "chat":
        answer_context = [{"role": "user", "content": text}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    return format_str

class vLLMAutoregressiveSampler:
    def __init__(self, model_name, model_type, device="cuda"):
        self.llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_type = model_type
        self.device = device
        
        if hasattr(self.llm.llm_engine.model_config, 'max_model_len'):
             self.block_size = self.llm.llm_engine.model_config.max_model_len
        else:
             self.block_size = 2048

def get_logprobs_vLLM(p: vLLMAutoregressiveSampler, question, current):
    context_str = format_prompt(question, p.model_type, p.tokenizer)
    context_tokens = p.tokenizer.encode(context_str)
    
    full_sequence_ids = context_tokens + current
    c = len(context_tokens)

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    
    output = p.llm.generate(
        TokensPrompt(prompt_token_ids=full_sequence_ids),
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    prompt_logprobs = output[0].prompt_logprobs
    relevant_logprobs = prompt_logprobs[c:]

    log_likelihood = []
    for i, logprob_dict in enumerate(relevant_logprobs):
        if logprob_dict:
            token_id = current[i]
            log_likelihood.append(logprob_dict[token_id].logprob)

    if not log_likelihood:
        return float("-inf"), 0.0, 0.0

    return sum(log_likelihood)/len(log_likelihood), 0.0, 0.0

def proposal_vLLM(p: vLLMAutoregressiveSampler, question, solution, current, ext_len):
    current_txt = p.tokenizer.decode(current, skip_special_tokens=True)
    
    context_prompt = SEQ_PROMPT_SCI.format(PROBLEM=question, SOLUTION=solution, RESPONSE=current_txt)
    formatted_prompt = format_prompt(context_prompt, p.model_type, p.tokenizer)
    
    context_tokens = p.tokenizer.encode(formatted_prompt)
    prefix_ids = context_tokens + current
    
    sampling_params = SamplingParams(
        max_tokens=ext_len,
        temperature=0.6,
        stop_token_ids=[p.tokenizer.eos_token_id],
        repetition_penalty=1.05,
        logprobs=0 # We need logprobs of generated tokens
    )
    
    output = p.llm.generate(
        TokensPrompt(prompt_token_ids=prefix_ids),
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    generated_output = output[0].outputs[0]
    new_tokens = list(generated_output.token_ids)


    new_log_probs_norm = []
    for i, tok_id in enumerate(new_tokens):
        lp_dict = generated_output.logprobs[i]
        lp_obj = lp_dict.get(tok_id, None)
        if lp_obj is None:
            raise RuntimeError(f"Missing generated token_id={tok_id} in logprobs at step i={i}.")
        new_log_probs_norm.append(float(lp_obj.logprob))
            
    prop = current + new_tokens

    return prop, new_log_probs_norm


def get_proposal_logprobs_vLLM(p: vLLMAutoregressiveSampler, question, solution, current, prop):
    current_txt = p.tokenizer.decode(current, skip_special_tokens=True)
    
    context_prompt = SEQ_PROMPT_SCI.format(PROBLEM=question, SOLUTION=solution, RESPONSE=current_txt)
    formatted_prompt = format_prompt(context_prompt, p.model_type, p.tokenizer)
    
    context_tokens = p.tokenizer.encode(formatted_prompt)
    full_sequence_ids = context_tokens + prop

    c = len(context_tokens) + len(current)

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    
    output = p.llm.generate(
        TokensPrompt(prompt_token_ids=full_sequence_ids),
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    prompt_logprobs = output[0].prompt_logprobs

    relevant_logprobs = prompt_logprobs[c:]

    full_logprobs_per_step = []
    for step_lp_dict in relevant_logprobs:
        cur_step = {}
        for tok_id, lp_obj in step_lp_dict.items():
            cur_step[int(tok_id)] = float(lp_obj.logprob)
        full_logprobs_per_step.append(cur_step)


    return full_logprobs_per_step



def proj_samp_vLLM(p: vLLMAutoregressiveSampler, question, solution, mcmc_steps, max_new_tokens, block_num=16):
    gen = []
    log_probs_norm = []
    trace = []

    print(f"Max new tokens: {max_new_tokens}")
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(f"Jump size: {jump_size}")
    
    attempts = 0
    acceptances = 0

    # Main Block Loop
    for _ in tqdm(range(block_num), desc="Blocks"):
        trace_round = []
        # Initial proposal for this block
        gen_new, lp_norm = proposal_vLLM(p, question, solution, gen, ext_len=jump_size)
        # full_token_teacher_logprobs = get_proposal_logprobs_vLLM(p, question, solution, gen, gen_new)
        trace_round.append((gen, gen_new))

        gen = gen_new
        del gen_new


        # teacher_logprobs.extend(full_token_teacher_logprobs)
        log_probs_norm.extend(lp_norm)
        
        # Score current trajectory
        target_log_prob_cur = get_logprobs_vLLM(p, question, gen)

        # MCMC Loop
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)
            if t == 0: continue
            
            idx = random.randint(0, t - 1)
            
            prop, log_prob_prop = proposal_vLLM(p, question, solution, gen[:idx], ext_len=t - idx)
            
            target_log_prob_prop = get_logprobs_vLLM(p, question, prop)
            
            # Validation logic
            s = len(prop)
            
            if target_log_prob_prop[0] > target_log_prob_cur[0]:
                trace_round.append((gen[:idx], prop))

                acceptances += 1
                gen = prop
                
                # Update log norms
                # Ensure lists match size before assignment
                if len(log_probs_norm) < s:
                    log_probs_norm.extend([0.0] * (s - len(log_probs_norm)))
                elif len(log_probs_norm) > s:
                    log_probs_norm = log_probs_norm[:s]
                    
                log_probs_norm[idx:] = log_prob_prop
                target_log_prob_cur = target_log_prob_prop

        print(p.tokenizer.decode(gen, skip_special_tokens=True))
        print("---------------------")

        trace.append(trace_round)


        # EOS Check
        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            # teacher_logprobs = teacher_logprobs[:eos_idx+1]
            trace.append(eos_idx)
            
            # If we hit EOS, we might want to return early or just stop extending
            # For this logic, we return early as per original script
            return gen, target_log_prob_cur, log_probs_norm, trace

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0
    return gen, target_log_prob_cur, log_probs_norm, trace



def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())



MCQ_PROMPT = "You are an expert chemist. Given a question and four options, think step by step, and then return the correct letter: A, B, C, or D. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}').\n"
MCQ_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."

BALANCE_PROMPT = "You are an expert chemist. Given a chemical equation, please balance the equation and maintain the order of reactants and products as given. Here is a unbalanced chemical equation:\n"
BALANCE_COT = "\nExplain your reasoning, thinking step by step, and return the balanced equation on one line. Keep reactants and products in the SAME ORDER as provided."


from typing import Dict, Tuple, List, Optional

def extract_unbalanced_from_question(q: str) -> Optional[str]:
    for line in q.splitlines():
        if ("=" in line) or ("->" in line):
            return line.strip()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "sci_boost/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "olmo"])
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--num_batches", action = "store", type = int, default = 15)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--block_num", action = "store", type = int, default = 58)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    args = parser.parse_args()


    random.seed(0)

    model = args.model
    device = args.device
    mcmc_steps = args.mcmc_steps
    block_num = args.block_num

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)
    out_path = os.path.join(save_str, "boosted_" + str(args.batch_idx)+ ".jsonl")
    trace_out_path = os.path.join(save_str, "trace_boosted_" + str(args.batch_idx)+ ".jsonl")


    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B-Instruct"
    if model =="olmo":
        model_str = "allenai/Olmo-3-7B-Instruct"

    balance_grader = same_balanced_equation_olmo if model == "olmo" else same_balanced_equation

    with open("sci_data/train_data.jsonl", "r", encoding="utf-8") as f:
        train_set = [json.loads(line) for line in f if line.strip()]

    p = vLLMAutoregressiveSampler(model_name=model_str, model_type="chat", device=device)

    num_batches = args.num_batches
    all_idxs = [i for i in range(len(train_set))]
    chunk_size = (len(all_idxs) + num_batches - 1) // num_batches
    start = args.batch_idx * chunk_size
    end = min(start + chunk_size, len(all_idxs))
    batch_idxs = all_idxs[start:end]

    for i in tqdm(batch_idxs):
        type_q = train_set[i]["type"]

        if type_q.startswith("mcq"):
            opt_lines = []
            for lbl, txt in zip(train_set[i]["choices"].get("label", []), train_set[i]["choices"].get("text", [])):
                opt_lines.append(f"{lbl}) {txt}")

            question = MCQ_PROMPT + train_set[i]["question"] + "\n\nOptions:\n" + "\n".join(opt_lines) + MCQ_COT
            solution = train_set[i]["chosen"]

        else:
            question = BALANCE_PROMPT + extract_unbalanced_from_question(train_set[i]["question"])+BALANCE_COT
            solution = train_set[i]["chosen"]


        t0 = time.time()
        boosted, avg_log_prob, lps, trace  = proj_samp_vLLM(p, question, solution, mcmc_steps=mcmc_steps, max_new_tokens=1856, block_num=block_num)
        dt = time.time() - t0

        boosted_rollout = p.tokenizer.decode(boosted, skip_special_tokens=True)

        if type_q.startswith("mcq"):
            answer = parse_answer_gpqa(boosted_rollout)
            is_correct = int(grade_answer(answer, train_set[i]["answerKey"]))
            has_boxed = "boxed" in boosted_rollout

        else:
            out = same_balanced_equation_olmo(boosted_rollout, train_set[i]["answer"])
            answer = out["equation_1"]
            is_correct = out["same"]


        rec = {
            "idx": i,
            "prompt": question,
            "solution": solution,
            "gen": boosted_rollout,
            "target_log_prob_cur": float(avg_log_prob[0]),
            "token_logprobs": [float(x) for x in lps],
            "runtime_sec": dt,
            "answer": answer,
            "is_correct": is_correct,
        }

        trace_rec = {
            "idx": i,
            "prompt": question,
            "solution": solution,
            "trace": trace,
        }
        append_jsonl(trace_out_path, trace_rec)

        append_jsonl(out_path, rec)

            
