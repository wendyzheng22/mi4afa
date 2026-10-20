import random
import sys
import time

from tqdm import tqdm

sys.path.append("./../")
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODELS, base_prompt, get_train_test_split, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model name",
        default="qwen0.5",
        choices=list(MODELS.keys()),
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="data set name",
        default="algo",
        choices=["algo", "craft", "smolagents", "magentic-one", "alfagent", "mathchat"],
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt type",
        default="vanilla",
        choices=["vanilla", "with_step", "sequential", "step_by_step", "random"],
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    return parser.parse_args()

def random_step(sample):
    random_step = random.randint(0, len( sample["history"]) - 1)
    return {
        "sample": sample,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][random_step]["name"],
        "pred_step": random_step,
    }

def vanilla_prompt(sample, tokenizer, model):
    inputs, _ = base_prompt(sample, tokenizer, with_step=False)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    response = response.split("Respond ONLY with a number")[-1].strip()
    if response.startswith("assistant"):
        response = response[len("assistant") :].strip()

    step = -1
    for digit in re.findall(r"(\d+)", response):
        if int(digit) == int(sample["mistake_step"]):
            step = int(sample["mistake_step"])
            break
        else:
            if step == -1 and int(digit) < len(sample["history"]):
                step = int(digit)

    del inputs, out
    torch.cuda.empty_cache()
    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][step]["name"] if step != -1 else None,
        "pred_step": step,
    }


def with_step_prompt(sample, tokenizer, model):
    inputs, _ = base_prompt(sample, tokenizer, with_step=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    response = response.split("Respond ONLY with a number")[-1].strip()
    if response.startswith("assistant"):
        response = response[len("assistant") :].strip()

    step = -1
    for digit in re.findall(r"(\d+)", response):
        if int(digit) == int(sample["mistake_step"]):
            step = int(sample["mistake_step"])
            break
        else:
            if step == -1 and int(digit) < len(sample["history"]):
                step = int(digit)

    del inputs, out
    torch.cuda.empty_cache()
    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][step]["name"] if step != -1 else None,
        "pred_step": step,
    }
    
def step_by_step(sample, tokenizer, model):
    prompt_chunks = []
    prefix = ("You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        "Your task is to predict if the current step is incorrect or contains a failure\n")
    postfix = "Respond ONLY with YES or NO\n"
    prompt_chunks  =  [prefix + (
        f"The problem is: {sample['question']}\n"
        f"The answer for the problem is: {sample['ground_truth']}\n"
        f"Here's the conversation history:"
    )]

    pred_step = -1
    for i, entry in enumerate(sample["history"]):
        step = f"{entry.get("name", 'Unknown Agent').strip()}: {entry.get('content', '').strip()}"
        prompt_chunks.append(step)
        messages = [
            {
                "role": "system",
                "content": "You are a precise judge that identifies the most critical mistake in multi-agent conversations that fail to derive the correct answer.",
            },
            {"role": "user", "content": "\n".join(prompt_chunks + [postfix])},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        response = response.split(postfix.strip())[-1].strip()
        if response.startswith("assistant"):
            response = response[len("assistant") :].strip()
        if "yes" in response.lower():
            pred_step = i
            break

    del inputs, out
    torch.cuda.empty_cache()
    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][pred_step]["name"] if pred_step != -1 else None,
        "pred_step": pred_step,
    }


def sequential_prompt(sample, tokenizer, model):
    prefix = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        "Your task is to predict the agent that made the mistake directly responsible for the wrong solution.\n"
    )
    postfix = "Focus on errors that clearly derail the process.\nRespond ONLY with the agent name\n"
    inputs, _ = base_prompt(
        sample, tokenizer, prefix=prefix, postfix=postfix, with_step=True
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    response = response.split("Respond ONLY with the agent name")[-1].strip()

    agent = None
    for entry in sample["history"]:
        if (
            entry["name"].lower() in response.lower()
            or entry["name"].split(" ")[0].lower() in response.lower()
        ):
            agent = entry["name"]

    del inputs, out
    torch.cuda.empty_cache()

    step = -1
    if agent is not None:
        prefix = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
            f"Your task is to predict the step that {agent} made the mistake directly responsible for the wrong solution.\n"
        )
        postfix = "Focus on errors that clearly derail the process.\nRespond ONLY with a number\n"
        inputs, _ = base_prompt(
            sample, tokenizer, prefix=prefix, postfix=postfix, with_step=True
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        response = response.split("Respond ONLY with a number")[-1].strip()

        for digit in re.findall(r"(\d+)", response):
            if int(digit) == int(sample["mistake_step"]):
                step = int(sample["mistake_step"])
        del inputs, out
        torch.cuda.empty_cache()

    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": agent,
        "pred_step": step,
    }


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    set_seed(42)

    args = parse_args()
    args.device = f"cuda:{args.device}"

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    tokenizer.pad_token = tokenizer.eos_token
    if args.model == "qwen72":
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model], device_map="balanced"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model], dtype=torch.bfloat16
        ).to(args.device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    ds = json.load(open("./../data/processed_data.json"))[args.data]

    _, test_data = get_train_test_split(ds, args.data)

    start = time.time()
    all_out, agent_acc, step_acc = [], 0, 0
    for sample in tqdm(test_data, desc=f"{args.model}_{args.data}_{args.prompt}"):
        if args.prompt == "vanilla":
            all_out.append(vanilla_prompt(sample, tokenizer, model))
        elif args.prompt == "with_step":
            all_out.append(with_step_prompt(sample, tokenizer, model))
        elif args.prompt == "sequential":
            all_out.append(sequential_prompt(sample, tokenizer, model))
        elif args.prompt == "step_by_step":
            all_out.append(step_by_step(sample, tokenizer, model))
        elif args.prompt == "random":
            all_out.append(random_step(sample))
            
        if all_out[-1]["pred_agent"] is None:
            continue
        if all_out[-1]["pred_agent"].startswith(all_out[-1]["gt_agent"]) or all_out[-1][
            "gt_agent"
        ].startswith(all_out[-1]["pred_agent"]):
            agent_acc += 1
        if all_out[-1]["pred_step"] == all_out[-1]["gt_step"]:
            step_acc += 1

    total_time = time.time() - start
    agent_acc = agent_acc / len(test_data)
    step_acc = step_acc / len(test_data)
    summary = {}
    if os.path.exists("./results/summary.json"):
        summary = json.load(open("./results/summary.json"))
    summary[f"{args.model}_{args.data}_{args.prompt}"] = {
        "agent_acc": agent_acc,
        "step_acc": step_acc,
        "time": total_time,
    }
    json.dump(
        {k: summary[k] for k in sorted(summary)},
        open("./results/summary.json", "w"),
        indent=4,
    )
    json.dump(
        all_out,
        open(f"./results/{args.model}_{args.data}_{args.prompt}.json", "w"),
        indent=4,
    )
