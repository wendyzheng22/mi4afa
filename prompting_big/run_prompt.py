import sys
import time

sys.path.append("./../")
import argparse
import json
import os
import re

from cursor_sdk import Agent, AgentOptions
from tqdm import tqdm
from utils import base_prompt, get_train_test_split, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model name",
        choices=[
            "gpt-5.6-sol",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-opus-5",
            "gpt-5.6-terra",
        ],
    )
    parser.add_argument(
        "-d", "--data", type=str, help="data set name", choices=["algo", "craft"]
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt type",
        choices=["vanilla", "with_step", "sequential"],
    )
    return parser.parse_args()


def vanilla_prompt(sample, model):
    messages = base_prompt(sample, None, with_step=False, prompt_only=True)
    prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=os.environ["CURSOR_API_KEY"],
            model=model,
        ),
    )
    response = result.result
    step = -1
    for digit in re.findall(r"(\d+)", response):
        if int(digit) == int(sample["mistake_step"]):
            step = int(sample["mistake_step"])
            break
        else:
            if step == -1 and int(digit) < len(sample["history"]):
                step = int(digit)
    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][step]["name"] if step != -1 else None,
        "pred_step": step,
    }


def with_step_prompt(sample, model):
    messages = base_prompt(sample, None, with_step=True, prompt_only=True)
    prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=os.environ["CURSOR_API_KEY"],
            model=model,
        ),
    )
    response = result.result
    step = -1
    for digit in re.findall(r"(\d+)", response):
        if int(digit) == int(sample["mistake_step"]):
            step = int(sample["mistake_step"])
            break
        else:
            if step == -1 and int(digit) < len(sample["history"]):
                step = int(digit)
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
    prefix = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        "Your task is to predict if the current step is incorrect or contains a failure\n"
    )
    postfix = "Respond ONLY with YES or NO\n"
    prompt_chunks = [
        prefix
        + (
            f"The problem is: {sample['question']}\n"
            f"The answer for the problem is: {sample['ground_truth']}\n"
            f"Here's the conversation history:"
        )
    ]

    step = -1
    for i, entry in enumerate(sample["history"]):
        step = f"{entry.get('name', 'Unknown Agent').strip()}: {entry.get('content', '').strip()}"
        prompt_chunks.append(step)
        messages = [
            {
                "role": "system",
                "content": "You are a precise judge that identifies the most critical mistake in multi-agent conversations that fail to derive the correct answer.",
            },
            {"role": "user", "content": "\n".join(prompt_chunks + [postfix])},
        ]
        result = Agent.prompt(
            messages,
            AgentOptions(
                api_key=os.environ["CURSOR_API_KEY"],
                model=model,
            ),
        )
        response = result.result
        if "yes" in response.lower():
            step = i
            break
    return {
        "sample": sample,
        "output": response,
        "gt_agent": sample["mistake_agent"].strip(),
        "gt_step": int(sample["mistake_step"]),
        "pred_agent": sample["history"][step]["name"] if step != -1 else None,
        "pred_step": step,
    }


def sequential_prompt(sample, model):
    prefix = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        "Your task is to predict the agent that made the mistake directly responsible for the wrong solution.\n"
    )
    postfix = "Focus on errors that clearly derail the process.\nRespond ONLY with the agent name\n"
    messages = base_prompt(
        sample, None, prefix=prefix, postfix=postfix, with_step=True, prompt_only=True
    )
    prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])

    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=os.environ["CURSOR_API_KEY"],
            model=model,
        ),
    )
    response = result.result

    agent = None
    for entry in sample["history"]:
        if (
            entry["name"].lower() in response.lower()
            or entry["name"].split(" ")[0].lower() in response.lower()
        ):
            agent = entry["name"]

    step = -1
    if agent is not None:
        prefix = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
            f"Your task is to predict the step that {agent} made the mistake directly responsible for the wrong solution.\n"
        )
        postfix = "Focus on errors that clearly derail the process.\nRespond ONLY with a number\n"
        messages = base_prompt(
            sample,
            None,
            prefix=prefix,
            postfix=postfix,
            with_step=True,
            prompt_only=True,
        )
        prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=os.environ["CURSOR_API_KEY"],
                model=model,
            ),
        )
        response = result.result
        for digit in re.findall(r"(\d+)", response):
            if int(digit) == int(sample["mistake_step"]):
                step = int(sample["mistake_step"])
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

    ds = json.load(open("./../data/processed_data.json"))[args.data]
    _, test_data = get_train_test_split(ds, args.data)

    summary = {}
    if os.path.exists("./results/summary.json"):
        summary = json.load(open("./results/summary.json"))

    start = time.time()
    all_out, agent_acc, step_acc = [], 0, 0
    for sample in tqdm(test_data, desc=f"{args.model}_{args.data}_{args.prompt}"):
        if args.prompt == "vanilla":
            all_out.append(vanilla_prompt(sample, args.model))
        elif args.prompt == "with_step":
            all_out.append(with_step_prompt(sample, args.model))
        elif args.prompt == "sequential":
            all_out.append(sequential_prompt(sample, args.model))

        if all_out[-1]["pred_agent"] is None:
            continue
        if all_out[-1]["pred_agent"].startswith(all_out[-1]["gt_agent"]) or all_out[-1][
            "gt_agent"
        ].startswith(all_out[-1]["pred_agent"]):
            agent_acc += 1
        if all_out[-1]["pred_step"] == all_out[-1]["gt_step"]:
            step_acc += 1
        time.sleep(1)

    total_time = time.time() - start
    agent_acc = agent_acc / len(test_data)
    step_acc = step_acc / len(test_data)
    summary[f"{args.model}_{args.data}_{args.prompt}"] = {
        "agent_acc": agent_acc,
        "step_acc": step_acc,
        "time": total_time,
    }
    json.dump(
        all_out,
        open(f"./results/{args.model}_{args.data}_{args.prompt}.json", "w"),
        indent=4,
    )
    json.dump(
        {k: summary[k] for k in sorted(summary)},
        open("./results/summary.json", "w"),
        indent=4,
    )
