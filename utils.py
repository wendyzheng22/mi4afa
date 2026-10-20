import os
import random

import numpy as np
import torch
import transformers

ROOT = os.path.dirname(__file__)

MODELS = {
    "qwen0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7": "Qwen/Qwen2.5-7B-Instruct",
    "llama1": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma1": "google/gemma-3-1b-it",
    "gemma4": "google/gemma-3-4b-it",
}

DEFAULT_PREFIX = (
    "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
    "Your task is to predict the step with the mistake that should be directly responsible for the wrong solution.\n"
)
DEFAULT_POSTFIX = (
    "Focus on errors that clearly derail the process.\nRespond ONLY with a number\n"
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)


def base_prompt(sample, tokenizer, prefix=None, postfix=None, with_step=True, prompt_only=False):
    prompt_chunks, turn_idxs = [], []
    if prefix is None:
        prefix = DEFAULT_PREFIX
    if postfix is None:
        postfix = DEFAULT_POSTFIX
    prefix += (
        f"The problem is: {sample['question']}\n"
        f"The answer for the problem is: {sample['ground_truth']}\n"
        f"Here's the conversation history:"
    )
    prompt_chunks.append(prefix)

    for i, entry in enumerate(sample["history"]):
        step = f"{entry.get("name", 'Unknown Agent').strip()}: {entry.get('content', '').strip()}"
        if with_step:
            step = f"{i} - " + step
        prompt_chunks.append(step)
    prompt_chunks.append(postfix)
    messages = [
        {
            "role": "system",
            "content": "You are a precise judge that identifies the most critical mistake in multi-agent conversations that fail to derive the correct answer.",
        },
        {"role": "user", "content": "\n".join(prompt_chunks)},
    ]
    if prompt_only:
        return messages

    tokenized_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    toks = tokenized_prompt["input_ids"].squeeze()
    idx = len(tokenizer(prompt_chunks[0])["input_ids"])
    for chunk in prompt_chunks[1:-1]:
        while (
            not tokenizer.decode(toks[idx : idx + 30], skip_special_tokens=True)
            .strip()
            .startswith(chunk[:30])
        ):
            idx += 1
        turn_idxs.append(idx)
        idx += len(tokenizer(chunk)["input_ids"]) - 5
    idx = toks.shape[0]
    while (
        not tokenizer.decode(toks[idx - 30 : idx], skip_special_tokens=True)
        .strip()
        .endswith(prompt_chunks[-2][-30:])
    ):
        idx -= 1
    turn_idxs.append(idx)
    return tokenized_prompt, turn_idxs


def get_train_test_split(ds, ds_name):
    if not os.path.exists(f"{ROOT}/data/train_test_split_{ds_name}.pt"):
        train_idx = random.sample(list(range(len(ds))), k=int(len(ds) * 0.5))
        test_idx = [i for i in range(len(ds)) if i not in train_idx]
        torch.save([train_idx, test_idx], f"{ROOT}/data/train_test_split_{ds_name}.pt")
    train_idx, test_idx = torch.load(f"{ROOT}/data/train_test_split_{ds_name}.pt")
    train_data = [ds[i] for i in train_idx]
    test_data = [ds[i] for i in test_idx]
    return train_data, test_data
