import sys
import time

sys.path.append("./../")
import argparse
import json
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODELS, base_prompt, get_train_test_split, set_seed


class Probe:
    def __init__(self, X, y, device):
        self.X = X
        self.mu = self.X.mean(0, keepdim=True)
        self.sd = self.X.std(0, keepdim=True) + 1e-6
        self.y = y
        self.device = device
        self.w = torch.zeros(X.shape[1], device=device, requires_grad=True)
        self.b = torch.zeros(1, device=device, requires_grad=True)

    def fit(self, epochs=300, lr=0.05, l2=1e-3):
        Xn = (self.X - self.mu) / self.sd
        pos = self.y.sum().clamp(min=1)
        pos_w = (len(self.y) - pos) / pos  # up-weight the rare positive (mistake) class
        opt = torch.optim.AdamW([self.w, self.b], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(epochs):
            opt.zero_grad()
            logit = Xn @ self.w + self.b
            loss = loss_fn(logit, self.y) + l2 * (self.w @ self.w)
            loss.backward()
            opt.step()
        self.w = self.w.detach()
        self.b = self.b.detach()

    def score(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b

@torch.no_grad()
def extract(model, tokenizer, ds):
    name = "model"
    if hasattr(model.model, "language_model"):
        name = "model.language_model"

    hooks, cache = [], {}

    def attn_hook(module, input, layer):
        cache[layer] = input[0].detach().cpu()

    for i, l in enumerate(model.get_submodule(name).layers):
        hooks.append(l.mlp.register_forward_pre_hook(partial(attn_hook, layer=i)))

    for sample in tqdm(ds):
        inputs, turn_idxs = base_prompt(sample, tokenizer)
        inputs = inputs.to(model.device)
        hs = model(**inputs, output_hidden_states=True, do_sample=False)[
            "hidden_states"
        ]
        acts = []
        for i in range(max(cache)):
            acts.append(torch.stack([cache[i][0, idx - 1] for idx in turn_idxs[1:]]))
            acts.append(
                torch.stack([hs[i][0, idx - 1].detach().cpu() for idx in turn_idxs[1:]])
            )
        sample["acts"] = torch.stack(acts).float()
        del inputs, hs
        torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()
    return ds


def train_probe(train_data, test_data, device):
    per_layer = []
    for l in tqdm(range(train_data[0]["acts"].shape[0])):
        Xtr = torch.cat([s["acts"][l] for s in train_data]).to(device)
        ytr = torch.cat(
            [
                torch.tensor(
                    [float(i == s["mistake_step"]) for i in range(s["acts"].shape[1])]
                )
                for s in train_data
            ]
        ).to(device)

        probe = Probe(Xtr, ytr, device)
        probe.fit()
        agent_hit, step_hit = 0, 0
        for sample in test_data:
            sc = probe.score(sample["acts"][l].to(device))
            pred = int(sc.argmax())
            gt = sample["mistake_step"]
            step_hit += pred == gt
            pred_name = sample["history"][pred]["name"]
            if sample["mistake_agent"].startswith(pred_name) or pred_name.startswith(
                sample["mistake_agent"]
            ):
                agent_hit += 1
        per_layer.append(
            {
                "layer": l // 2,
                "comp": "attn" if l % 2 == 0 else "mlp",
                "agent_acc": agent_hit / len(test_data),
                "step_acc": step_hit / len(test_data),
            }
        )
    return per_layer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="qwen0.5", type=str)
    p.add_argument("--data", "-d", default="algo", type=str)
    p.add_argument("--k", "-k", default=5, type=int)
    p.add_argument("--device", default=0, type=int)
    args = p.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    os.makedirs("./results", exist_ok=True)
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], dtype=torch.bfloat16
    ).to(args.device)
    ds = json.load(open("./../data/processed_data.json"))[args.data]
    train_data, test_data = get_train_test_split(ds, args.data)

    start = time.time()

    train_data = extract(model, tokenizer, train_data)
    test_data = extract(model, tokenizer, test_data)
    res = train_probe(train_data, test_data, args.device)
    total_time = time.time() - start
    json.dump(res, open(f"./results/{args.model}_{args.data}.json", "w"), indent=4)

    summary = {}
    if os.path.exists("./results/summary.json"):
        summary = json.load(open("./results/summary.json"))

    cur_res = [(r["step_acc"], i) for i, r in enumerate(res)]
    summary[f"{args.model}_{args.data}"] = res[sorted(cur_res, reverse=True)[0][-1]]
    summary[f"{args.model}_{args.data}"]["total_time"] = total_time
    json.dump(
        {k: summary[k] for k in sorted(summary)},
        open("./results/summary.json", "w"),
        indent=4,
    )
