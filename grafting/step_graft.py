import re
import sys
import time

sys.path.append("./../")
import argparse
import gc
import json
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from utils import MODELS, base_prompt, get_train_test_split, set_seed


def get_prompt(sample, tokenizer):
    inputs, turn_idxs = base_prompt(sample, tokenizer)
    cand_toks = []
    for i in range(len(sample["history"])):
        cand_toks.append(
            torch.tensor(
                [
                    tokenizer(str(i), add_special_tokens=False)["input_ids"][0],
                    tokenizer(" " + str(i), add_special_tokens=False)["input_ids"][0],
                ]
            )
        )
    return inputs, turn_idxs, cand_toks


class StepPatch:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODELS[model_name], dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        self.device = device
        cfg = self.model.config
        if model_name == "gemma4":
            cfg = cfg.get_text_config()
            self.name = "model.language_model"
            self.norm = self.model.model.language_model.norm
        else:
            self.name = "model"
            self.norm = self.model.model.norm
        self.nL = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.hd = self.model.get_submodule(self.name).layers[0].self_attn.head_dim
        self.gain = self.model.get_submodule(self.name).norm.weight.detach().float()
        self.WU = self.model.lm_head.weight.detach().float()
        self.nH = cfg.num_attention_heads
        self.nKV = cfg.num_key_value_heads
        self.hooks = []
        self.reset()

    def reset(self):
        if "cache" in dir(self):
            for k in list(self.cache["attn"].keys()):
                del self.cache["attn"][k]
            for k in list(self.cache["z"].keys()):
                del self.cache["z"][k]
        self.cache = {"z": {}, "attn": {}}
        self.loc_ans, self.turn_idxs, self.cand_toks = [], [], []

    def register_cache_hooks(self):
        def z(module, inputs, kwargs, layer):
            hs = kwargs["hidden_states"].detach()
            cos, sin = kwargs["position_embeddings"]
            B, S, _ = hs.shape
            q = module.q_proj(hs).view(B, S, -1, self.hd).transpose(1, 2)
            k = module.k_proj(hs).view(B, S, -1, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, self.nH // self.nKV)
            sc = torch.matmul(q[:, :, -1:, :], k.transpose(-1, -2)) * module.scaling
            val = torch.softmax(sc.float(), dim=-1)[0, :, 0, :].detach().cpu()
            idxs = list(zip(self.turn_idxs[:-1], self.turn_idxs[1:]))
            turn_z = [val[:, i:j].sum(-1) for i, j in idxs]
            self.cache["z"][layer] = torch.stack(turn_z, dim=-1).cpu()

        def attn(module, inputs, layer):
            weight = module.weight.reshape(-1, self.nH, self.hd).transpose(0, 1)
            inp = inputs[0].reshape(-1, self.nH, self.hd).permute(1, 2, 0)
            self.cache["attn"][layer] = (
                torch.bmm(weight, inp[:, :, -1:]).detach().float().cpu().squeeze()
            )

        for i, l in enumerate(self.model.get_submodule(self.name).layers):
            self.hooks.append(
                l.self_attn.register_forward_pre_hook(
                    partial(z, layer=i), with_kwargs=True
                )
            )
            self.hooks.append(
                l.self_attn.o_proj.register_forward_pre_hook(partial(attn, layer=i))
            )

    def register_grafting_hooks(self, locators, writers):
        def locator_hook_attn(module, inputs, outputs, layer, idxs):
            if inputs[0].shape[1] == 1:  # skip if generating
                return
            self.loc_ans.extend(self.cache["z"][layer][idxs].argmax(-1).float())

        def writer_hook(module, inputs, layer, idxs):
            if len(self.loc_ans) == 0:
                return
            loc_ans, count = torch.tensor(self.loc_ans).unique(return_counts=True)
            final_ans = loc_ans[count.argmax()].item()
            final_tok = self.tokenizer(str(final_ans), add_special_tokens=False)[
                "input_ids"
            ][0]
            dla = self.readout_direction(final_tok, self.cand_toks)

            W = module.weight.reshape(-1, self.nH, self.hd).permute(1, 2, 0)
            bs, seq, _ = inputs[0].shape
            inp = inputs[0].reshape(bs, seq, self.nH, -1)
            inp_norm = inp[:, self.end_idx].norm()
            for i in idxs:
                val = torch.matmul(W[i], dla.to(inp.dtype))
                coeff = self.beta * inp_norm / val.norm()
                inp[:, self.end_idx, i] += coeff * val
            return inp.reshape(bs, seq, -1)

        locator_map, writer_map = {}, {}
        for l in locators:
            if l[0] not in locator_map:
                locator_map[l[0]] = []
            locator_map[l[0]].append(l[1])
        for w in writers:
            if w[0] not in writer_map:
                writer_map[w[0]] = []
            writer_map[w[0]].append(w[1])

        for i, l in enumerate(self.model.get_submodule(self.name).layers):
            if i in locator_map and len(locator_map[i]) > 0:
                self.hooks.append(
                    l.self_attn.o_proj.register_forward_hook(
                        partial(locator_hook_attn, layer=i, idxs=locator_map[i])
                    )
                )
            if i in writer_map:
                self.hooks.append(
                    l.self_attn.o_proj.register_forward_pre_hook(
                        partial(writer_hook, layer=i, idxs=list(set(writer_map[i])))
                    )
                )

    def readout_direction(self, tok_id, cand=None):
        ref = 0
        if cand is not None:
            ref = self.WU[torch.tensor(sorted(set(cand)))].mean(0)
        r = self.gain * (self.WU[tok_id] - ref)
        return r


def eval_pred(pred_step, sample):
    if not pred_step.isdigit() or int(pred_step) >= len(sample["history"]):
        return False, False
    pred_step = int(pred_step)
    pred_name = sample["history"][pred_step]["name"]
    gt_name = sample["mistake_agent"]
    agent_hit = gt_name.startswith(pred_name) or pred_name.startswith(gt_name)
    step_hit = pred_step == sample["mistake_step"]
    return agent_hit, step_hit


def select_heads(sp, ds):
    dla = torch.zeros(sp.nL, sp.nH)
    top1_acc = torch.zeros(sp.nL, sp.nH)
    for sample in tqdm(ds):
        inputs, turn_idxs, cand_toks = get_prompt(sample, sp.tokenizer)
        sp.turn_idxs = turn_idxs
        with torch.no_grad():
            out = sp.model(
                **inputs.to(sp.device), output_hidden_states=True, do_sample=False
            )

        z = torch.stack(list(sp.cache["z"].values())).detach().float()
        attn = torch.stack(list(sp.cache["attn"].values())).detach().float()
        top1_acc += z.argmax(-1) == sample["mistake_step"]
        rms = (
            out.hidden_states[-1][0, -1].float().pow(2).mean().detach().cpu() + sp.eps
        ).sqrt()
        logits = out.logits[0, -1].float().detach().cpu()
        cand_toks = torch.stack([t[logits[t].argmax()] for t in cand_toks])
        emit_tok = cand_toks[logits[cand_toks].argmax()]

        cur_dla = sp.readout_direction(emit_tok, cand_toks).cpu()
        dla += torch.matmul(attn, cur_dla) / rms

        del inputs, out, attn, z
        sp.reset()
        gc.collect()
        torch.cuda.empty_cache()

    dla_idxs = (dla >= dla[1:-1].max() * 0.9).nonzero().tolist()
    if len([(dla[l, h], l, h) for l, h in dla_idxs if l > 0]) < 2:
        dla_idxs = dla.nonzero().tolist()
    dla_rank = [(dla[l, h], l, h) for l, h in dla_idxs if l > 0]
    writers = [(l, h) for _, l, h in sorted(dla_rank, reverse=True)][:5]

    idxs = (top1_acc >= top1_acc[1:-1].max() * 0.9).nonzero().tolist()
    if len([(top1_acc[l, h], l, h) for l, h in idxs if l < sp.nL - 1]) < 2:
        idxs = top1_acc.nonzero().tolist()
    final_acc = [(top1_acc[l, h], l, h) for l, h in idxs if l < sp.nL - 1]
    locators = [(l, h) for _, l, h in sorted(final_acc, reverse=True)][:5]
    return locators, writers, dla


def run_graft(ds, sp):
    agent_acc, step_acc = 0, 0
    for sample in tqdm(ds):
        inputs, turn_idxs, cand_ids = get_prompt(sample, sp.tokenizer)
        cand_ids = torch.stack([cand[0] for cand in cand_ids])
        max_toks = len(
            sp.tokenizer.encode(
                str(len(sample["history"]) - 2), add_special_tokens=False
            )
        )

        pred_toks = []
        for _ in range(max_toks):
            input_ids = (
                torch.cat([inputs["input_ids"], torch.tensor([pred_toks])], dim=-1)
                .to(sp.device)
                .int()
            )
            sp.turn_idxs, sp.cand_toks = turn_idxs, cand_ids.to(sp.device)
            sp.end_idx = inputs["input_ids"].shape[-1] - 1
            with torch.no_grad():
                out = sp.model(input_ids, do_sample=False)
            pred_toks.append(out.logits[0, -1].argmax(-1).cpu().int())
            del input_ids, out
            sp.reset()
            gc.collect()
            torch.cuda.empty_cache()
        pred_step = sp.tokenizer.decode(pred_toks, skip_special_tokens=True)
        pred_step = re.findall(r"(\d+)", pred_step)
        if len(pred_step) > 0:
            pred_step = pred_step[0]
        else:
            pred_step = ""
        agent_hit, step_hit = eval_pred(pred_step, sample)
        agent_acc, step_acc = agent_acc + int(agent_hit), step_acc + int(step_hit)
    return agent_acc / len(ds), step_acc / len(ds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--model", "-m", default="qwen0.5", type=str)
    p.add_argument("--data", "-d", default="algo", type=str)
    p.add_argument("--betas", "-b", type=float, nargs="+", default=[0.5, 1, 2, 3])
    a = p.parse_args()
    device = f"cuda:{a.device}" if torch.cuda.is_available() else "cpu"
    os.makedirs("./results", exist_ok=True)

    summary = {}
    if os.path.exists("./results/summary.json"):
        summary = json.load(open("./results/summary.json"))
    set_seed(42)
    ds = json.load(open("./../data/processed_data.json"))[a.data]
    train_data, test_data = get_train_test_split(ds, a.data)
    sp = StepPatch(a.model, device)

    start = time.time()
    sp.register_cache_hooks()
    locator, writer, _ = select_heads(sp, train_data)
    sp.register_grafting_hooks(locator, writer)

    curve = []
    for beta in a.betas:
        sp.beta = beta
        agent_acc, step_acc = run_graft(test_data, sp)
        curve.append({"beta": beta, "agent_acc": agent_acc, "step_acc": step_acc})
    total_time = time.time() - start

    best_g = sorted(curve, key=lambda x: (-x["step_acc"], -x["agent_acc"], x["beta"]))
    best_g = best_g[0]

    res = {
        "agent_acc": best_g["agent_acc"],
        "step_acc": best_g["step_acc"],
        "graft_beta": best_g["beta"],
        "locator": locator,
        "writer": writer,
        "curve": curve,
        "total_time": total_time,
    }

    json.dump(
        res,
        open(f"./results/{a.model}_{a.data}.json", "w"),
        indent=2,
    )
    summary[f"{a.model}_{a.data}"] = {
        k: res[k]
        for k in [
            "agent_acc",
            "step_acc",
            "total_time",
        ]
    }
    json.dump(
        {k: summary[k] for k in sorted(summary)},
        open(f"./results/summary.json", "w"),
        indent=4,
    )
