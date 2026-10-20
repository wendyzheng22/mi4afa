import json
from datasets import load_dataset


DATA = {"Algorithm-Generated": "algo", "Hand-Crafted": "craft"}

processed_data = {}
for key in DATA:
    ds = load_dataset("Kevin355/Who_and_When", key)["train"]
    cur_data = []
    for sample in ds:
        if int(sample["mistake_step"].strip()) > len(sample['history']):
            continue
        cur_sample = {"mistake_agent": sample["mistake_agent"].strip()}
        cur_sample["mistake_step"] = int(sample["mistake_step"].strip())
        cur_sample["mistake_reason"] = sample['mistake_reason']
        cur_sample['question'] = sample['question']
        cur_sample['need_inspect'] = False

        if key == "Algorithm-Generated":
            cur_sample['ground_truth'] = sample['ground_truth']
            cur_sample['history'] = sample['history']
            if sample['history'][cur_sample["mistake_step"]]['name'] != cur_sample['mistake_agent']:
                cur_sample['need_inspect'] = True
        elif key == "Hand-Crafted":
            cur_sample['ground_truth'] = sample['groundtruth']
            history, mistake_step = [], -1
            for h in sample["history"][1:]:
                if "Error" not in h['content']:
                    if "(->" in h["role"]:
                        continue
                    if h["role"] == 'Orchestrator (thought)' and "Next speaker" in h['content']:
                        continue
                if "Updated Ledger:\n{\n" in h["content"]:
                    start_idx = h["content"].index("instruction_or_question") - 1
                    end_idx = h['content'][start_idx:].index("}") + start_idx + 1
                    h["content"] = json.loads("{" + h['content'][start_idx:end_idx] + "}")['instruction_or_question']['answer']
                h["name"] = h["role"]
                history.append(h)
                if h["content"] == sample["history"][cur_sample["mistake_step"]]['content'] and h['role'] == sample["history"][cur_sample["mistake_step"]]['role']:
                    mistake_step = len(history) - 1
            cur_sample["history"] = history
            cur_sample['mistake_step'] = mistake_step
            if mistake_step == -1:
                cur_sample['need_inspect'] = True
        cur_data.append(cur_sample)
    processed_data[DATA[key]] = cur_data

json.dump(processed_data, open("processed_data.json", "w"), indent=4)
