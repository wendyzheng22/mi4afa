import sys

sys.path.append("./../")
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import MODELS

data = ["algo", "craft", "smolagents", "magentic-one", "alfagent", "mathchat"]
prompt = ["step_by_step", "vanilla", "with_step", "sequential"]
max_gpus = 1


def run_job(model, data_name, prompt_type, device):
    cmd = [
        "python",
        "run_prompt.py",
        "-m",
        model,
        "-d",
        data_name,
        "-p",
        prompt_type,
        "--device",
        str(device),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd = os.path.dirname(os.path.abspath(__file__)), check=True)

if __name__ == "__main__":
    idx = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for prompt_type in prompt:
            for data_name in data:
                for model in MODELS:
                    if os.path.exists(f"./results/{model}_{data_name}_{prompt_type}.json"):
                        continue
                    device = idx
                    futures.append(
                        executor.submit(
                            run_job, model, data_name, prompt_type, device
                        )
                    )
                    idx += 1
                    if idx % max_gpus == 0:
                        idx = 0
        for future in as_completed(futures):
            future.result()