import sys

sys.path.append("./../")
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

data = ["algo", "craft"]
prompt = ["vanilla", "with_step", "sequential"]
models = [
    "gpt-5.6-sol",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-5",
    "gpt-5.6-terra",
]


def run_job(model, data_name, prompt_type):
    cmd = [
        "python",
        "run_prompt.py",
        "-m",
        model,
        "-d",
        data_name,
        "-p",
        prompt_type,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), check=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for prompt_type in prompt:
            for data_name in data:
                for model in models:
                    if os.path.exists(
                        f"./results/{model}_{data_name}_{prompt_type}.json"
                    ):
                        continue
                    futures.append(
                        executor.submit(run_job, model, data_name, prompt_type)
                    )
        for future in as_completed(futures):
            future.result()
