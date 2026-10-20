# A Mechanistic Study of Automated Failure Attribution in LLM-based Agentic Systems

> **TLDR:** when a team of AI agents fails at a task, we can figure out *which* agent's step broke things — not by asking the model (it usually guesses wrong), but by reading the "thoughts" inside the model that already know the answer.


## 1. Motivation

Modern AI systems often work as a *team of agents*: several LLM (Large Language Model) "agents" talk to each other, each doing part of a job (planning, searching, coding, checking). When the team gets the wrong final answer, the natural question is: **which step first went wrong?** Answering that automatically is the **Automated Failure Attribution (AFA)** problem, and it's the bottleneck for debugging and improving these agent pipelines.

The obvious approach is to just *ask a model*: give it the whole transcript and have it name the failing step. **That barely works.** Small and mid-size instruction-tuned models score near random chance, and even large frontier models are unreliable — they confidently point at the **wrong** step.

The key observation driving this project: **the models actually know the answer, they just won't say it.** The correct failing step can be recovered from the model's *internal activations* (the intermediate numeric "thoughts" a model computes while reading the transcript) even when the model's written answer is wrong. We call this the **"knows-but-won't-say" gap**. So AFA isn't only a knowledge problem — it's a *readout / behavior* problem, and that is exactly what **mechanistic interpretability** (the study of the internal computations of neural networks) is built to attack.


## 2. Project overview

The project is organized as a progression that opens the model up one layer at a time. A **black box** means we only see inputs and outputs; a **white box** means we can see and change the internal machinery. Each method is a stronger form of access than the last:

```
        BLACK BOX ────────────────────────────► WHITE BOX

          prompting ────────► probing ────────► grafting 
         (behavioral,        (read the         (rewire the      
         ask & score         internals,        circuit, causal,
         the output)         non-causal)       label-free)          
              │                   │                  │               
         "won't say"         "does know"       "can be made     
                            (measurement)       to say it"       
```


### Method 1 — Prompting (black box)

We only touch inputs and outputs, treating the model as a black box. The transcript is fed in and a regex (pattern match) pulls the predicted agent + step back out of the generated text. Three prompting strategies are compared:

- **`vanilla`** — ask for the failing agent and step with no step indices shown.
- **`with_step`** — each line is prefixed `i - name:` so the model has explicit numbers to name.
- **`sequential`** — a two-stage prompt: first ask *which agent* failed, then ask *which step*.
- **`step_by_step`** — provide the history incrementally, asking the model if a failure occurred in the current step


This is the baseline that establishes the problem: it shows how far plain prompting gets, and (through a prompt-format ablation) whether the wording is what's holding the model back. **It isn't** — step accuracy stays low even across strategies, which is the "won't-say" failure that motivates looking inside the model.

### Method 2 — Probing (reading the internals, non-causal)

A **probe** is a tiny classifier trained on the model's internal activations to test what information is present inside — without changing anything the model does. The pipeline:

1. Run each transcript through the model and, at every turn position, cache two internal vectors per layer: the attention-block output and the hidden state (the "residual stream" — the model's running internal summary at that point).
2. Train a standardized **logistic probe** (a simple linear yes/no classifier) to predict "is this turn the mistake?".
3. Use **conversation-grouped 5-fold cross-validation** — turns from the same conversation never appear in both the training and test splits, so the probe can't cheat by memorizing a conversation.
4. Take the highest-scoring turn as the predicted failing step, and report the best-performing layer.

This is a *measurement*, not a fix: it proves the failing step **is** linearly decodable from the activations (the "knows" half of the gap) while the model's actual output stays unchanged.

### Method 3 — Grafting (causal, label-free rewire)

Inside the model, attention is split across many **attention heads** (parallel sub-components, each attending to different parts of the text). Grafting identifies two kinds and wires one into the other, **without ever using the ground-truth label**:

- **Locator heads** — heads that attend to the failing turn (they've figured out *where* the mistake is).
- **Writer heads** — heads whose output most directly pushes the final emitted digit (they *write* the answer).

Natively the writer largely ignores the locator and emits a position-default answer (hence the wrong output). The graft fixes this at inference time:

1. **Find the heads automatically.** On a small held-out split, locators are picked by which heads' turn-attention most often points at the true failing turn, and writers by direct logit attribution (how much each head's output moves the answer digit).
2. **Read the locator.** Take the turn the locator heads collectively point to as the model's own internal guess.
3. **Amplify into the writer.** Add the direction for that guessed digit into the writer heads' value stream, scaled by a strength factor `beta`.
4. **Re-read the emitted digit.** The model now generates the corrected step number.

This is *causal*: it changes the token the model actually generates, using only the model's own computation and no labels — the first point in the arc where the model can be made to *say* what it already knows.

## 3. Experiment results

### Dataset & models

- **Data:** [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When)
- **Open-source models:** Qwen2.5-{0.5B, 1.5B, 7B}-Instruct, Llama-3.2-{1B, 3B}-Instruct, Gemma-3-{1b, 4b}-it.
- **Large frontier models** (prompting only, for reference): GPT-5.6 Sol, GPT-5.6 Terra, Claude Opus 4.8 and 5, and Claude Sonnet 5.

### Evaluation Metrics
Every method is evaluated on both datasets. Two things are measured throughout:

- **Agent accuracy** — did the method correctly identify the *agent* that made the failure?
- **Step accuracy** — did it correctly identify the exact *step* (turn) where the failure happened? This is the harder, more useful metric.

## 4. How to run
First, create the environment and activate it with the following commands:
```bash
conda env create -f environment.yml && conda activate mi4afa
```

To run the prompting experiments, run the following command in the project root directory:

```bash
cd prompting && python run_all_prompts.py
```

To run the prompting experiments on the large models, obtain a Cursor API key and run the following command in the project root directory:

```bash
cd prompting_big && CURSOR_API_KEY=[API_KEY] python run_all_prompts.py
```

To run the probing experiments, run the following command in the project root directory:

```bash
cd probing && python run_all_probes.py
```

To run the grafting experiments, run the following command in the project root directory:

```bash
cd grafting && python run_all_graft.py
```