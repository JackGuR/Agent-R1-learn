# Quick Start

This quick start is a **sanity check**, not the main Agent-R1 workflow. Its purpose is to verify that your environment, dataset path, model path, and training stack are wired correctly.

## 1. Prepare a Minimal Dataset

Use the GSM8K preprocessing script:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

This produces:

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

## 2. Run the Sanity Check Script

Use the provided single-step script:

```bash
bash examples/run_qwen2.5-3b.sh
```

If needed, adjust the following values before running:

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- dataset paths under `~/data/gsm8k`

The example defaults to one GPU. To wait for an idle GPU and start automatically:

```bash
bash examples/run_when_gpu_available.sh
```

The watcher prints GPU status while waiting, streams training output to the
terminal, and saves the same output under `logs/`.

For a slower single-GPU profile targeting roughly 30-40 GiB of GPU memory,
use the CPU-offload configuration:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen2.5-3b_low_memory.sh
```

This profile reduces rollout concurrency and response length, disables CUDA
graphs for rollout, and offloads actor parameters, optimizer state, and
activations to CPU memory.

For better throughput while keeping LoRA and targeting roughly 30-40 GiB on
an 80 GiB GPU, use the balanced profile:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen2.5-3b_balanced.sh
```

The balanced profile disables CPU offload, increases rollout concurrency, and
uses CUDA graphs. It defaults to three epochs, saves every 100 steps, and uses
a fixed 2 GiB vLLM KV cache so startup does not depend on fluctuating free GPU
memory. Override it with `VLLM_KV_CACHE_MEMORY_BYTES` when needed.

For a substantive run intended to finish in roughly one day on one A800, train
the 1.5B model on the complete GSM8K dataset for three epochs:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen2.5-1.5b_day.sh
```

This profile uses full-parameter GRPO rather than LoRA, targets roughly
30-40 GiB, and saves and validates every 100 steps.

The script entrypoint is [`examples/run_qwen2.5-3b.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/run_qwen2.5-3b.sh), which launches `python3 -m agent_r1.trainer.main_agent_ppo`.

## 3. What to Do Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to understand the main training abstraction.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see how `AgentFlowBase`, `AgentEnvLoop`, and `ToolEnv` fit together.
- Continue to the [`Agent Task Tutorial`](../tutorials/agent-task.md) for the main Agent-R1 workflow based on multi-step interaction.
