#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-2147483648}"

# A substantive single-GPU learning run: full-parameter GRPO on the complete
# GSM8K dataset for three epochs. The 1.5B model and fixed vLLM KV cache target
# roughly 30-40 GiB on an 80 GiB GPU while avoiding shared-GPU memory profiling.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

bash "$SCRIPT_DIR/run_qwen2.5-3b.sh" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_activation_offload=False \
    data.train_batch_size=32 \
    data.max_response_length=512 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="$KV_CACHE_MEMORY_BYTES" \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name=qwen2_5_1_5b_day \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    "$@"
