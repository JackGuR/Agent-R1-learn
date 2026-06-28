#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep LoRA's small optimizer state, but trade the low-memory profile's CPU
# offload overhead for higher single-GPU throughput. This profile is intended
# to use roughly 30-40 GiB on an otherwise idle 80 GiB GPU.
bash "$SCRIPT_DIR/run_qwen2.5-3b_low_memory.sh" \
    data.train_batch_size=32 \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.28 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.experiment_name=qwen2_5_3b_balanced \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    "$@"
