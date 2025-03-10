echo -e "\033[31m$(printf '=%.0s' {1..1000})\033[0m"

set -x
source .env

# , "env_vars": {"CUDA_LAUNCH_BLOCKING": "1"}

/root/.local/bin/uv run ray job submit --address=$RAY_ADDRESS \
  --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' \
  --working-dir . \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 2 \
  --ref_num_gpus_per_node 4 \
  --actor_num_nodes 2 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 4 \
  --vllm_sync_backend nccl \
  --colocate_actor_ref \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --save_path checkpoint/dummy_rl \
  --micro_train_batch_size 8 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 64 \
  --rollout_batch_size 64 \
  --n_samples_per_prompt 3 \
  --max_samples 10000 \
  --max_epochs 3 \
  --prompt_max_len 16384 \
  --generate_max_len 4096 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --critic_learning_rate 1e-5 \
  --init_kl_coef 0.01 \
  --prompt_data /app/swe_bench_rl/data/train.json \
  --input_key problem_statement \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_API_KEY \
  --wandb_project dummy_rl \
  --advantage_estimator grpo \
  --env_file dummy_env \
  --env_class DummyEnv \
  --eval_steps -1 \

#   --colocate_all_models \
#   --vllm_gpu_memory_utilization 0.6 \
#   --vllm_enable_sleep \

#   --use_kl_loss \
#   --use_kl_estimator_k3 \
#   --enforce_eager \
#   --colocate_all_models \
#   --vllm_gpu_memory_utilization 0.6 \
#   --vllm_enable_sleep \

# the following relationship should be verified:
# micro_train_batch_size * gradient_accumulation_steps * actor num_nodes == train_batch_size
# rollout_batch_size % vllm_num_engines == 0
