set -x
source .env

uv run ray stop
uv run ray start --head --port 6380 --num-gpus 8 --num-cpus 32

uv run ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"], "env_vars": {"CUDA_LAUNCH_BLOCKING": "1"}}' \
  --working-dir . \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
  --save_path checkpoint/dummy_rl \
  --micro_train_batch_size 1 \
  --train_batch_size 48 \
  --micro_rollout_batch_size 12 \
  --rollout_batch_size 48 \
  --n_samples_per_prompt 3 \
  --max_samples 10000 \
  --max_epochs 3 \
  --prompt_max_len 4096 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --critic_learning_rate 1e-5 \
  --init_kl_coef 0.01 \
  --prompt_data /home/ubuntu/vlad/bash-bench/valid_tasks-repeated.json \
  --input_key problem_statement \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_API_KEY \
  --wandb_project bash_bench_rl \
  --advantage_estimator grpo \
  --env_file bash_bench_env \
  --env_class BashBenchEnv \

# the following relationship should be verified:
# micro_train_batch_size * gradient_accumulation_steps * actor num_nodes == train_batch_size
# rollout_batch_size % vllm_num_engines == 0
