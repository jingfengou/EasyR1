#!/bin/bash

set -x

# Change to the script's parent directory (EasyR1) to ensure relative paths work correctly.
cd "$(dirname "$0")/.."

# This script runs a short test for the Visual-ARFT agent training setup.

# Set PYTHONPATH to ensure the 'verl' module can be found from the project root.
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Define the model path
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# Run the main trainer script using python3
# We override several parameters from the config.yaml for this specific test:
# - data.train_files: Use the small, converted dataset.
# - data.prompt_key:  Tell the loader to use the "prompt" key from our JSONL file.
# - data.format_prompt: Apply our custom Jinja2 template to prepend the system prompt.
# - worker.actor.model.model_path: Specify the model to use.
# - trainer.experiment_name: Give this test run a unique name.
# - trainer.max_steps: Limit the training to only 2 steps for a quick test.
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/oujingfeng/project/visual_RFT/Visual-RFT/EasyR1/examples/visual_arft_training_data_small.jsonl \
    data.val_files=/home/oujingfeng/project/visual_RFT/Visual-RFT/EasyR1/examples/visual_arft_validation_data_small.jsonl \
    data.rollout_batch_size=2 \
    worker.actor.global_batch_size=2 \
    data.prompt_key=prompt \
    data.format_prompt=examples/format_prompt/visual_arft_prompt.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=visual_arft_test_full_prompt \
    trainer.n_gpus_per_node=2 \
    trainer.max_steps=2 \
    worker.reward.reward_function=examples/reward_function/visual_arft_reward.py:compute_score