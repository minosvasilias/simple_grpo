# Simple GRPO

This repository provides a minimal setup for training models using GRPO that can be easily reproduced on fresh GPU nodes using a one-liner setup script.

The default configuration ensures that models in the 8b parameter range (tested using [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)) can be trained on an affordable GPU environment such as a 4xH100 node.

This utilizes a manual `trl` installation based on [these](https://github.com/huggingface/trl/pull/2669) improvements made by [andyl98](https://github.com/andyl98). (This should be updated once the PR is merged into the main repo.)

Training script adapted from Llama-1b GRPO gist made by [willccbb](https://github.com/willccbb).

## Installation

To automatically clone the repo, install dependencies and setup the environment, run the following command:

```bash
source <(curl -sL https://raw.githubusercontent.com/minosvasilias/simple_grpo/main/setup.sh)
```

For manual steps, please refer to the [setup.sh](setup.sh) script for more details.

## Training

To train a model, run the following command:

```bash
bash train_llama_8b.sh
```

Hyperparameters for the training run are defined within the [train_llama_8b.sh](train_llama_8b.sh) script, and the default deepspeed configuration (located at [configs/zero3.yaml](configs/zero3.yaml)) is set up for 4 GPUs with 94GB of memory each.

To support different GPU setups, modify `num_processes` in [configs/zero3.yaml](configs/zero3.yaml) to the GPU count you are using.

In `train.sh`, keep in mind that `num_processes` should be set to the number of GPUs you are using minus 1. One GPU is reserved for inference using vLLM.

## Memory Usage / Hardware Constraints

Several tests have been run on 4x80GB (A100, H100) setups, and training using the default configuration in this repo is able to run for several steps (longest successful run was 33 steps), however eventually ends in an OOM failure. A seemingly stable run over >100 steps was achieved using `max_prompt_length` and `max_completion_length` values of `512` each.

For more workable context lengths and memory buffers, the minimum recommendation for this code is either a 4x94GB (H100) or 8x80GB (A100, H100) node.

All experiments have been run using nodes from [vast.ai](https://vast.ai/), using the default [NVIDIA CUDA image](https://cloud.vast.ai?ref_id=62897&template_id=091a17ac1994276b9798c045b5a88358).