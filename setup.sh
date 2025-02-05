#!/bin/bash
set -e

echo "=== Cloning simple_grpo repository ==="
# Clone the repo into a temporary directory
git clone https://github.com/minosvasilias/simple_grpo.git
cd simple_grpo

echo "=== Setting up the Python environment ==="
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
echo "=== Installing requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

# Install custom TRL branch for memory optimization
echo "=== Installing custom TRL branch ==="
git clone --branch grpo-vram-optimization https://github.com/andyl98/trl.git trl_custom
cd trl_custom
# Tested, stable commit
git checkout ccc95472f6245f2db00986a08ca16da68bf32c14
pip install .
cd ..

# Ask user for Huggingface API key to set HF_TOKEN env var
echo "=== Setting up Huggingface API key ==="
read -rp "Please enter your Huggingface API key to access gated models such as Llama-3.1-8B-Instruct: " HF_TOKEN
export HF_TOKEN
echo "export HF_TOKEN=$HF_TOKEN" >> venv/bin/activate
echo "HF_TOKEN has been set to: $HF_TOKEN"

# Also ask for Wandb API key to log training runs
echo "=== Setting up Wandb API key ==="
read -rp "Please enter your Wandb API key to log training runs: " WANDB_API_KEY
export WANDB_API_KEY
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> venv/bin/activate
echo "Wandb API key has been set to: $WANDB_API_KEY"

echo "=== Setup complete! ==="
