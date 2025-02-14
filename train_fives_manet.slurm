#!/bin/bash
#SBATCH -J MANet        # Job name
#SBATCH -o MANet_output_%j.log   # Output file
#SBATCH -e MANet_error_%j.log    # Error file
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH -c 32                   # Request 32 CPU cores
#SBATCH --mem=32G               # Request 32GB of RAM
#SBATCH -p medium               # Partition (queue)
#SBATCH -t 3-00:00:00           # Time limit (3 days)

# --- Environment Setup ---

# 1. Load necessary modules (adjust for your cluster)
module load cesga/2020
module load python/3.9.9

# 2. Activate your virtual environment (adjust the path)
source ../vcompetencia/bin/activate

# 3. Change directory to your project directory (adjust the path)
cd /home/usc/ec/rsm/otrosfives

# --- Run the Training Script ---

srun python train_baseline.py --config config/fives_manet.yaml
