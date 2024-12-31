#!/bin/bash
#SBATCH --job-name=do_generate
#SBATCH --output=%x-%j.out         # stdout
#SBATCH --error=%x-%j.err          # stderr
#SBATCH --time=12:00:00            # walltime (HH:MM:SS)
#SBATCH --nodes=1                  # single node
#SBATCH --ntasks=1                 # single task/process
#SBATCH --cpus-per-task=16         # CPU cores per task
#SBATCH --gres=gpu:4               # 4 GPUs
#SBATCH --mem=400G                 # memory
#SBATCH -p PB                      # partition/queue name

# (Optional) Load modules, e.g., if your cluster uses them
# module load cuda/11.7
# module load anaconda/2023a

# Source your .bashrc which has the Conda init lines
# (Make sure .bashrc has the 'conda init' lines)
source ~/.bashrc

# Now activate the desired Conda environment
conda activate rstar

# Optional environment variables
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change directory to where your code is located so local imports work
cd /home/n3/jongyeon/rStar

echo "Starting do_generate.py ..."

python run_src/do_generate.py \
    --model_ckpt meta-llama/Llama-3.1-8B \
    --dataset_name GSM8K \
    --note tensor_parallelism \
    --num_rollouts 16 \
    --api vllm \
    --model_parallel \
    --tensor_parallel_size 4

echo "Finished do_generate.py."