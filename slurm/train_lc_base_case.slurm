#!/bin/bash
#SBATCH --job-name=basecase
#SBATCH --output=/scratch/gpfs/jz6521/latent_circuit_theory/slurm_out/slurm-%A.%a.out # stdout file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00  
#SBATCH --array=41-100
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jz6521@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2023.3
conda activate lcm
python ~/Documents/latent_circuit_train/src/train_latent_net.py $SLURM_ARRAY_TASK_ID /scratch/gpfs/jz6521/latent_circuit_theory/base_case $SLURM_ARRAY_JOB_ID