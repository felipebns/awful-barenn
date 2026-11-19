#!/bin/bash -l
#SBATCH --job-name=barenn-gpu
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=barenn-gpu-%j.out
# uso: make barenn_gpu && scripts/teste_grande.sh 50 && sbatch slurm/gpu.slurm
#      (REPETICOES=3 sbatch slurm/gpu.slurm para a v1.0, que é lenta)

module load cuda/12.8.1 2>/dev/null || true
echo "Job $SLURM_JOB_ID | nó $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
lscpu | grep -E 'Model name'
nvcc --version | tail -1

n=${REPETICOES:-5}
scripts/validar.sh ./barenn_gpu
scripts/bench.sh "$n" ./barenn_gpu
echo "--- teste 50x (optdigits_x50.tes)"
scripts/bench.sh "$n" ./barenn_gpu optdigits.tra optdigits_x50.tes
