#!/bin/bash
#SBATCH --job-name=barenn-mpi
#SBATCH --partition=sunny_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=barenn-mpi-%j.out
# uso: make && scripts/teste_grande.sh 50 && sbatch slurm/mpi.slurm

echo "Job $SLURM_JOB_ID | nós $SLURM_JOB_NODELIST | $(date)"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L1d|L2|L3'
mpic++ --version | head -1

for p in 1 2 4 8; do
    echo "===== $p processo(s) MPI ====="
    scripts/validar.sh srun --mpi=pmix -n "$p" ./barenn
    scripts/bench.sh 5 srun --mpi=pmix -n "$p" ./barenn
    echo "--- teste 50x (optdigits_x50.tes)"
    scripts/bench.sh 5 srun --mpi=pmix -n "$p" ./barenn optdigits.tra optdigits_x50.tes
done
