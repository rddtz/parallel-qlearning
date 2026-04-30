#!/bin/bash
#SBATCH --job-name=q_learning_parallel
#SBATCH --partition=hype[1]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

module load gcc

gcc -03 -g qlearning_cli.c -o qlearning_cli -fopenmp

# mode: normal, hard, extreme
for mode in normal hard extreme; do
    for threads in 1 2 4 8 16 32 40; do
        echo "=== mode: ${mode} ${threads}==="
        export OMP_NUM_THREADS=${threads}
        
        vtune -collect performance-snapshot \
            -result-dir "vtune_results_${mode}_${threads}" z
            -quiet \
            -- ./qlearning_cli --mode "${mode}" >/dev/null
        echo
    done
done



