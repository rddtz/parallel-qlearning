#!/bin/bash
source /opt/intel/oneapi/setvars.sh
echo "Carregando o vtune"
echo "compilando"
make

for threads in 1 2 4; do
    echo "=== mode: ${threads}==="
    export OMP_NUM_THREADS=${threads}
    
    vtune -collect performance-snapshot \
        -result-dir "vtune_results_${mode}_${threads}" \
        -- ./bin/qlearning_parallel --gridx 50 --gridy 50 --obstacles 250 --maxsteps 2500 --episodes 40000 >> "log_${threads}"
    echo
done



