#!/bin/bash
#SBATCH --job-name=q_learning_batch_bench
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ==============================================================================
# SCRIPT DE BENCHMARK BATCH (SLURM) - Q-LEARNING (100x100)
# ==============================================================================

# Carregamento de variáveis do oneAPI (ajuste o path se necessário no cluster)
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
elif [ -f "/home/intel/oneapi/setvars.sh" ]; then
    source /home/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

echo "=== Preparando Ambiente ==="
make clean && make > /dev/null 2>&1

BASE_DIR="results_batch_vtune_100x100"
LOG_GERAL="benchmark_log_geral.csv"
mkdir -p "$BASE_DIR"

# Inicializa o log geral com cabeçalho
echo "Tipo,Grid,Episodios,Passos,Obs_Pct,Schedule,Sync_Mode,Threads,Duracao_Vtune,Convergiu" > "$LOG_GERAL"

# Parâmetros Fixos
GRID=100
EPS=100000
STEPS=1000
EPSILON=0.3
SEED=42
PCT=25
num_obs=$(( (GRID * GRID * PCT) / 100 ))

echo "Iniciando Benchmark em Batch (Grid ${GRID}x${GRID}, ${EPS} Eps, ${PCT}% Obstáculos)"

# ==============================================================================
# 1. EXECUÇÃO SEQUENCIAL
# ==============================================================================
echo "----------------------------------------------------"
echo "Executando Versão Sequencial..."

RUN_ID="seq_obs${PCT}pct"
LOG_FILE="$BASE_DIR/log_$RUN_ID"

vtune -collect performance-snapshot \
    -result-dir "$BASE_DIR/run_$RUN_ID" \
    -- ./bin/qlearning \
    --gridx "$GRID" \
    --gridy "$GRID" \
    --obstacles "$num_obs" \
    --episodes "$EPS" \
    --maxsteps "$STEPS" \
    --epsilon "$EPSILON" \
    --seed "$SEED" > "$LOG_FILE" 2>&1

# Extrai tempo e convergência
vtune_time=$(grep "Elapsed Time" "$LOG_FILE" | sed 's/.*Elapsed Time: //' | sed 's/s//')
if grep -q "OBJETIVO ALCANCADO!" "$LOG_FILE"; then convergiu="Sim"; else convergiu="Nao"; fi

echo "Sequencial,$GRID,$EPS,$STEPS,$PCT,N/A,N/A,1,$vtune_time,$convergiu" >> "$LOG_GERAL"
echo "  [CONCLUÍDO] Tempo VTune: $vtune_time | Convergiu: $convergiu"

# ==============================================================================
# 2. EXECUÇÃO PARALELA
# ==============================================================================
echo "----------------------------------------------------"
echo "Iniciando Testes Paralelos..."

SCHEDULES=("static" "dynamic" "guided")
SYNC_MODES=("sqrt" "statespace" "hogwild")
THREADS=(2 4 8 16 20)

for sched in "${SCHEDULES[@]}"; do
    export OMP_SCHEDULE="$sched"
    echo ">>> Schedule: $sched"

    for mode in "${SYNC_MODES[@]}"; do
        for threads in "${THREADS[@]}"; do
            export OMP_NUM_THREADS=$threads
            
            RUN_ID="par_s${sched}_m${mode}_t${threads}_obs${PCT}pct"
            LOG_FILE="$BASE_DIR/log_$RUN_ID"
            
            echo "Executando: Sch: $sched | Mode: $mode | Threads: $threads"
            
            vtune -collect performance-snapshot \
                -result-dir "$BASE_DIR/run_$RUN_ID" \
                -- ./bin/qlearning_parallel \
                --gridx "$GRID" \
                --gridy "$GRID" \
                --obstacles "$num_obs" \
                --episodes "$EPS" \
                --maxsteps "$STEPS" \
                --epsilon "$EPSILON" \
                --sync-mode "$mode" \
                --seed "$SEED" > "$LOG_FILE" 2>&1
            
            # Extrai tempo e convergência
            vtune_time=$(grep "Elapsed Time" "$LOG_FILE" | sed 's/.*Elapsed Time: //' | sed 's/s//')
            if grep -q "OBJETIVO ALCANCADO!" "$LOG_FILE"; then convergiu="Sim"; else convergiu="Nao"; fi
            
            echo "Paralelo,$GRID,$EPS,$STEPS,$PCT,$sched,$mode,$threads,$vtune_time,$convergiu" >> "$LOG_GERAL"
            echo "  [CONCLUÍDO] Tempo VTune: $vtune_time | Convergiu: $convergiu"
        done
    done
done

echo "----------------------------------------------------"
echo "Benchmark concluído. Log geral: $LOG_GERAL"
