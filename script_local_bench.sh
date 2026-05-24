#!/bin/bash

# ==============================================================================
# SCRIPT DE BENCHMARK LOCAL COM VTUNE (PERF SNAPSHOT) E LOG GERAL - 100x100
# ==============================================================================

# Carregamento de variáveis do oneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
elif [ -f "/home/intel/oneapi/setvars.sh" ]; then
    source /home/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

# Cores para saída
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Preparando Ambiente ===${NC}"
make clean && make

BASE_DIR="results_local_vtune_100x100"
LOG_GERAL="benchmark_log_geral.csv"
mkdir -p "$BASE_DIR"

# Inicializa o log geral com cabeçalho
echo "Tipo,Grid,Episodios,Passos,Obs_Pct,Schedule,Sync_Mode,Threads,Duracao_Vtune,Convergiu" > "$LOG_GERAL"

# ------------------------------------------------------------------------------
# PARÂMETROS DE BENCHMARK
# ------------------------------------------------------------------------------
GRID=100
EPS=100000
STEPS=1000
EPSILON=0.3
SEED=42
PCT=25
num_obs=$(( (GRID * GRID * PCT) / 100 ))

echo -e "${BLUE}Iniciando Benchmark (Grid ${GRID}x${GRID}, ${EPS} Eps, ${PCT}% Obstáculos)${NC}"

# ==============================================================================
# 1. TESTE SEQUENCIAL
# ==============================================================================
echo -e "\n${GREEN}>>> EXECUTANDO VERSÃO SEQUENCIAL <<<${NC}"

RUN_ID="seq_obs${PCT}pct"
TEMP_LOG="$BASE_DIR/temp_exec.log"

echo -e "\n--- Sequencial | Obs: ${PCT}% ($num_obs) ---"

vtune -collect performance-snapshot \
    -result-dir "$BASE_DIR/run_$RUN_ID" \
    -- ./bin/qlearning \
    --gridx "$GRID" \
    --gridy "$GRID" \
    --obstacles "$num_obs" \
    --episodes "$EPS" \
    --maxsteps "$STEPS" \
    --epsilon "$EPSILON" \
    --seed "$SEED" 2>&1 | tee "$TEMP_LOG"

# Extrai o tempo do VTune: "Elapsed Time: 7.456s" -> "7.456"
vtune_time=$(grep "Elapsed Time" "$TEMP_LOG" | sed 's/.*Elapsed Time: //' | sed 's/s//')

if grep -q "OBJETIVO ALCANCADO!" "$TEMP_LOG"; then
    convergiu="Sim"
else
    convergiu="Nao"
fi

echo "Sequencial,$GRID,$EPS,$STEPS,$PCT,N/A,N/A,1,$vtune_time,$convergiu" >> "$LOG_GERAL"

# ==============================================================================
# 2. TESTE PARALELO
# ==============================================================================
echo -e "\n${GREEN}>>> EXECUTANDO VERSÃO PARALELA <<<${NC}"

SCHEDULES=("static" "dynamic" "guided")
SYNC_MODES=("sqrt" "statespace" "hogwild")
THREADS=(2 4 8 16 20)

for sched in "${SCHEDULES[@]}"; do
    export OMP_SCHEDULE="$sched"
    
    for mode in "${SYNC_MODES[@]}"; do
        for threads in "${THREADS[@]}"; do
            export OMP_NUM_THREADS=$threads
            
            RUN_ID="par_s${sched}_m${mode}_t${threads}_obs${PCT}pct"
            echo -e "\n--- Paralelo | Sch: $sched | Mode: $mode | Threads: $threads | Obs: ${PCT}% ---"
            
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
                --seed "$SEED" 2>&1 | tee "$TEMP_LOG"
            
            # Extrai o tempo do VTune
            vtune_time=$(grep "Elapsed Time" "$TEMP_LOG" | sed 's/.*Elapsed Time: //' | sed 's/s//')
            
            if grep -q "OBJETIVO ALCANCADO!" "$TEMP_LOG"; then
                convergiu="Sim"
            else
                convergiu="Nao"
            fi
            
            echo "Paralelo,$GRID,$EPS,$STEPS,$PCT,$sched,$mode,$threads,$vtune_time,$convergiu" >> "$LOG_GERAL"
        done
    done
done

rm -f "$TEMP_LOG"

echo -e "\n${BLUE}=== Benchmark e Coleta VTune Concluídos ===${NC}"
echo -e "Log geral salvo em: $LOG_GERAL"
