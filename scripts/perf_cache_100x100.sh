#!/bin/bash
# perf_cache_100x100.sh -- cache miss probe for 100x100 grid across thread counts
#
# Runs perf stat with L1/L2/L3 cache events for each (threads x sync_mode) combo.
# One run each, no replications -- just enough to see the coherency pattern.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY_SEQ="$REPO_ROOT/bin/qlearning"
BINARY_PAR="$REPO_ROOT/bin/qlearning_parallel"

# Fixed parameters
GRID=100
EPISODES=10000
OBSTACLES=0
MAX_STEPS=1000
EPSILON=0.3
SEED=54

THREADS_LIST=(1 2 4 8 16 20 40)
SCHEDULES=(static dynamic guided)

SYNC_CONFIGS=(
    "sqrt:sqrt:0"
    "every10:sqrt:10"
    "every1000:sqrt:1000"
    "statespace:statespace:0"
    "hogwild:hogwild:0"
)

PERF_EVENTS="L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-references,cache-misses"

OUT_DIR="$REPO_ROOT/results/perf_cache_100x100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"

printf "%-8s %-12s %-8s %14s %14s %14s %14s %10s\n" \
    "threads" "config" "schedule" "L1-loads" "L1-misses" "LLC-loads" "LLC-misses" "time_s" \
    | tee "$SUMMARY"
printf '%s\n' "$(printf '%0.s-' {1..106})" | tee -a "$SUMMARY"

for threads in "${THREADS_LIST[@]}"; do
    for cfg_entry in "${SYNC_CONFIGS[@]}"; do
        IFS=':' read -r label sync_mode sync_interval <<< "$cfg_entry"
        for sched in "${SCHEDULES[@]}"; do

        # threads=1: sequential binary, one run only
        if [ "$threads" -eq 1 ] && { [ "$label" != "sqrt" ] || [ "$sched" != "static" ]; }; then
            continue
        fi

        if [ "$threads" -eq 1 ]; then
            BINARY="$BINARY_SEQ"
            SYNC_ARGS=()
        else
            BINARY="$BINARY_PAR"
            SYNC_ARGS=(--sync-mode "$sync_mode")
            [ "$sync_interval" -gt 0 ] && SYNC_ARGS+=(--sync-interval "$sync_interval")
        fi

        export OMP_NUM_THREADS="$threads"
        export OMP_SCHEDULE="$sched"
        if [ "$threads" -le 20 ]; then
            export OMP_PLACES=cores
        else
            export OMP_PLACES=threads
        fi
        export OMP_PROC_BIND=close

        LOG="$OUT_DIR/t${threads}_${label}_${sched}.log"

        perf stat -e "$PERF_EVENTS" \
            -- "$BINARY" \
                --gridx "$GRID" --gridy "$GRID" \
                --obstacles "$OBSTACLES" \
                --episodes "$EPISODES" \
                --maxsteps "$MAX_STEPS" \
                --epsilon "$EPSILON" \
                --seed "$SEED" \
                "${SYNC_ARGS[@]}" \
                --quiet \
            > "$LOG" 2>&1

        l1_loads=$(grep "L1-dcache-loads"     "$LOG" | awk '{gsub(/,/,"",$1); print $1}' | head -1)
        l1_miss=$(grep  "L1-dcache-load-miss" "$LOG" | awk '{gsub(/,/,"",$1); print $1}' | head -1)
        llc_loads=$(grep "LLC-loads"          "$LOG" | awk '{gsub(/,/,"",$1); print $1}' | head -1)
        llc_miss=$(grep  "LLC-load-misses"    "$LOG" | awk '{gsub(/,/,"",$1); print $1}' | head -1)
        time_s=$(grep    "TRAINING_TIME_S:"   "$LOG" | awk '{print $2}'                  | head -1)

        l1_loads="${l1_loads:-NA}"
        l1_miss="${l1_miss:-NA}"
        llc_loads="${llc_loads:-NA}"
        llc_miss="${llc_miss:-NA}"
        time_s="${time_s:-NA}"

        disp_label="$label"
        disp_sched="$sched"
        [ "$threads" -eq 1 ] && disp_label="(seq)" && disp_sched="-"

        printf "%-8s %-12s %-8s %14s %14s %14s %14s %10s\n" \
            "$threads" "$disp_label" "$disp_sched" \
            "$l1_loads" "$l1_miss" "$llc_loads" "$llc_miss" "$time_s" \
            | tee -a "$SUMMARY"
        done
    done
done

echo ""
echo "Raw logs : $OUT_DIR"
echo "Summary  : $SUMMARY"
