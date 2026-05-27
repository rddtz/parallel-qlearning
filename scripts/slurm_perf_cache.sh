#!/bin/bash
# =============================================================================
# slurm_perf_cache.sh -- SLURM wrapper for perf_cache_10x10.sh
#
# Submit from repo root:
#   sbatch scripts/slurm_perf_cache.sh
#
# Results land in:
#   <submission_dir>/results/perf_cache_10x10_<EXP_ID>/
# =============================================================================
#SBATCH --job-name=perf_cache_10x10
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

SUBMIT_DIR="$(pwd)"
EXP_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Scratch workspace
# ---------------------------------------------------------------------------
WORK_DIR="${SCRATCH:-/tmp}/perf_cache_${EXP_ID}"
mkdir -p "$WORK_DIR"

echo "EXP_ID    = $EXP_ID"
echo "WORK_DIR  = $WORK_DIR"

cp -r "$SUBMIT_DIR/src"     "$WORK_DIR/"
cp -r "$SUBMIT_DIR/scripts" "$WORK_DIR/"
cp    "$SUBMIT_DIR/Makefile" "$WORK_DIR/"

# ---------------------------------------------------------------------------
# Copy results back on exit (even on error or cancellation)
# ---------------------------------------------------------------------------
copy_back() {
    echo "Copying results back to $SUBMIT_DIR/results/ ..."
    mkdir -p "$SUBMIT_DIR/results"
    if ls "$WORK_DIR/results"/perf_cache_* &>/dev/null; then
        cp -r "$WORK_DIR/results"/perf_cache_* "$SUBMIT_DIR/results/"
        echo "Done."
    else
        echo "WARNING: no results found in $WORK_DIR/results" >&2
    fi
}
trap copy_back EXIT

# ---------------------------------------------------------------------------
# Compile on the compute node
# ---------------------------------------------------------------------------
cd "$WORK_DIR"
echo "Compiling ..."
export REPO_ROOT="$WORK_DIR"
make clean && make
echo "Compilation done."

# ---------------------------------------------------------------------------
# Check perf permissions
# perf_event_paranoid > 1 blocks hardware counters for non-root users.
# Print a warning but continue -- perf will output <not supported> per event.
# ---------------------------------------------------------------------------
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
echo "perf_event_paranoid = $paranoid"
if [ "$paranoid" != "unknown" ] && [ "$paranoid" -gt 1 ]; then
    echo "WARNING: perf_event_paranoid=$paranoid -- hardware counters may be blocked." >&2
    echo "         Ask sysadmin: sysctl -w kernel.perf_event_paranoid=1" >&2
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Starting perf cache probe ..."
REPO_ROOT="$WORK_DIR" bash "$WORK_DIR/scripts/perf_cache_10x10.sh"

# trap EXIT fires here
