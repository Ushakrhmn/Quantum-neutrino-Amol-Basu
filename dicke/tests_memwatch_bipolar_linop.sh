#!/bin/bash
#SBATCH --account=def-nilic
#SBATCH --mem=128G               # memory per node
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=USR1@120        # (optional) pre-timeout signal to help debugging

set -Eeuo pipefail

mkdir -p logs

# -------- Configuration (tune here) --------
export PYTHONUNBUFFERED=1        # immediate Python stdout/stderr
export PYTHONFAULTHANDLER=1      # stack traces on fatal errors
: "${BIPOLAR_MEM_INTERVAL:=300}"      # seconds between memory samples (default 60s; set to 300 for 5 min)
: "${BIPOLAR_LOG:=logs/bipolar_linop_${SLURM_JOB_ID:-$$}.log}"
: "${BIPOLAR_MEMLOG:=logs/mem_bipolar_linop_${SLURM_JOB_ID:-$$}.log}"
# ------------------------------------------

# Activate your environment
source ../env.sh

log() { echo "$(date +'%F %T%z') $*" | tee -a "$BIPOLAR_LOG"; }

detect_cgroup_mem_files() {
  local cg cg2
  # cgroup v2
  if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
    [[ -n "$cg" ]] || cg="/"
    local base="/sys/fs/cgroup${cg}"
    if [[ -f "${base}/memory.current" ]]; then
      echo "${base}/memory.current|${base}/memory.max"
      return
    fi
  fi
  # cgroup v1 (memory)
  cg2=$(awk -F: '$2=="memory"{print $3}' /proc/self/cgroup)
  if [[ -n "$cg2" ]] && [[ -d "/sys/fs/cgroup/memory${cg2}" ]]; then
    echo "/sys/fs/cgroup/memory${cg2}/memory.usage_in_bytes|/sys/fs/cgroup/memory${cg2}/memory.limit_in_bytes"
    return
  fi
  echo "|"
}

mem_watch() {
  local out="$1"
  local interval="${BIPOLAR_MEM_INTERVAL}"
  IFS='|' read -r usage_file limit_file < <(detect_cgroup_mem_files)
  echo "# mem_watch interval=${interval}s usage_file=${usage_file} limit_file=${limit_file}" >> "$out"
  while :; do
    local ts usage limit u_mb l_mb pct
    ts="$(date +'%F %T%z')"
    if [[ -n "$usage_file" && -f "$usage_file" ]]; then
      usage="$(cat "$usage_file" 2>/dev/null || echo 0)"
    else
      usage=0
    fi
    if [[ -n "$limit_file" && -f "$limit_file" ]]; then
      limit="$(cat "$limit_file" 2>/dev/null || echo 0)"
    else
      limit=0
    fi
    if [[ "$limit" == "max" ]]; then
      printf '%s [mem] cgroup=%0.1fMB/ max\n' "$ts" "$(awk -v u="$usage" 'BEGIN{print u/1048576}')" >> "$out"
    else
      u_mb="$(awk -v u="$usage" 'BEGIN{printf "%.1f", u/1048576}')"
      l_mb="$(awk -v l="$limit" 'BEGIN{printf "%.1f", l/1048576}')"
      pct="$(awk -v u="$usage" -v l="$limit" 'BEGIN{if(l>0) printf "%.1f", (u/l)*100; else print "NA"}')"
      printf '%s [mem] cgroup=%sMB/%sMB (%s%%)\n' "$ts" "$u_mb" "$l_mb" "$pct" >> "$out"
    fi
    sleep "$interval" || break
  done
}

log "Starting job ${SLURM_JOB_ID:-N/A} on $(hostname)"
log "Sampling memory every ${BIPOLAR_MEM_INTERVAL}s → ${BIPOLAR_MEMLOG}"

mem_watch "$BIPOLAR_MEMLOG" &
MEM_WATCH_PID=$!
trap 'kill $MEM_WATCH_PID >/dev/null 2>&1 || true' EXIT

# Your command (fully unbuffered). Keep your original arguments.
# Using LinearOperator-based streaming (matrix-free, memory-efficient)
# Bipolar setup: bin 1 is pure nu_e, bin 2 is pure nubar_e
# CMD=(python -u bipolar_2bin_opt_instrumented_linop.py --e 1 --b 1 --j 1.0 --chunk 1)
# CMD=(python bipolar_2bin_opt_instrumented_linop.py --e 10 --b 10 --j 1.0 --chunk 1)
CMD=(python bipolar_2bin_opt_instrumented_linop.py --e 4000 --b 4000 --j 5.0 --chunk 1 --l 8)
log "Command: ${CMD[*]}"

# -u disables SLURM buffering; tee writes logs live to file
srun -u "${CMD[@]}" 2>&1 | stdbuf -oL -eL tee -a "$BIPOLAR_LOG"
STATUS=${PIPESTATUS[0]}

log "Program exit code: $STATUS"
exit "$STATUS"

