#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
BENCH_BIN="./numa_bench"
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
DURATION=5.0
PERF_ENABLED="${PERF_ENABLED:-0}" # Запускать с PERF_ENABLED=1 ./run_experiments.sh
SIZES_MB=(32 128 512)             # L3 fit, DRAM, DRAM overflow
THREADS_LIST=(1 4)
WORKLOADS_LIST=(ptr_chase sequential random false_share)

# === UTILS ===
log() { printf "[%-8s] %s\n" "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

# === SYSTEM TUNING ===
apply_tuning() {
    [[ $EUID -ne 0 ]] && die "Run as root or with sudo (required for sysctl/drop_caches/cpufreq)"

    log "Disabling NUMA balancing..."
    sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || die "sysctl failed"

    log "Setting CPU governor to performance..."
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$f" ] && echo performance > "$f"
    done

    log "Dropping page cache..."
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

    log "System prepped: NUMA balancing OFF, governor=performance, caches dropped."
}

# === BUILD ===
compile() {
    log "Compiling benchmark..."
    g++ -O3 -march=native -std=c++17 -lnuma -lpthread \
        -o "$BENCH_BIN" main.cpp stats.cpp numa_utils.cpp workload.cpp || die "g++ failed"
    chmod +x "$BENCH_BIN"
}

# === RUN SINGLE ===
run_case() {
    local cpu=$1 mem=$2 mode=$3 wl=$4 threads=$5 size=$6 extra_flags=${7:-}
    local raw="${RESULTS_DIR}/${mode}_${wl}_t${threads}_${size}MB_raw.csv"

    [[ ! -x "$BENCH_BIN" ]] && die "Benchmark binary missing. Run compile first."

    local cmd=(
        "$BENCH_BIN"
        --cpu_node "$cpu" --mem_node "$mem"
        --workload "$wl" --threads "$threads"
        --size_mb "$size" --duration "$DURATION"
        --mode "$mode" --raw_output "$raw"
        $extra_flags
    )

    log "▶ ${cmd[*]}"

    if [[ "$PERF_ENABLED" -eq 1 ]]; then
        command -v perf &>/dev/null || die "perf not installed (apt install linux-tools-generic)"
        local perf_out="${RESULTS_DIR}/${mode}_${wl}_t${threads}_${size}MB_perf.txt"
        perf stat -e cache-misses,LLC-loads,LLC-load-misses,cycles,instructions \
                  -x ',' -o "$perf_out" -- "${cmd[@]}" >> "${RESULTS_DIR}/perf_combined.csv"
    else
        "${cmd[@]}" >> "${RESULTS_DIR}/combined.csv"
    fi
}

# === MAIN ===
main() {
    mkdir -p "$RESULTS_DIR"
    echo "mode,threads,mem_node,cpu_node,workload,size_mb,p50,p90,p99,p99_9,p999" > "${RESULTS_DIR}/combined.csv"

    if [[ "$PERF_ENABLED" -eq 1 ]]; then
        echo "mode,threads,mem_node,cpu_node,workload,size_mb,event,count" > "${RESULTS_DIR}/perf_combined.csv"
    fi

    apply_tuning
    compile

    log "Starting experiment matrix..."

    for spec in "0:0:local" "0:1:remote" "0:0:interleaved"; do
        IFS=':' read -r cpu mem mode <<< "$spec"
        local flags=""
        [[ "$mode" == "interleaved" ]] && flags="--interleaved"

        for wl in "${WORKLOADS_LIST[@]}"; do
            for t in "${THREADS_LIST[@]}"; do
                # false_share в коде аллоцирует только 64B*threads, размер из CLI игнорируется
                local run_size=1
                [[ "$wl" != "false_share" ]] && run_size="${SIZES_MB[0]}"

                for sz in "${SIZES_MB[@]}"; do
                    [[ "$wl" == "false_share" ]] && break # false_share прогоняем один раз на размер 1MB
                    run_case "$cpu" "$mem" "$mode" "$wl" "$t" "$sz" "$flags"
                done
            done
        done
    done

    log "Done. Results: ${RESULTS_DIR}/combined.csv"
    log "Raw dumps : ${RESULTS_DIR}/*_raw.csv"
    [[ "$PERF_ENABLED" -eq 1 ]] && log "Perf stats: ${RESULTS_DIR}/perf_combined.csv"
}

main "$@"