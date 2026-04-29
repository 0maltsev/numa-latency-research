#!/usr/bin/env bash
set -euo pipefail

BENCH_BIN="./numa_bench"
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
DURATION=5.0
PERF_ENABLED="${PERF_ENABLED:-0}"
SIZES_MB=(32 128 512)
THREADS_LIST=(1 4)
WORKLOADS_LIST=(ptr_chase sequential random false_share)

log() { printf "[%-8s] %s\n" "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

apply_tuning() {
    [[ $EUID -ne 0 ]] && die "Run as root or with sudo"
    log "Disabling NUMA balancing..."
    sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
    log "Setting CPU governor to performance..."
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$f" ] && echo performance > "$f"
    done
    log "Dropping page cache..."
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
}

install_deps() {
    if ! dpkg -l libnuma-dev 2>/dev/null | grep -q '^ii'; then
        log "Installing libnuma-dev..."
        apt-get update -qq && apt-get install -y -qq libnuma-dev >/dev/null 2>&1
    fi
    if [[ "$PERF_ENABLED" -eq 1 ]] && ! command -v perf &>/dev/null; then
        log "Installing perf..."
        apt-get install -y -qq linux-tools-common linux-tools-generic >/dev/null 2>&1
    fi
}

compile() {
    log "Compiling benchmark..."
    g++ -O3 -march=native -std=c++17 -flto \
        -o "$BENCH_BIN" main.cpp stats.cpp numa_utils.cpp workload.cpp \
        -lnuma -lpthread || die "g++ failed"
    chmod +x "$BENCH_BIN"
}

run_case() {
    local cpu=$1 mem=$2 mode=$3 wl=$4 threads=$5 size=$6 extra_flags=${7:-}
    local raw="${RESULTS_DIR}/${mode}_${wl}_t${threads}_${size}MB_raw.csv"

    [[ ! -x "$BENCH_BIN" ]] && die "Benchmark binary missing."

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
        local perf_out="${RESULTS_DIR}/${mode}_${wl}_t${threads}_${size}MB_perf.txt"
        perf stat -e cache-misses,LLC-loads,LLC-load-misses,cycles,instructions \
                  -x ',' -o "$perf_out" -- "${cmd[@]}" >> "${RESULTS_DIR}/perf_combined.csv" 2>/dev/null || true
    else
        "${cmd[@]}" >> "${RESULTS_DIR}/combined.csv"
    fi
}

main() {
    mkdir -p "$RESULTS_DIR"
    echo "mode,threads,mem_node,cpu_node,workload,size_mb,p50,p90,p99,p99_9,p999" > "${RESULTS_DIR}/combined.csv"
    [[ "$PERF_ENABLED" -eq 1 ]] && echo "mode,threads,mem_node,cpu_node,workload,size_mb,event,count" > "${RESULTS_DIR}/perf_combined.csv"

    install_deps
    apply_tuning
    compile

    log "Starting experiment matrix..."
    for spec in "0:0:local" "0:1:remote" "0:0:interleaved"; do
        IFS=':' read -r cpu mem mode <<< "$spec"
        local flags=""
        [[ "$mode" == "interleaved" ]] && flags="--interleaved"

        for wl in "${WORKLOADS_LIST[@]}"; do
            for t in "${THREADS_LIST[@]}"; do
                local run_size=1
                [[ "$wl" != "false_share" ]] && run_size="${SIZES_MB[0]}"
                for sz in "${SIZES_MB[@]}"; do
                    [[ "$wl" == "false_share" ]] && break
                    run_case "$cpu" "$mem" "$mode" "$wl" "$t" "$sz" "$flags"
                done
            done
        done
    done
    log "Done. Results: ${RESULTS_DIR}/combined.csv"
}

main "$@"