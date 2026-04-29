#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include <cstring>
#include <numa.h>

#include "stats.h"
#include "numa_utils.h"
#include "workload.h"

struct BenchmarkConfig {
    int threads = 1;
    int cpu_node = 0;
    int mem_node = 0;
    size_t size_mb = 64;
    double duration_sec = 2.0;
    WorkloadType workload = WorkloadType::POINTER_CHASE;
    bool interleaved = false;
    bool huge_pages = false;
    std::string mode_str = "local";
    bool warmup = true;
};

struct ThreadResult {
    std::vector<uint64_t> cycles;
};

static WorkloadType parse_workload(const std::string& s) {
    if (s == "ptr_chase") return WorkloadType::POINTER_CHASE;
    if (s == "random") return WorkloadType::RANDOM_ACCESS;
    if (s == "sequential") return WorkloadType::SEQUENTIAL_SCAN;
    if (s == "false_share") return WorkloadType::FALSE_SHARING;
    throw std::runtime_error("Unknown workload: " + s);
}

static void worker_thread(const BenchmarkConfig& cfg, const WorkloadParams& params, ThreadResult& result) {
    WorkloadState state{};
    prepare_workload(state, params);

    if (cfg.warmup) {
        run_workload(params, state, 1'000'000, result.cycles);
        result.cycles.clear();
    }

    auto start_time = std::chrono::steady_clock::now();
    size_t batch = 100'000;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - start_time).count() >= cfg.duration_sec) break;
        run_workload(params, state, batch, result.cycles);
    }
    destroy_workload(state);
}

int main(int argc, char* argv[]) {
    BenchmarkConfig cfg;
    std::string workload_str = "ptr_chase";
    std::string raw_out = "latencies_raw.csv";

    static struct option long_opts[] = {
        {"threads", required_argument, nullptr, 't'},
        {"cpu_node", required_argument, nullptr, 'c'},
        {"mem_node", required_argument, nullptr, 'm'},
        {"size_mb", required_argument, nullptr, 's'},
        {"duration", required_argument, nullptr, 'd'},
        {"workload", required_argument, nullptr, 'w'},
        {"mode", required_argument, nullptr, 'o'},
        {"interleaved", no_argument, nullptr, 'i'},
        {"huge_pages", no_argument, nullptr, 'h'},
        {"raw_output", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:c:m:s:d:w:o:ir:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 't': cfg.threads = std::stoi(optarg); break;
            case 'c': cfg.cpu_node = std::stoi(optarg); break;
            case 'm': cfg.mem_node = std::stoi(optarg); break;
            case 's': cfg.size_mb = std::stoul(optarg); break;
            case 'd': cfg.duration_sec = std::stod(optarg); break;
            case 'w': workload_str = optarg; break;
            case 'o': cfg.mode_str = optarg; break;
            case 'i': cfg.interleaved = true; break;
            case 'h': cfg.huge_pages = true; break;
            case 'r': raw_out = optarg; break;
            default: exit(EXIT_FAILURE);
        }
    }
    cfg.workload = parse_workload(workload_str);

    if (numa_available() == -1) {
        std::cerr << "Error: NUMA not supported on this system.\n";
        return 1;
    }

    double freq_mhz = calibrate_tsc_freq_mhz();
    std::cerr << "[INFO] Calibrated TSC: " << freq_mhz << " MHz\n";

    size_t total_size = cfg.size_mb * 1024ULL * 1024ULL;
    if (cfg.workload == WorkloadType::FALSE_SHARING) total_size = 64 * cfg.threads;

    NumaConfig ncfg{cfg.cpu_node, cfg.mem_node, cfg.interleaved, cfg.huge_pages};
    void* shared_mem = allocate_numa_memory(total_size, ncfg);
    prefault_memory(shared_mem, total_size);

    std::cerr << "[INFO] Memory allocated on node " << cfg.mem_node << " (interleaved="
              << cfg.interleaved << ", hp=" << cfg.huge_pages << ")\n";

    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(cfg.threads);

    for (int i = 0; i < cfg.threads; ++i) {
        threads.emplace_back([&cfg, i, shared_mem, total_size, &results]() {
            pin_thread_to_numa(cfg.cpu_node);
            WorkloadParams wp{
                .mem = shared_mem,
                .size_bytes = total_size,
                .type = cfg.workload,
                .is_shared = (cfg.workload == WorkloadType::FALSE_SHARING),
                .thread_id = i,
                .stride = 1
            };
            worker_thread(cfg, wp, results[i]);
        });
    }

    for (auto& t : threads) t.join();

    std::vector<uint64_t> merged_cycles;
    size_t total_samples = 0;
    for (auto& r : results) total_samples += r.cycles.size();
    merged_cycles.reserve(total_samples);
    for (auto& r : results) {
        merged_cycles.insert(merged_cycles.end(), r.cycles.begin(), r.cycles.end());
        r.cycles.clear();
    }

    LatencyStats stats = LatencyStats::compute(merged_cycles, freq_mhz);
    LatencyStats::save_raw(merged_cycles, raw_out);

    std::cout << "mode,threads,mem_node,cpu_node,workload,p50,p90,p99,p99_9,p999\n";
    std::cout << cfg.mode_str << "," << cfg.threads << "," << cfg.mem_node << ","
              << cfg.cpu_node << "," << workload_str << ","
              << stats.p50 << "," << stats.p90 << "," << stats.p99 << ","
              << stats.p99_9 << "," << stats.p999 << "\n";

    free_numa_memory(shared_mem, total_size, cfg.huge_pages);
    return 0;
}