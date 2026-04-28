#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct LatencyStats {
    double p50;
    double p90;
    double p99;
    double p99_9;
    double p999; // 99.99%

    static LatencyStats compute(const std::vector<uint64_t>& raw_cycles, double freq_mhz);
    static void save_raw(const std::vector<uint64_t>& raw_cycles, const std::string& path);
};