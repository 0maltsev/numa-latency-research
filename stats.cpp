#include "stats.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>

LatencyStats LatencyStats::compute(const std::vector<uint64_t>& raw_cycles, double freq_mhz) {
    if (raw_cycles.empty()) throw std::runtime_error("No samples collected");

    std::vector<uint64_t> sorted = raw_cycles;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) -> uint64_t {
        // Standard nearest-rank method
        size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size())) - 1;
        return sorted[idx];
    };

    double to_ns = 1000.0 / freq_mhz;
    return {
        percentile(50.0) * to_ns,
        percentile(90.0) * to_ns,
        percentile(99.0) * to_ns,
        percentile(99.9) * to_ns,
        percentile(99.99) * to_ns
    };
}

void LatencyStats::save_raw(const std::vector<uint64_t>& raw_cycles, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open raw output file: " + path);
    ofs << "cycle_latencies\n";
    for (auto c : raw_cycles) ofs << c << "\n";
}