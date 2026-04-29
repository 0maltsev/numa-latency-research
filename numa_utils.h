#pragma once
#include <cstddef>

struct NumaConfig {
    int cpu_node;
    int mem_node;
    bool interleaved;
    bool huge_pages;
};

void pin_thread_to_numa(int cpu_node);
void* allocate_numa_memory(size_t size, const NumaConfig& cfg);
void free_numa_memory(void* ptr, size_t size, bool huge_pages);
void prefault_memory(void* ptr, size_t size);
double calibrate_tsc_freq_mhz();