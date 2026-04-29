#include "numa_utils.h"
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

void pin_thread_to_numa(int cpu_node) {
    if (numa_available() == -1) throw std::runtime_error("NUMA not supported");

    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(cpu_node, cpus);

    int target_cpu = -1;
    for (int i = 0; i < cpus->size; ++i) {
        if (numa_bitmask_isbitset(cpus, i)) {
            target_cpu = i;
            break;
        }
    }
    if (target_cpu == -1) throw std::runtime_error("No CPUs on node " + std::to_string(cpu_node));

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(target_cpu, &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        throw std::runtime_error("sched_setaffinity failed");

    numa_run_on_node(cpu_node);
    numa_free_cpumask(cpus);
}

void* allocate_numa_memory(size_t size, const NumaConfig& cfg) {
    if (numa_available() == -1) throw std::runtime_error("NUMA not available");

    void* ptr = nullptr;
    if (cfg.huge_pages) {
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) {
            std::cerr << "[WARN] HugePages allocation failed, falling back to regular pages\n";
            ptr = nullptr;
        }
    }

    if (!ptr) {
        if (cfg.interleaved) ptr = numa_alloc_interleaved(size);
        else ptr = numa_alloc_onnode(size, cfg.mem_node);
    }

    if (!ptr) throw std::runtime_error("Memory allocation failed");

    if (cfg.huge_pages && !cfg.interleaved) {
        unsigned long nodemask[1] = {1UL << cfg.mem_node};
        if (mbind(ptr, size, MPOL_BIND, nodemask, sizeof(nodemask)*8, MPOL_MF_STRICT) != 0)
            std::cerr << "[WARN] mbind failed for hugepages\n";
    }

    return ptr;
}

void free_numa_memory(void* ptr, size_t size, bool huge_pages) {
    if (huge_pages) munmap(ptr, size);
    else numa_free(ptr, size);
}

void prefault_memory(void* ptr, size_t size) {
    volatile char* p = static_cast<volatile char*>(ptr);
    for (size_t i = 0; i < size; i += sysconf(_SC_PAGESIZE)) p[i] = 0;
    __asm__ volatile("" ::: "memory");
}

// Современная, устойчивая калибровка TSC
double calibrate_tsc_freq_mhz() {
#ifdef __x86_64__
    auto read_tsc = []() -> uint64_t {
        uint32_t lo, hi;
        __asm__ volatile("lfence; rdtsc; lfence" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    };

    uint64_t tsc_start = read_tsc();
    auto ts_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Точнее busy-wait
    auto ts_end = std::chrono::steady_clock::now();
    uint64_t tsc_end = read_tsc();

    double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_start).count();
    if (elapsed_us <= 0 || tsc_end <= tsc_start) {
        std::cerr << "[WARN] TSC calibration failed (non-monotonic). Using fallback 2500 MHz\n";
        return 2500.0;
    }

    double freq_mhz = static_cast<double>(tsc_end - tsc_start) / elapsed_us;
    // Валидация: реальные CPU 1.0 - 5.5 GHz
    if (freq_mhz < 1000.0 || freq_mhz > 6000.0) {
        std::cerr << "[WARN] TSC calibration out of range (" << freq_mhz
                  << " MHz). Using fallback 2500 MHz\n";
        return 2500.0;
    }
    return freq_mhz;
#else
    std::cerr << "[WARN] Non-x86 architecture. Using fallback TSC freq 2500 MHz\n";
    return 2500.0;
#endif
}