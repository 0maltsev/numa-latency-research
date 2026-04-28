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

void pin_thread_to_numa(int cpu_node) {
    if (numa_available() == -1) throw std::runtime_error("NUMA not supported");

    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(cpu_node, cpus);

    // Pick first available core on the node
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

    // Bind memory to node explicitly if not using libnuma allocators
    if (cfg.huge_pages && !cfg.interleaved) {
        unsigned long nodemask[1] = {1UL << cfg.mem_node};
        if (mbind(ptr, size, MPOL_BIND, nodemask, sizeof(nodemask)*8, MPOL_MF_STRICT) != 0)
            std::cerr << "[WARN] mbind failed for hugepages\n";
    }

    return ptr;
}

void free_numa_memory(void* ptr, size_t size) {
    if (cfg.huge_pages) munmap(ptr, size);
    else numa_free(ptr, size);
}

void prefault_memory(void* ptr, size_t size) {
    // Touch every page to avoid page faults during measurement
    volatile char* p = static_cast<volatile char*>(ptr);
    for (size_t i = 0; i < size; i += sysconf(_SC_PAGESIZE)) p[i] = 0;
    __asm__ volatile("" ::: "memory"); // Compiler barrier
}

double calibrate_tsc_freq_mhz() {
    uint64_t tsc_start, tsc_end;
    __asm__ volatile("lfence\nrdtsc\nlfence" : "=A"(tsc_start));

    auto ns_start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - ns_start).count() < 100) {}
    auto ns_end = std::chrono::high_resolution_clock::now();

    __asm__ volatile("lfence\nrdtsc\nlfence" : "=A"(tsc_end));

    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ns_end - ns_start).count();
    return (tsc_end - tsc_start) / (elapsed_ns * 0.001); // MHz
}