#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class WorkloadType { POINTER_CHASE, RANDOM_ACCESS, SEQUENTIAL_SCAN, FALSE_SHARING };

struct WorkloadParams {
    void* mem;
    size_t size_bytes;
    WorkloadType type;
    bool is_shared;
    int thread_id;
    int stride;
};

struct WorkloadState {
    uint64_t* ptr_chain;
    size_t chain_len;
    size_t* indices;
    size_t current_idx;
    void* shared_buf;
    uint64_t* private_buf;
};

void prepare_workload(WorkloadState& state, const WorkloadParams& params);
void run_workload(const WorkloadParams& params, WorkloadState& state, size_t iterations, std::vector<uint64_t>& out_cycles);
void destroy_workload(WorkloadState& state);