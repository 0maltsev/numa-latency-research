#include "workload.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include <atomic>

// LCG для генерации индексов без вызова rand() в hot-path
static inline size_t lcg_next(size_t state) {
    return (state * 6364136223846793005ULL + 1442695040888963407ULL) & 0x7FFFFFFFFFFFFFFFULL;
}

static inline uint64_t read_tsc_serialized() {
    uint64_t tsc;
    __asm__ volatile("lfence\nrdtsc\nlfence" : "=A"(tsc));
    return tsc;
}

void prepare_workload(WorkloadState& state, const WorkloadParams& params) {
    state.current_idx = 0;
    state.ptr_chain = nullptr;
    state.indices = nullptr;
    state.shared_buf = nullptr;
    state.private_buf = nullptr;

    size_t elements = params.size_bytes / sizeof(uint64_t);
    if (elements < 1) throw std::runtime_error("Working set too small");

    if (params.type == WorkloadType::POINTER_CHASE) {
        // Инициализация цепочки указателей (random permutation)
        state.private_buf = static_cast<uint64_t*>(params.mem);
        state.chain_len = elements;

        // Fisher-Yates shuffle для создания случайного графа доступа
        for (size_t i = 0; i < elements; ++i) state.private_buf[i] = i;
        for (size_t i = elements - 1; i > 0; --i) {
            size_t j = lcg_next(i) % (i + 1);
            std::swap(state.private_buf[i], state.private_buf[j]);
        }
        // Замыкаем цикл
        state.private_buf[elements-1] = 0;

    } else if (params.type == WorkloadType::RANDOM_ACCESS) {
        state.private_buf = static_cast<uint64_t*>(params.mem);
        size_t num_indices = std::min(elements, 1024UL * 1024); // Кэш индексов
        state.indices = new size_t[num_indices];
        size_t state_rng = 42;
        for (size_t i = 0; i < num_indices; ++i) {
            state_rng = lcg_next(state_rng);
            state.indices[i] = state_rng % elements;
        }

    } else if (params.type == WorkloadType::SEQUENTIAL_SCAN) {
        state.private_buf = static_cast<uint64_t*>(params.mem);
        state.current_idx = 0;

    } else if (params.type == WorkloadType::FALSE_SHARING) {
        // Выделяем буфер с выравниванием на 64B
        state.shared_buf = params.mem;
    }
}

void run_workload(const WorkloadParams& params, WorkloadState& state, size_t iterations, std::vector<uint64_t>& out_cycles) {
    out_cycles.reserve(iterations);
    uint64_t dummy = 0; // Зависимость для компилятора

    if (params.type == WorkloadType::POINTER_CHASE) {
        uint64_t* ptr = state.private_buf;
        for (size_t i = 0; i < iterations; ++i) {
            uint64_t start = read_tsc_serialized();
            ptr = reinterpret_cast<uint64_t*>(*ptr); // Зависимость данных блокирует OoO
            uint64_t end = read_tsc_serialized();
            out_cycles.push_back(end - start);
        }
        dummy += reinterpret_cast<uintptr_t>(ptr);

    } else if (params.type == WorkloadType::RANDOM_ACCESS) {
        uint64_t* buf = state.private_buf;
        size_t* idxs = state.indices;
        size_t num_idxs = std::min(params.size_bytes / sizeof(uint64_t), 1024UL * 1024UL);
        size_t cnt = 0;
        for (size_t i = 0; i < iterations; ++i) {
            uint64_t start = read_tsc_serialized();
            size_t idx = idxs[cnt % num_idxs];
            dummy ^= buf[idx]; // Зависимость чтения
            __asm__ volatile("" ::: "memory"); // Barrier против compiler reordering
            uint64_t end = read_tsc_serialized();
            out_cycles.push_back(end - start);
            ++cnt;
        }

    } else if (params.type == WorkloadType::SEQUENTIAL_SCAN) {
        uint64_t* buf = state.private_buf;
        size_t elements = params.size_bytes / sizeof(uint64_t);
        size_t idx = state.current_idx;
        for (size_t i = 0; i < iterations; ++i) {
            uint64_t start = read_tsc_serialized();
            dummy ^= buf[idx];
            idx = (idx + 1) % elements;
            uint64_t end = read_tsc_serialized();
            out_cycles.push_back(end - start);
        }
        state.current_idx = idx;

    } else if (params.type == WorkloadType::FALSE_SHARING) {
        // Каждый поток пишет/читает в своей 64-байтной строке
        volatile uint64_t* buf = static_cast<volatile uint64_t*>(state.shared_buf);
        size_t offset = (params.thread_id * 64) / sizeof(uint64_t);
        for (size_t i = 0; i < iterations; ++i) {
            uint64_t start = read_tsc_serialized();
            buf[offset] = buf[offset] + 1; // Write -> RFO -> Invalidate
            __asm__ volatile("" ::: "memory");
            dummy ^= buf[offset];
            uint64_t end = read_tsc_serialized();
            out_cycles.push_back(end - start);
        }
    }

    // Предотвращаем dead-code elimination
    __asm__ volatile("" ::: "memory");
}

void destroy_workload(WorkloadState& state) {
    delete[] state.indices;
}