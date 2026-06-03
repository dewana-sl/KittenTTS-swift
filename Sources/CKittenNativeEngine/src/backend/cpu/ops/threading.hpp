#pragma once

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kt_cpu {

inline thread_local int g_thread_id = 0;
inline thread_local int g_thread_count = 1;
inline thread_local bool g_in_parallel = false;

inline int configured_thread_count()
{
    if (const char* env = std::getenv("KITTEN_NATIVE_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) return parsed;
    }

    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    return std::clamp(static_cast<int>(hw), 1, 8);
}

inline int worker_count(size_t work_items)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    const int requested = configured_thread_count();
    if (work_items == 0) return requested;
    return std::max(1, std::min(requested, static_cast<int>(work_items)));
#endif
}

inline bool in_parallel_region()
{
#ifdef _OPENMP
    if (omp_in_parallel()) return true;
#endif
    return g_in_parallel;
}

inline int current_thread_id()
{
#ifdef _OPENMP
    if (omp_in_parallel()) return omp_get_thread_num();
#endif
    return g_thread_id;
}

inline int current_thread_count()
{
#ifdef _OPENMP
    if (omp_in_parallel()) return omp_get_num_threads();
#endif
    return g_thread_count;
}

template <typename Fn>
void parallel_run(int nthreads, Fn&& fn)
{
    nthreads = std::max(1, nthreads);
    if (nthreads == 1 || g_in_parallel) {
        fn(0, 1);
        return;
    }

    const int prev_id = g_thread_id;
    const int prev_count = g_thread_count;
    const bool prev_parallel = g_in_parallel;

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nthreads - 1));
    for (int tid = 1; tid < nthreads; ++tid) {
        workers.emplace_back([&, tid] {
            g_thread_id = tid;
            g_thread_count = nthreads;
            g_in_parallel = true;
            fn(tid, nthreads);
            g_thread_id = 0;
            g_thread_count = 1;
            g_in_parallel = false;
        });
    }

    g_thread_id = 0;
    g_thread_count = nthreads;
    g_in_parallel = true;
    fn(0, nthreads);

    for (auto& worker : workers) worker.join();

    g_thread_id = prev_id;
    g_thread_count = prev_count;
    g_in_parallel = prev_parallel;
}

} // namespace kt_cpu
