[![GPU Glossary](/_app/immutable/assets/modal-logo-terminal.Cs4Cm_SQ.svg)

GPU Glossary](/)

GPU Glossary

Terminal Light green Light

[Deploy on GPUs](/signup)

TABLE OF CONTENTS

[Home](/gpu-glossary)  -

[README](/gpu-glossary/readme)

[Device Hardware](/gpu-glossary/device-hardware)  -

[CUDA (Device Architecture)](/gpu-glossary/device-hardware/cuda-device-architecture)

[Streaming Multiprocessor

SM](/gpu-glossary/device-hardware/streaming-multiprocessor)

[Core](/gpu-glossary/device-hardware/core)

[Special Function Unit

SFU](/gpu-glossary/device-hardware/special-function-unit)

[Load/Store Unit

LSU](/gpu-glossary/device-hardware/load-store-unit)

[Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler)

[CUDA Core](/gpu-glossary/device-hardware/cuda-core)

[Tensor Core](/gpu-glossary/device-hardware/tensor-core)

[Tensor Memory Accelerator

TMA](/gpu-glossary/device-hardware/tensor-memory-accelerator)

[Streaming Multiprocessor Architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)

[Texture Processing Cluster

TPC](/gpu-glossary/device-hardware/texture-processing-cluster)

[Graphics/GPU Processing Cluster

GPC](/gpu-glossary/device-hardware/graphics-processing-cluster)

[Register File](/gpu-glossary/device-hardware/register-file)

[L1 Data Cache](/gpu-glossary/device-hardware/l1-data-cache)

[Tensor Memory](/gpu-glossary/device-hardware/tensor-memory)

[GPU RAM](/gpu-glossary/device-hardware/gpu-ram)

[Device Software](/gpu-glossary/device-software)  -

[CUDA (Programming Model)](/gpu-glossary/device-software/cuda-programming-model)

[Streaming ASSembler

SASS](/gpu-glossary/device-software/streaming-assembler)

[Parallel Thread eXecution

PTX](/gpu-glossary/device-software/parallel-thread-execution)

[Compute Capability](/gpu-glossary/device-software/compute-capability)

[Thread](/gpu-glossary/device-software/thread)

[Warp](/gpu-glossary/device-software/warp)

[Warpgroup](/gpu-glossary/device-software/warpgroup)

[Cooperative Thread Array](/gpu-glossary/device-software/cooperative-thread-array)

[Kernel](/gpu-glossary/device-software/kernel)

[Thread Block](/gpu-glossary/device-software/thread-block)

[Thread Block Grid](/gpu-glossary/device-software/thread-block-grid)

[Thread Hierarchy](/gpu-glossary/device-software/thread-hierarchy)

[Memory Hierarchy](/gpu-glossary/device-software/memory-hierarchy)

[Registers](/gpu-glossary/device-software/registers)

[Shared Memory](/gpu-glossary/device-software/shared-memory)

[Global Memory](/gpu-glossary/device-software/global-memory)

[CUDA Tile Programming Model](/gpu-glossary/device-software/cuda-tile-programming-model)

[Host Software](/gpu-glossary/host-software)  -

[CUDA (Software Platform)](/gpu-glossary/host-software/cuda-software-platform)

[CUDA C++ (programming language)](/gpu-glossary/host-software/cuda-c)

[NVIDIA GPU Drivers](/gpu-glossary/host-software/nvidia-gpu-drivers)

[nvidia.ko](/gpu-glossary/host-software/nvidia-ko)

[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api)

[libcuda.so](/gpu-glossary/host-software/libcuda)

[NVIDIA Management Library

NVML](/gpu-glossary/host-software/nvml)

[libnvml.so](/gpu-glossary/host-software/libnvml)

[nvidia-smi](/gpu-glossary/host-software/nvidia-smi)

[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api)

[libcudart.so](/gpu-glossary/host-software/libcudart)

[CUDA Graphs](/gpu-glossary/host-software/cuda-graph)

[NVIDIA CUDA Compiler Driver

nvcc](/gpu-glossary/host-software/nvcc)

[NVIDIA Runtime Compiler](/gpu-glossary/host-software/nvrtc)

[NVIDIA CUDA Profiling Tools Interface

CUPTI](/gpu-glossary/host-software/cupti)

[NVIDIA Nsight Systems](/gpu-glossary/host-software/nsight-systems)

[CUDA Binary Utilities](/gpu-glossary/host-software/cuda-binary-utilities)

[cuBLAS](/gpu-glossary/host-software/cublas)

[cuDNN](/gpu-glossary/host-software/cudnn)

[CUTLASS](/gpu-glossary/host-software/cutlass)

[CuTe](/gpu-glossary/host-software/cute)

[CuTe DSL](/gpu-glossary/host-software/cute-dsl)

[Performance](/gpu-glossary/perf)  -

[Performance Bottleneck](/gpu-glossary/perf/performance-bottleneck)

[Roofline Model](/gpu-glossary/perf/roofline-model)

[Compute-bound](/gpu-glossary/perf/compute-bound)

[Memory-bound](/gpu-glossary/perf/memory-bound)

[Arithmetic Intensity](/gpu-glossary/perf/arithmetic-intensity)

[Overhead](/gpu-glossary/perf/overhead)

[Little's Law](/gpu-glossary/perf/littles-law)

[Memory Bandwidth](/gpu-glossary/perf/memory-bandwidth)

[Arithmetic Bandwidth](/gpu-glossary/perf/arithmetic-bandwidth)

[Latency Hiding](/gpu-glossary/perf/latency-hiding)

[Warp Execution State](/gpu-glossary/perf/warp-execution-state)

[Active Cycle](/gpu-glossary/perf/active-cycle)

[Occupancy](/gpu-glossary/perf/occupancy)

[Pipe Utilization](/gpu-glossary/perf/pipe-utilization)

[Peak Rate](/gpu-glossary/perf/peak-rate)

[Issue Efficiency](/gpu-glossary/perf/issue-efficiency)

[Streaming Multiprocessor Utilization](/gpu-glossary/perf/streaming-multiprocessor-utilization)

[Warp Divergence](/gpu-glossary/perf/warp-divergence)

[Scoreboard Stall](/gpu-glossary/perf/scoreboard-stall)

[Branch Efficiency](/gpu-glossary/perf/branch-efficiency)

[Memory Coalescing](/gpu-glossary/perf/memory-coalescing)

[Bank Conflict](/gpu-glossary/perf/bank-conflict)

[Register Pressure](/gpu-glossary/perf/register-pressure)

[Contributors](/gpu-glossary/contributors)

/device-hardware/streaming-multiprocessor

[?](https://github.com/modal-labs/gpu-glossary/issues/new)

Something seem wrong?
Or want to contribute?

Click
this button to
let us know on GitHub.

# What is a Streaming Multiprocessor?

SM

When we [program GPUs](/gpu-glossary/host-software/cuda-software-platform) , we
produce
[sequences of instructions](/gpu-glossary/device-software/streaming-assembler)
for its Streaming Multiprocessors to carry out.

Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the
cores of CPUs. That is, SMs both execute computations and store state available
for computation in registers, with associated caches. Compared to CPU cores, GPU
SMs are simple, weak processors. Execution in SMs is pipelined within an
instruction (as in almost all CPUs since the 1990s) but there is no speculative
execution or instruction pointer prediction (unlike all contemporary
high-performance CPUs).

However, GPU SMs can execute more
[threads](/gpu-glossary/device-software/thread)  in parallel.

For comparison: an
[AMD EPYC 9965](https://www.techpowerup.com/cpu-specs/epyc-9965.c3904)  CPU draws
at most 500 W and has 192 cores, each of which can execute instructions for at
most two threads at a time, for a total of 384 threads in parallel, running at
about 1.25 W per thread.

An H100 SXM GPU draws at most 700 W and has 132 SMs, each of which has four
[Warp Schedulers](/gpu-glossary/device-hardware/warp-scheduler)  that can each
issue instructions to 32 threads (aka a
[warp](/gpu-glossary/device-software/warp) ) in parallel per clock cycle, for a
total of 128 × 132 > 16,000 parallel threads running at about 5 cW apiece. Note
that this is truly parallel: each of the 16,000 threads can make progress with
each clock cycle.

GPU SMs also support a large number of *concurrent* threads -- threads of
execution whose instructions are interleaved.

A single SM on an H100 can concurrently execute up to 2048 threads split across
64 thread groups of 32 threads each. With 132 SMs, that's a total of over
250,000 concurrent threads.

CPUs can also run many threads concurrently. But switches between
[warps](/gpu-glossary/device-software/warp)  happen at the speed of a single
clock cycle (over 1000x faster than context switches on a CPU), again powered by
the SM's [Warp Schedulers](/gpu-glossary/device-hardware/warp-scheduler) . The
volume of available [warps](/gpu-glossary/device-software/warp)  and the speed of
[warp switches](/gpu-glossary/device-hardware/warp-scheduler)  help
[hide latency](/gpu-glossary/perf/latency-hiding)  caused by memory reads, thread
synchronization, or other expensive instructions, ensuring that the
[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth)  provided by the
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core)  and
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core)  is well utilized.

This [latency-hiding](/gpu-glossary/perf/latency-hiding)  is the secret to GPUs'
strengths. CPUs seek to hide latency from end-users and programmers by
maintaining large, hardware-managed caches and sophisticated instruction
prediction. This extra hardware limits the fraction of their silicon area,
power, and heat budgets that CPUs can allocate to computation.

For programs or functions like neural network inference or sequential database
scans for which it is relatively straightforward for programmers to
[express](/gpu-glossary/device-software/cuda-programming-model)  the behavior of
[caches](/gpu-glossary/device-hardware/l1-data-cache)  — e.g. store a chunk of
each input matrix and keep it in cache for long enough to compute the related
outputs — the result is much higher throughput.

[![Modal Logo](data:image/svg+xml...)

## Building on GPUs? We know a thing or two about it.

Modal is an ergonomic Python SDK wrapped around a global GPU fleet. Deploy serverless AI workloads instantly without worrying
about quota requests, driver compatibility issues, or managing
bulky ML dependencies.

Deploy serverless AI workloads instantly without worrying about quota
requests, driver compatibility issues, or managing bulky ML
dependencies.

Deploy on GPUs](/signup)

[CUDA (Device Architecture)](/gpu-glossary/device-hardware/cuda-device-architecture) [Core](/gpu-glossary/device-hardware/core)