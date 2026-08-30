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

/device-hardware/cuda-device-architecture

[?](https://github.com/modal-labs/gpu-glossary/issues/new)

Something seem wrong?
Or want to contribute?

Click
this button to
let us know on GitHub.

# What is a CUDA Device Architecture?

CUDA stands for *Compute Unified Device Architecture*. Depending on the context,
"CUDA" can refer to multiple distinct things: a high-level device architecture,
a
[parallel programming model](/gpu-glossary/device-software/cuda-programming-model)
for architectures with that design, or a
[software platform](/gpu-glossary/host-software/cuda-software-platform)  that
extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the *device architecture* part of CUDA. The core feature of a
"compute unified device architecture" is simplicity, relative to preceding GPU
architectures.

Prior to the GeForce 8800 and the Tesla data center GPUs it spawned, NVIDIA GPUs
were designed with a complex pipeline shader architecture that mapped software
shader stages onto heterogeneous, specialized hardware units. This architecture
was challenging for the software and hardware sides alike: it required software
engineers to map programs onto a fixed pipeline and forced hardware engineers to
guess the load ratios between pipeline steps.

GPU devices with a unified architecture are much simpler: the hardware units are
entirely uniform, each capable of a wide array of computations. These units are
known as
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
and their main subcomponents are the
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core)  and (for recent GPUs)
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core) .

For an accessible introduction to the history and design of CUDA hardware
architectures, see [this blog post](https://fabiensanglard.net/cuda/)  by Fabien
Sanglard. That blog post cites its (high-quality) sources, like NVIDIA's
[Fermi Compute Architecture white paper](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf) .
The white paper by
[Lindholm et al. in 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
introducing the Tesla architecture is both well-written and thorough. The
[NVIDIA whitepaper for the Tesla P100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
is less scholarly but documents the introduction of a number of features that
are critical for today's large-scale neural network workloads, like NVLink and
[on-package high-bandwidth memory](/gpu-glossary/device-hardware/gpu-ram) .

[![Modal Logo](data:image/svg+xml...)

## Building on GPUs? We know a thing or two about it.

Modal is an ergonomic Python SDK wrapped around a global GPU fleet. Deploy serverless AI workloads instantly without worrying
about quota requests, driver compatibility issues, or managing
bulky ML dependencies.

Deploy serverless AI workloads instantly without worrying about quota
requests, driver compatibility issues, or managing bulky ML
dependencies.

Deploy on GPUs](/signup)

[Device Hardware](/gpu-glossary/device-hardware) [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor)