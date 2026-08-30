[![Standard Kernel](/logo/logo1.svg)](/)

[About](/)
[Blog](/blog)
[Join](https://docs.google.com/forms/d/e/1FAIpQLSc7_F7F6LLh_yX7k2_v874P72t4KXgqX6Mda5fOQyLFOTa8Kg/viewform?usp=publish-editor)

[← Blog](/blog)

24 February 2026

# “This Kernel Was Faster Yesterday” — In Pursuit of High-Fidelity GPU Kernel Benchmarking

![](images/image6.png)

*After Salvador Dalí’s *The Persistence of Memory**

# TL;DR

* **GPU timing is deceptively hard and highly variable**: power limits, thermal state, clock behavior, idle transitions, caching, and measurement methods all matter.
* **High-fidelity evaluation is critical,** especially for automated RL systems.In high-value matmul kernels, where even 5% matters, measurement noise can look like real gains and mislead RL into reinforcing changes that don’t improve true performance.
* **Accurate kernel benchmarking requires more than just common-sense hygiene** (synchronizing correctly, isolating gpu resources during benchmarking, timing with the lowest-overhead, warming up the GPU before measuring, and aggregating results over many trials). **Subtle factors can materially change benchmarking outcomes:**

* **Clock speed cannot be truly locked under power constraints.** If a workload pushes into the power limit, hardware will throttle regardless of manual settings, so “locked” frequencies are not guaranteed and runtime will vary accordingly.
* **Sleeping between trials reduces dynamic clock effects but shifts the GPU out of steady state.** Although power draw drops and clocks appear stable, the workload becomes bursty rather than continuous, so the measured runtime no longer reflects sustained, real-world throughput performance.
* **Sensitivity to cache state (cold vs. warm) varies by workload.** Be explicit about which condition you measure since the right choice depends on the behavior you intend to evaluate.
* **Timing methods matter—and the “right” one depends on the problem.** For microsecond-scale kernels, true execution time is often inseparable from measurement overhead.
* **Compilation flags can change GPU performance** on the same kernel and hardware.
* **Getting identical hardware with same settings from cloud providers is not guaranteed.**

## The Essential Timing Basics

Let's start with the fundamentals — the common-sense basics that every timing script should follow.

### Synchronize correctly.

The most common mistake is forgetting proper synchronization. CUDA kernel launches are asynchronous, so if you start a timer, launch a kernel, and then stop the timer without synchronizing, you’re only measuring launch overhead—not the kernel’s execution time.

Another important detail is stream consistency. You must record timing events in the **same stream** where the kernel is launched. If you launch the kernel in one stream but record and synchronize events in a different stream, the synchronization can complete immediately. This creates the false impression that the kernel finished instantly, even though it is still executing in another stream:

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);
matmul<<<grid, threads, 0,stream2>>>(d_A, d_B, d_C, N);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
```

### Do warmup runs before timing.

The first run is usually dramatically slower due to factors such as JIT compilation, cache cold start, and power or voltage ramp-up, among others. Always discard early iterations and measure after reaching steady state.

![](images/image25.png)

### Avoid GPU resource contention.

We want to understand the impact of both CPU and GPU resource contention on kernel benchmarking. To do this, we benchmark a 4096×4096 BF16 GEMM (MatmulFP16) kernel under four conditions: free GPU + low CPU, free GPU + busy CPU, busy GPU + low CPU, and both busy.

GPU contention is introduced by launching three background CUDA streams launched from separate host threads that continuously run vector-add kernels in separate Python threads, forcing the target kernel to compete for GPU resources. CPU contention is created by running 16 Python threads executing NumPy BLAS matrix multiplies in tight loops.

![](images/image2.png)

We observe that CPU contention alone has negligible impact (~+0.3%), indicating that once dispatched, a GPU kernel runs largely independent of CPU load. In contrast, GPU contention significantly increases mean latency (~+21%) and, more importantly, dramatically amplifies variance (30× increase in standard deviation), with P95 rising substantially due to intermittent stalls from shared resource pressure. When both CPU and GPU are busy (~+36%), performance degrades further, but the dominant bottleneck remains GPU-side resource sharing rather than CPU load.

### Turn on persistent mode.

Persistent mode keeps the GPU driver loaded and the device initialized even when no processes are actively using it. This typically reduces startup latency and avoids repeated driver reinitialization. It is generally considered best practice for benchmarking because it can improve consistency across runs. In our experiments, however, enabling persistent mode did not produce a noticeable difference in steady-state kernel timing, likely because we were already running continuous workloads that kept the GPU active. You can enable persistent mode with `sudo nvidia-smi -pm 1` and verify it using`nvidia-smi -q | grep "Persistence Mode"`.

### Aggregate timing statistics.

In this experiment, we deliberately create noisy timing conditions to study how different aggregation methods behave with outliers. We compare the minimum, mean, median, and a trimmed mean that removes the top 10% and bottom 10% of samples. As expected, the arithmetic mean is the most sensitive to outliers and inflates noticeably under heavy-tailed latency spikes. The trimmed mean reduces this sensitivity and typically moves closer to the median, providing a more stable estimate of typical performance. The minimum reflects best-case execution under ideal conditions, but it may not represent realistic operating environments. The appropriate aggregation method ultimately depends on what you care about — peak capability, typical latency, or robustness.

![](images/image27.png)

## Hardware Effects: Performance Is Not Constant

### Dynamic Clock Frequency and Power

#### Clock frequency directly determines execution speed.

GPUs expose two primary clocks that matter for performance: the SM clock controls the frequency of the streaming multiprocessors (compute units), and the Memory (DRAM) clock

controls the frequency of the HBM/GDDR memory subsystem. You can observe both with

```
nvidia-smi --query-gpu=clocks.sm,clocks.mem
```

Clock frequency directly determines execution speed. Higher SM frequency means more instructions are retired per second, which translates into lower kernel latency; for example timing could be 4x different for a matrix multiplication operation depending on the SM clock frequency the GPU is using.

![](images/image20.png)

Modern GPUs do not run at a fixed frequency. Instead, they use **dynamic frequency scaling** in which the hardware continuously adjusts SM and memory clocks based on factors such as power draw, temperature, voltage limits, and workload characteristics. What you observe in nvidia-smi is the output of a real-time control system reacting to physical constraints.

#### The clock gets throttled when power limit is reached.

We were baffled why our matmul time depends on how many trials we run, and the answer turns out to be dynamic clocks. If maximum power draw stays below the configured power limit, the SM clocks and latency remain stable. This is true even without explicitly locking clocks to a fixed frequency (we will discuss more in the next section). In this regime, clock locking often adds little benefit because the GPU naturally settles at a sustainable frequency. In our case a 2k\*2k\*2k matmul time stays consistent over 500 trials.

![](images/image10.png)

We then increased the size of the matmul to 16k\*16k\*16k such that the workload consistently hits the limit, and the behavior changes very clearly. Power draw now fluctuates around the 700W ceiling, and the SM clock frequency adjusts dynamically in response. As the clock drops and rises to enforce the cap, kernel latency moves with it.

![](images/image13.png)

#### Clock frequency locking isn't guaranteed with power limit is reached.

Clock frequencies can be controlled using nvidia-smi.

Set application clocks

```
sudo nvidia-smi -ac <mem_clock>,<sm_clock>
```

Reset to default clocks

```
sudo nvidia-smi -rac
```

However, manual clock control is restricted. You cannot arbitrarily set memory clocks, and SM clock selection is limited to supported application clock pairs rather than fully free frequency values. Some systems may not permit manual clock changes at all. Nvidia-smi shows the list of supported clock frequency configurations:

```
nvidia-smi -q -d SUPPORTED_CLOCKS
```

**Even when clock locking is available, it is not a guarantee of stability** if the selected frequency drives the GPU into the power limit. If the locked SM clock causes power draw to exceed the configured cap, the hardware will still throttle to enforce the limit. In other words, power constraints override manual clock settings, so locking only works reliably when the chosen frequency keeps power consumption below the ceiling.

Locking the SM clock to 1980 MHz does not truly “lock” the frequency. At that setting, the matmul workload drives power draw up to the 700W cap, so the hardware intervenes and throttles the clock downward to enforce the limit. As a result, we still observe clock dropping below 1980MHz and corresponding latency increases –– manual locking is effectively overridden by the power constraint.

In contrast, locking the SM clock to 1200 MHz does produce a stable frequency. At this lower setting, power draw remains below 700 W, so the GPU has no reason to throttle. The clock stays fixed at 1200 MHz and kernel runtime becomes stable, but the kernel runtime is slower as a result of the lower frequency.

![](images/image15.png)

#### Sleeping between trials stabilizes power draw but shifts GPU out of steady state.

A practice we sometimes observe in benchmarking scripts is inserting sleep intervals between trials to mitigate potential thermal or power related effects.

Without sleeping between trials, the workload runs continuously and power draw quickly rises to 700W during the first ~50 iterations. In this steady-state regime, the average kernel time is 12.5 ms as shown earlier.

When we insert a 500 ms sleep between each timed trial, power consumption stays much lower, around 150–160W, and the reported SM clock stabilizes at approximately 1970–1980 MHz. Under these conditions, the measured kernel time decreases to 11.04 ms.

![](images/image1.png)

We also experimented with different sleep durations and observed that longer sleep intervals consistently led to faster average kernel times. While sleeping reduces sustained power draw and mitigates thermal and power-limit effects, the measured kernel times reflect a bursty, low-duty-cycle scenario rather than sustained throughput performance, and therefore may not represent real-world steady-state behavior.

| **Config** | **Mean (ms)** | **Std (ms)** |
| --- | --- | --- |
| No Sleep | 12.36 | 0.89 |
| Sleep 20ms | 11.11 | 0.08 |
| Sleep 100ms | 11.07 | 0.02 |
| Sleep 500ms | 11.04 | 0.02 |

#### Power also affects end-to-end performance.

Power and clock settings don’t just affect microbenchmarks, they can change end-to-end outcomes. In some real workloads, a slightly slower kernel with lower power draw can avoid throttling and lead to better overall system throughput. This blog focused on timing and power behavior in isolation and demonstrates its effects in ablation studies, but real-world performance also depends on how kernels interact with power limits, thermals, and sustained execution.

### ECC

ECC (Error-Correcting Code) mode enables GPUs to detect and automatically correct single-bit memory errors. In our testing, we did not observe any noticeable performance or behavior differences with ECC mode enabled versus disabled.

### Caching

#### Warm Cache vs. Cold Cache

When profiling kernels, warm vs. cold cache states can meaningfully affect measurements, and failing to account for this can lead to misleading results. We define a warm cache as the required data already residing in L1/L2, and a cold cache as the opposite. With a cold cache, requests miss and fetch from global memory, incurring high latency.

To study this effect, we benchmark warm- and cold-cache performance for several common kernels on a single H100-SXM-5 GPU with 50MB of L2 cache. Each kernel is warmed up 100 times and then timed over 500 averaged runs. We find that**the gap between warm and cold cache timing varies across workloads, with some kernels showing more sensitivity than others**.

We rely on writing a dummy tensor sized to greater than the GPU's L2 cache to flush the cache. By deliberately writing a massive block of new data (like zeros) that fills the entire L2 capacity, we physically force the hardware to evict the residual data from our previous kernel run, ensuring our benchmarks measure authentic memory fetch latency rather than artificially inflated speeds from accidental cache hits.

#### Effects of cache state are workload dependent.

#### Vector Addition

Vector addition (VectorAdd) is the element-wise addition of two arrays, with output C, i.e., C[i] = A[i] + B[i]. For each thread i, the GPU performs 2 reads from A[i] and B[i], and 1 write to C[i], plus 1 floating point addition, resulting in a low arithmetic intensity (AI) = N / 3NB = 1 / 3B, where N is the number of elements and B is the byte size of data type. This low arithmetic intensity makes VectorAdd memory-bound on modern GPUs. As a result, performance is primarily determined by memory bandwidth and data locality, and cache state can influence timing.

The plots below show our benchmark results of VectorAdd on different sizes of vectors for FP16 and FP32. For cold cache experiments, we flush the cache before each kernel run. With a vector size of 10 million elements, the total memory footprint for both FP16 and FP32 operations exceeds the 50 MiB L2 cache capacity, and we see the performance gap between warm and cold cache narrowing.

![](images/image24.png)
![](images/image8.png)

#### Matrix Multiplication

For a classic matrix multiplication (GEMM), if the matrix size is N\*N, we have roughly O(N^3) FLOPs computation and approach O(N^2) memory complexity with sufficient data reuse, resulting in high AI. In the experiments shown below, the warm and cold caches yield similar timing results.

![](images/image14.png)
![](images/image5.png)

#### Matrix-Vector Multiplication

For matrix-vector multiplication (GEMV), the AI is effectively O(1). We observe a noticeable warm cache benefit on this problem. For larger sizes, the warm-cache benefit starts to diminish.

![](images/image19.png)
![](images/image26.png)

#### Stencil/Conv2D

We benchmarked a standard Conv2D configuration with 8 channels and a 3×3 kernel. We did not observe a significant difference between warm and cold cache measurements.

![](images/image4.png)
![](images/image7.png)

#### Strided MemoryAccess

At smaller strides, warm cache runs benefit from previously loaded lines and consistently outperform cold cache runs.

Note: For sizes starting from 256, the sudden performance drop is due to the design choice of PyTorch (we are benchmarking torch.sum here). When the stride is greater than 128, PyTorch clips the grid size from 16 to 1.

![](images/image23.png)
![](images/image28.png)

#### Microsecond-scale kernels

For small kernels with execution times on the order of microseconds, what we refer to as “microsecond-scale kernels", both cache flushing and the timing method itself can introduce noticeable uncertainty. We discuss this in more detail in the next section.

## Software Level Choices: Timers and Flags

### Not all timing methods are equal.

#### Nature of Asynchronous Execution

The CPU does not wait for the GPU to complete work before proceeding. Instead, it enqueues commands to the GPU, which then fetches and executes them asynchronously. As a result, GPU calls are non-blocking on the CPU side, creating a gap between when the CPU issues a command and when the GPU actually begins and finishes execution.

CPU timing measures elapsed time on the host, while GPU timing reflects execution based on the device’s internal clock. Using a CPU timer around a kernel call includes launch overhead, queueing, execution, and synchronization, making it unsuitable for isolating kernel execution time. GPU timing, by contrast, captures execution directly on the device timeline and avoids host-side overhead. CUDA events implement this mechanism by placing start and end markers directly in the GPU command stream, allowing precise measurement of kernel execution time.

#### Common Timing Methods (CUDA events, CUDA graphs, ...)

In this blog, we investigate several commonly used timing methods, including CUDA events, CUDA graph, PyTorch profiler, Triton’s timing function, and NVIDIA Nsight Compute CLI (ncu).

* **CUDA events** are markers inserted into the GPU command stream. By recording start and stop events, we can measure elapsed time with high precision using GPU-side timestamps. However, if the kernel is launched between those event records from the CPU, the measured interval may still include launch overhead.
* **CUDA graphs**: Using torch.cuda.Event in a loop still requires the CPU to launch each kernel, adding launch overhead. CUDA graphs instead capture a sequence of operations into a single executable graph, reducing CPU involvement and making execution more deterministic. CUDA graphs do incur some startup memory overhead, but this can be amortized by capturing multiple iterations and replaying a single graph. Graph capture itself, however, is relatively costly and reduces flexibility in CPU-side control flow.
* **PyTorch profiler** is built on the NVIDIA Kineto library and CUPTI, recording execution metadata on both CPU and GPU. It helps link the high-level Python code with low-level CUDA kernels and provides memory tracking. However, because it collects detailed metadata, the profiler is heavyweight and can introduce substantial overhead.
* **Triton** exposes timing via triton.testing.do\_bench, which is built on torch.cuda.Event and adds conveniences such as automated warmup and L2 cache flushing. However, because it relies on CUDA events, it inherits the same underlying limitations.
* **NVIDIA Nsight Compute (ncu)** intercepts kernel launches and collects detailed performance counters from the GPU’s microarchitecture, providing instruction-level metrics in a controlled environment. This enables deep hardware insight, but profiling is time-consuming and typically evaluates kernels in isolation. Because execution may be replayed and prior cache state is not reliably preserved, it is not well suited for controlling cache state behavior across repeated runs.
* **Graph Differential**: One way to amortize CUDA graph overhead is to record multiple executions in a single graph. However, this becomes tricky for cold-cache measurements, which require cache flushing. It is difficult to isolate how much of each iteration’s time is due to flushing. A natural idea is to place cache flushing in a separate loop and subtract its time from the combined cold-cache measurement. This corresponds to the “graph differential timing” method shown in the plots below. In practice, it can produce stable results that may better approximate pure execution time. However, because it relies on subtraction between two measurements, it can amplify errors in certain cases and should be interpreted with care.

#### Timing method choice determines what you think you measured.

In our timing experiments, we include warmup loops before measurement. For CUDA graphs, we record timings for each replay of a graph containing a single kernel execution. In the graph differential method, the warm-cache setup is similar, except that we amortize graph launch overhead by capturing multiple kernel runs in one graph. For the cold-cache case, we measure combined execution plus cache flushing and subtract the standalone cache-flush time to estimate kernel execution time.

**Based on our findings, we believe execution times under 10µs cannot be reliably measured, as they are increasingly dominated by system variance and the overhead of the timing method itself.**

Our results show that CUDA events, CUDA graphs, and Triton’s do\_bench (which is built on CUDA events) provide stable, low-variance measurements across kernel sizes and tasks, with timing results close to Nsight Compute. The PyTorch profiler is more heavyweight and can have more variance; for microsecond-scale kernels, it can even report zero timings. It is therefore better suited for bottleneck identification and latency-path analysis than precise microbenchmarking. As discussed earlier, the graph differential method relies on indirect subtraction and can amplify measurement errors, resulting in higher variance, particularly for microsecond-scale kernels.

For CUDA events and CUDA graphs, the latter can amortize launch overhead by capturing multiple runs in a single graph, making it theoretically more accurate since CUDA events may still include kernel launch overhead. However, for workloads with strong runtime dependencies, dynamic memory addresses, or branch divergence, CUDA graph results may deviate from real-world behavior. In addition, graph capture can be restrictive and inconvenient in such cases.

![](images/image31.png)
![](images/image18.png)

![](images/image29.png)
![](images/image16.png)

![](images/image3.png)
![](images/image12.png)

![](images/image17.png)
![](images/image11.png)

![](images/image30.png)
![](images/image22.png)

### Compilation flags can change performance.

Compilation flags alone can change performance even for the exact same kernel and hardware. Sensitivity to flags is workload-dependent—math-heavy kernels, reductions, and fused operators can react very differently to changes in optimization level, fast-math, register limits, or backend scheduling. If you benchmark kernels without clearly controlling and reporting compilation flags, the results can be misleading. In automated systems, inconsistent flags can easily look like algorithmic improvements when they are merely compiler effects.

![](images/image21.png)

## GPU Provider Variance: GPUs Aren’t Identical

### Getting the identical hardware is not guaranteed.

Getting identical hardware across cloud providers is not guaranteed. We requested A100 80GB GPUs from three different providers — anonymized as A, B, and C — and found that the instances were not even the same A100 variant. Provider A offered an A100-PCIe, provider B an A100-SXM4, and provider C a GRID A100D-80C running in a virtualized environment. These differences were not always clearly labeled in the provider listings. Beyond the SKU mismatch, the setups also varied in driver versions, CUDA versions, power caps, and clock configurations. Even when the name “A100 80GB” is the same, the underlying hardware and system configuration can differ in meaningful ways that impact performance and reproducibility.

Provider A:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          On  |   00000002:00:01.0 Off |                    0 |
| N/A   31C    P0             42W /  300W |       1MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
Clock Speed Limits:  1410 MHz, 1512 MHz
Persistence Mode:  Enabled
```

Provider B:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:05:00.0 Off |                  Off |
| N/A   33C    P0             58W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
Clock Speed Limits:  1410 MHz, 1593 MHz
Persistence Mode:  Enabled
```

Provider C:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  GRID A100D-80C                 On  |   00000000:06:00.0 Off |                    0 |
| N/A   N/A    P0             N/A /  N/A  |       1MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
Clock Speed Limits:  [N/A], [N/A]
Persistence Mode:  Enabled
```

We also tried a serverless GPU platform where we sent 4 job requests on A100-80GBs and received different A100-80GB variants: three PCIe GPUs and one SXM GPU.

### Cloud providers sometimes restrict access to hardware settings.

You may not be able to modify parameters like power limits or lock clock speeds, which limits your ability to control performance and reduce variance.

```
sudo nvidia-smi -pl 300
Changing power management limit is not supported in current scope for GPU: 00000000:06:00.0.
All done.
sudo nvidia-smi -rac
Setting applications clocks is not supported for GPU 00000000:06:00.0.
Treating as warning and moving on.
```

### Performance differs systematically across providers.

We benchmarked an 8192×8192×8192 matrix multiplication on providers A, B, and C. The runtimes cluster into distinct, non-overlapping ranges, suggesting consistent differences across providers. While the absolute gaps are not dramatic, the separation is systematic.

![](images/image9.png)

## Our Evaluation Infrastructure

We’ve been developing eval infrastructure in-house for RL workflows where measurements must be **accurate, consistent, and scalable** across many runs and kernels. Small timing errors or variance can compound quickly in iterative training loops, so correctness and stability are critical. We may share more details about this infrastructure as it evolves.

hi@standardkernel.com
[@standard\_kernel](https://twitter.com/Standard_Kernel)

© 2026 Standard Kernel Co.