[![Home](https://developer-blogs.nvidia.com/wp-content/themes/nvidia/dist/images/nvidia-logo_28b633c7.svg)![Home](data:image/svg+xml...)](/ "Home")
[DEVELOPER](/ "Home")

* [Home](/ "Home")
* [Blog](/blog "Blog")
* [Forums](https://forums.developer.nvidia.com/ "Forums")
* [Docs](https://docs.nvidia.com/ "Docs")
* [Downloads](https://developer.nvidia.com/downloads "Downloads")
* [Training](https://www.nvidia.com/en-us/training/ "Training")

* [Join](https://developer.nvidia.com/login)

[Technical Blog](https://developer.nvidia.com/blog)

[Subscribe](https://developer.nvidia.com/email-signup)

[Related Resources](#main-content-end)

Models / Libraries / Frameworks

English中文

# Using Shared Memory in CUDA C/C++

![](https://developer-blogs.nvidia.com/wp-content/uploads/2012/10/CUDA_Cube_1K-e1753800226468-1024x577.webp)

Jan 28, 2013

By [Mark Harris](https://developer.nvidia.com/blog/author/mharris/ "Posts by Mark Harris")

Like

[Discuss (36)](#entry-content-comments)

* [L](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [T](https://twitter.com/intent/tweet?text=Using+Shared+Memory+in+CUDA+C%2FC%2B%2B+%7C+NVIDIA+Technical+Blog+https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [F](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [R](https://www.reddit.com/submit?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F&title=Using+Shared+Memory+in+CUDA+C%2FC%2B%2B+%7C+NVIDIA+Technical+Blog)
* E

## AI-Generated Summary

* Shared memory provides on-chip storage with roughly 100x lower latency than uncached global memory, enabling faster data access for threads within the same thread block.
* The [array reversal example](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu) demonstrates how shared memory and \_\_syncthreads() enable coalesced global memory access patterns on any CUDA GPU.
* Static shared memory uses fixed-size \_\_shared\_\_ arrays declared at compile time, while dynamic shared memory uses extern \_\_shared\_\_ arrays sized at kernel launch via a third execution configuration parameter.
* Shared memory is organized into banks; concurrent accesses to different banks achieve high bandwidth, but multiple threads accessing the same bank serialize requests and reduce effective bandwidth.
* Devices of compute capability 2.x and 3.x allow runtime configuration of the on-chip memory partition between shared memory and L1 cache using cudaDeviceSetCacheConfig() or cudaFuncSetCacheConfig().

### Next Steps

* Explore the [shared memory code sample on GitHub](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu) to see complete working examples.
* Review the [cudaDeviceSetSharedMemConfig documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__DEVICE_1g1a4789fb687cc36dccc98f25c96f0cd8) for configuring bank size on compute capability 3.x devices.
* Read the [cudaDeviceSetCacheConfig documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__DEVICE_1gac27b566beee1aa9175373bb9e29b8d1) to learn how to partition shared memory and L1 cache on compute capability 2.x and 3.x devices.

Powered by NVIDIA Nemotron. AI-generated content may summarize information incompletely. Verify important information. [Learn more](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)

In the [previous post](https://developer.nvidia.com/blog/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/ "How to Access Global Memory Efficiently in CUDA C/C++ Kernels"), I looked at how global memory accesses by a group of threads can be coalesced into a single transaction, and how alignment and stride affect coalescing for various generations of CUDA hardware. For recent versions of CUDA hardware, misaligned data accesses are not a big issue. However, striding through global memory is problematic regardless of the generation of the CUDA hardware, and would seem to be unavoidable in many cases, such as when accessing elements in a multidimensional array along the second and higher dimensions. However, it is possible to coalesce memory access in such cases if we use shared memory. Before I show you how to avoid striding through global memory in the next post, first I need to describe shared memory in some detail.

# Shared Memory

Because it is on-chip, shared memory is much faster than local and global memory. In fact, shared memory latency is roughly 100x lower than uncached global memory latency (provided that there are no bank conflicts between the threads, which we will examine later in this post). Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory. Threads can access data in shared memory loaded from global memory by other threads within the same thread block. This capability (combined with thread synchronization) has a number of uses, such as user-managed data caches, high-performance cooperative parallel algorithms (parallel reductions, for example), and to facilitate global memory coalescing in cases where it would otherwise not be possible.

## Thread Synchronization

When sharing data between threads, we need to be careful to avoid race conditions, because while threads in a block run *logically* in parallel, not all threads can execute *physically* at the same time. Let’s say that two threads A and B each load a data element from global memory and store it to shared memory. Then, thread A wants to read B’s element from shared memory, and vice versa. Let’s assume that A and B are threads in two different warps. If B has not finished writing its element before A tries to read it, we have a race condition, which can lead to undefined behavior and incorrect results.

To ensure correct results when parallel threads cooperate, we must synchronize the threads. CUDA provides a simple barrier synchronization primitive, `__syncthreads()`. A thread’s execution can only proceed past a `__syncthreads()` after all threads in its block have executed the `__syncthreads()`. Thus, we can avoid the race condition described above by calling `__syncthreads()` after the store to shared memory and before any threads load from shared memory. It’s important to be aware that calling `__syncthreads()` in divergent code is undefined and can lead to deadlock—all threads within a thread block must call `__syncthreads()` at the same point.

## Shared Memory Example

Declare shared memory in CUDA C/C++ device code using the `__shared__` variable declaration specifier. There are multiple ways to declare shared memory inside a kernel, depending on whether the amount of memory is known at compile time or at run time. The following complete code ([available on GitHub](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu)) illustrates various methods of using shared memory.

```
#include

__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int));

  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
```

This code reverses the data in a 64-element array using shared memory. The two kernels are very similar, differing only in how the shared memory arrays are declared and how the kernels are invoked.

## Static Shared Memory

If the shared memory array size is known at compile time, as in the staticReverse kernel, then we can explicitly declare an array of that size, as we do with the array `s`.

```
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```

In this kernel, `t` and `tr` are the two indices representing the original and reverse order, respectively. Threads copy the data from global memory to shared memory with the statement `s[t] = d[t]`, and the reversal is done two lines later with the statement `d[t] = s[tr]`. But before executing this final line in which each thread accesses data in shared memory that was written by another thread, remember that we need to make sure all threads have completed the loads to shared memory, by calling `__syncthreads()`.

The reason shared memory is used in this example is to facilitate global memory coalescing on older CUDA devices (Compute Capability 1.1 or earlier). Optimal global memory coalescing is achieved for both reads and writes because global memory is always accessed through the linear, aligned index `t`. The reversed index `tr` is only used to access shared memory, which does not have the sequential access restrictions of global memory for optimal performance. The only performance issue with shared memory is bank conflicts, which we will discuss later. (Note that on devices of Compute Capability 1.2 or later, the memory system can fully coalesce even the reversed index stores to global memory. But this technique is still useful for other access patterns, as I’ll show in the next post.)

## Dynamic Shared Memory

The other three kernels in this example use dynamically allocated shared memory, which can be used when the amount of shared memory is not known at compile time. In this case the shared memory allocation size per thread block must be specified (in bytes) using an optional third execution configuration parameter, as in the following excerpt.

```
dynamicReverse<<<1, n, n*sizeof(int)>>>(d_d, n);
```

The dynamic shared memory kernel, `dynamicReverse()`, declares the shared memory array using an unsized extern array syntax, `extern shared int s[]` (note the empty brackets and use of the extern specifier). The size is implicitly determined from the third execution configuration parameter when the kernel is launched. The remainder of the kernel code is identical to the `staticReverse()` kernel.

What if you need multiple dynamically sized arrays in a single kernel? You must declare a single `extern` unsized array as before, and use pointers into it to divide it into multiple arrays, as in the following excerpt.

```
extern __shared__ int s[];
int *integerData = s;                        // nI ints
float *floatData = (float*)&integerData[nI]; // nF floats
char *charData = (char*)&floatData[nF];      // nC chars
```

In the kernel launch, specify the total shared memory needed, as in the following.

```
myKernel<<<gridSize, blockSize, nI*sizeof(int)+nF*sizeof(float)+nC*sizeof(char)>>>(...);
```

# Shared memory bank conflicts

To achieve high memory bandwidth for concurrent accesses, shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously. Therefore, any memory load or store of *n* addresses that spans *b* distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is *b* times as high as the bandwidth of a single bank.

However, if multiple threads’ requested addresses map to the same memory bank, the accesses are serialized. The hardware splits a conflicting memory request into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of colliding memory requests. An exception is the case where all threads in a warp address the same shared memory address, resulting in a broadcast. Devices of compute capability 2.0 and higher have the additional ability to multicast shared memory accesses, meaning that multiple accesses to the same location by any number of threads within a warp are served simultaneously.

To minimize bank conflicts, it is important to understand how memory addresses map to memory banks. Shared memory banks are organized such that successive 32-bit words are assigned to successive banks and the bandwidth is 32 bits per bank per clock cycle. For devices of compute capability 1.x, the warp size is 32 threads and the number of banks is 16. A shared memory request for a warp is split into one request for the first half of the warp and one request for the second half of the warp. Note that no bank conflict occurs if only one memory location per bank is accessed by a half warp of threads.

For devices of compute capability 2.0, the warp size is 32 threads and the number of banks is also 32. A shared memory request for a warp is not split as with devices of compute capability 1.x, meaning that bank conflicts can occur between threads in the first half of a warp and threads in the second half of the same warp.

Devices of compute capability 3.x have configurable bank size, which can be set using [cudaDeviceSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__DEVICE_1g1a4789fb687cc36dccc98f25c96f0cd8)() to either four bytes (cudaSharedMemBankSizeFourByte, the default) or eight bytes (`cudaSharedMemBankSizeEightByte)`. Setting the bank size to eight bytes can help avoid shared memory bank conflicts when accessing double precision data.

# Configuring the amount of shared memory

On devices of compute capability 2.x and 3.x, each multiprocessor has 64KB of on-chip memory that can be partitioned between L1 cache and shared memory. For devices of compute capability 2.x, there are two settings, 48KB shared memory / 16KB L1 cache, and 16KB shared memory / 48KB L1 cache. By default the 48KB shared memory setting is used. This can be configured during runtime API from the host for all kernels using `[cudaDeviceSetCacheConfig](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__DEVICE_1gac27b566beee1aa9175373bb9e29b8d1)()` or on a per-kernel basis using `[cudaFuncSetCacheConfig](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__EXECUTION_1g4f35d04be20a41c5df96613a748eecc1)()`. These accept one of three options: `cudaFuncCachePreferNone`, `cudaFuncCachePreferShared`, and `cudaFuncCachePreferL1`. The driver will honor the specified preference except when a kernel requires more shared memory per thread block than available in the specified configuration. Devices of compute capability 3.x allow a third setting of 32KB shared memory / 32KB L1 cache which can be obtained using the option `cudaFuncCachePreferEqual`.

# Summary

Shared memory is a powerful feature for writing well optimized CUDA code. Access to shared memory is much faster than global memory access because it is located on chip. Because shared memory is shared by threads in a thread block, it provides a mechanism for threads to cooperate. One way to use shared memory that leverages such thread cooperation is to enable global memory coalescing, as demonstrated by the array reversal in this post. By reversing the array using shared memory we are able to have all global memory reads and writes performed with unit stride, achieving full coalescing on any CUDA GPU. In the next post I will continue our discussion of shared memory by using it to optimize a matrix transpose.

[Discuss (36)](#entry-content-comments)

Like

## Tags

[AR / VR](https://developer.nvidia.com/blog/category/virtual-reality/) | [HPC / Scientific Computing](https://developer.nvidia.com/blog/recent-posts/?industry=HPC+%2F+Scientific+Computing) | [CUDA](https://developer.nvidia.com/blog/recent-posts/?products=CUDA) | [Intermediate Technical](https://developer.nvidia.com/blog/recent-posts/?learning_levels=Intermediate+Technical) | [CUDA C/C++](https://developer.nvidia.com/blog/tag/cuda-cc/) | [Memory](https://developer.nvidia.com/blog/tag/memory/) | [Shared Memory](https://developer.nvidia.com/blog/tag/shared-memory/)

## About the Authors

![](https://secure.gravatar.com/avatar/005c01a1c744d89a3bce53af5436068bdb22abd9b750fa5108fc5a6a86da1b21?s=131&d=retro&r=g)![](data:image/svg+xml...)

**About Mark Harris**

Mark is an NVIDIA Distinguished Engineer working on [RAPIDS](https://rapids.ai). Mark has over twenty years of experience developing software for GPUs, ranging from graphics and games, to physically-based simulation, to parallel algorithms and high-performance computing. While a Ph.D. student at The University of North Carolina he recognized a nascent trend and coined a name for it: GPGPU (General-Purpose computing on Graphics Processing Units).

[Follow @harrism on Twitter](https://twitter.com/intent/user?screen_name=harrism)

[View all posts by Mark Harris](https://developer.nvidia.com/blog/author/mharris/)

## Comments

## Related posts

![](https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/Cooperative-Groups-Featured.png)![](data:image/svg+xml...)

### Flexible CUDA Thread Programming

[Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/flexible-cuda-thread-programming/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2012/10/CUDA_Cube_1K-e1753800226468-660x370.jpg)![](data:image/svg+xml...)

### An Efficient Matrix Transpose in CUDA C/C++

[An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2012/12/cuda_fortran_simple.gif)![](data:image/svg+xml...)

### Using Shared Memory in CUDA Fortran

[Using Shared Memory in CUDA Fortran](https://developer.nvidia.com/blog/using-shared-memory-cuda-fortran/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2012/10/CUDA_Cube_1K-e1753800226468-660x370.jpg)![](data:image/svg+xml...)

### How to Access Global Memory Efficiently in CUDA C/C++ Kernels

[How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2012/12/cuda_fortran_simple.gif)![](data:image/svg+xml...)

### How to Access Global Memory Efficiently in CUDA Fortran Kernels

[How to Access Global Memory Efficiently in CUDA Fortran Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-fortran-kernels/)

## Related posts

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/DatacenterKV-2-660x370.png)![](data:image/svg+xml...)

### Just Released: cuDSS 0.3.0

[Just Released: cuDSS 0.3.0](https://nvda.ws/3VZLsmf#new_tab)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/CUDA-Blog-Image-1000x600-1.jpg)![](data:image/svg+xml...)

### Boosting Application Performance with GPU Memory Prefetching

[Boosting Application Performance with GPU Memory Prefetching](https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2020/09/cuda-featured.png)![](data:image/svg+xml...)

### Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture

[Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2020/04/image1-1.png)![](data:image/svg+xml...)

### Introducing Low-Level GPU Virtual Memory Management

[Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)

![GPU Pro Tip](https://developer-blogs.nvidia.com/wp-content/uploads/2017/02/GPU-Pro-Tip-e1753800348480.webp)![GPU Pro Tip](data:image/svg+xml...)

### GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell

[GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)

* [L](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [T](https://twitter.com/intent/tweet?text=Using+Shared+Memory+in+CUDA+C%2FC%2B%2B+%7C+NVIDIA+Technical+Blog+https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [F](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F)
* [R](https://www.reddit.com/submit?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fusing-shared-memory-cuda-cc%2F&title=Using+Shared+Memory+in+CUDA+C%2FC%2B%2B+%7C+NVIDIA+Technical+Blog)
* E