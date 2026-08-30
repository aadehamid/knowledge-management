![Cornell University](/icons/cornell-reduced-red.svg)

Cornell Virtual Workshop

[CVW Search Agent](/ChatSearch)

Cornell Virtual Workshop

* [Home](/index)
* [Topics](/topics)
* [Roadmaps](/roadmaps)
* [Notebook](/notebook)
* [Glossary](/glossary)
* [Contact](/contact)
* [Search](/chatsearch)
* [Login with Globus](/oauth2/index.php?from=gpu-architecture/gpu-memory/memory_types)

[###### **Understanding GPU Architecture**](/gpu-architecture)

[GPU Characteristics](/gpu-architecture/gpu-characteristics/index)

[**GPU Memory**](/gpu-architecture/gpu-memory/index)

* [Memory Levels](/gpu-architecture/gpu-memory/memory_levels)
* [Memory Types](/gpu-architecture/gpu-memory/memory_types)
* [Comparison to CPU Memory](/gpu-architecture/gpu-memory/comparison_cpu_mem)
* [Appendix: Finer Memory Slices](/gpu-architecture/gpu-memory/appendix)

[Horizon GPUs: Blackwell B200](/gpu-architecture/horizon-gpus-blackwell-b200/index)

[Frontera GPUs: RTX 5000](/gpu-architecture/frontera-gpus-rtx-5000/index)

[GPU Example: Tesla V100](/gpu-architecture/gpu-example-tesla-v100/index)

[Exercises](/gpu-architecture/gpu-exercises/index)

[Quiz](/gpu-architecture/extras/quiz?back=/gpu-architecture/gpu-memory/memory_types)

### Memory Types

[Understanding GPU Architecture](/gpu-architecture)
>
[GPU Memory](/gpu-architecture/gpu-memory/index)
>
Memory Types

The picture on the preceding page is more complex than it would be for a CPU, because the GPU reserves certain areas of memory for specialized use
during rendering. Here, we summarize the roles of each type of GPU memory for doing GPGPU computations.

The first list covers the on-chip memory areas that are closest to the CUDA cores.
They are part of every SM.

* **Register File** - denotes the area of memory that feeds directly into the CUDA cores. Accordingly, it is organized into 32 *banks*,
  matching the 32 threads in a warp. Think of the register
  file as a big matrix of 4-byte elements, having many rows and 32 columns. A warp operates on full rows; within a given row, each thread
  (CUDA core) operates on a different column (bank).
* **L1 Cache** - refers to the usual on-chip storage location providing fast access to data that are recently read from, or written to, main
  memory (RAM). Additionally, L1 serves as the overflow region when the amount of active data exceeeds what an SM's register file can hold, a condition
  which is termed "register spilling". In L1, the cache lines and spilled registers are organized into banks, just as in the register file.
* **Shared Memory** - is a memory area that physically resides in the same memory as the L1 cache, but differs from L1 in that all its data may
  be accessed by any thread in a thread block. This allows threads to communicate and share data with each other. Variables that occupy it must be
  declared explicitly by an application. The application can also set the dividing line between L1 and shared memory.
* **Constant Caches** - are special caches pertaining to variables declared as read-only constants in global memory. Such variables can be read by
  any thread in a thread block. The main and best use of these caches is to broadcast a single constant value to all the threads in a warp.

The second list pertains to the more distant, larger memory areas available to all the SMs.

* **L2 Cache** - is a further on-chip cache for retaining copies of the data that travel back and forth between the SMs and main memory. Like the
  L1, the L2 cache is intended to speed up subsequent reloads. But unlike the L1 cache(s), there is just one L2 that is shared by all the SMs. The L2
  cache is also situated in the path of data moving on or off the device via PCIe or NVLink.
* **Global Memory** - represents the bulk of the main memory of the device, equivalent to RAM in a CPU-based processor. For performance reasons,
  the Blackwell B200 and the Tesla V100 have special HBM high-bandwidth memory, while the Quadro RTX 5000 has fast GDDR6 graphics memory.
* **Texture and Constant Memory** - are regions of main memory that are treated as read-only by the device. When fetched to an SM, variables with
  a "texture" or "constant" declaration can be read by any thread in a kernel, serving as an expanded type of shared memory. Texture memory is cached in L1, while
  constant memory is cached in the constant caches.
* **Local Memory** - corresponds to specially mapped regions of main memory that are assigned to each SM. Whenever "register spilling" overflows
  the L1 cache on a particular SM, the excess data are further offloaded to L2, then to "local memory". The performance penalty for reloading a
  spilled register becomes steeper for every memory level that must be traversed in order to retrieve it.

Note from the second list that the L2 cache may hold two types of data: data that are accessible to any SM (global, texture, and constant memory),
and data that are accessible just to each SM individually (local memory).

* For global data, consistency is maintained entirely through L2. Whenever an SM does a write to global data, the L1 cache line having the data is
  immediately evicted to L2 ("write-through" policy). Because of this strategy, a GPU does not need to have a complex interconnect to maintain
  L1 cache coherence, like a CPU does.
  Everything happens through L2.
* In contrast, when an SM writes to its local memory, the eviction of the L1 cache line to L2 may be delayed ("write-back" policy), because the reserved
  area within L2 is not shared with other SMs.

[Back](/gpu-architecture/gpu-memory/memory_levels)

[Next](/gpu-architecture/gpu-memory/comparison_cpu_mem)

©
 |
[Cornell University](https://www.cornell.edu)
   |
[Center for Advanced Computing](https://www.cac.cornell.edu/)
   |
[Copyright Statement](https://www.cac.cornell.edu/copyright.aspx)
   |
[Access Statement](/AccessStatement)

CVW material development is supported by NSF OAC awards 1854828, 2321040, 2323116 (UT Austin) and 2005506 (Indiana University)

[![real time web analytics](https://c.statcounter.com/4742968/0/9445d005/1/)](https://statcounter.com/ "real time web
analytics")