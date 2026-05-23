[Algorithmica](/)
/
[HPC](/hpc/)

* Performance Engineering
* [Complexity Models](/hpc/complexity/)

1. [Modern Hardware](/hpc/complexity/hardware/)
2. [Programming Languages](/hpc/complexity/languages/)

* [Computer Architecture](/hpc/architecture/)

1. [Instruction Set Architectures](/hpc/architecture/isa/)
2. [Assembly Language](/hpc/architecture/assembly/)
3. [Loops and Conditionals](/hpc/architecture/loops/)
4. [Functions and Recursion](/hpc/architecture/functions/)
5. [Indirect Branching](/hpc/architecture/indirect/)
6. [Machine Code Layout](/hpc/architecture/layout/)

* [Instruction-Level Parallelism](/hpc/pipelining/)

1. [Pipeline Hazards](/hpc/pipelining/hazards/)
2. [The Cost of Branching](/hpc/pipelining/branching/)
3. [Branchless Programming](/hpc/pipelining/branchless/)
4. [Instruction Tables](/hpc/pipelining/tables/)
5. [Throughput Computing](/hpc/pipelining/throughput/)

* [Compilation](/hpc/compilation/)

1. [Stages of Compilation](/hpc/compilation/stages/)
2. [Flags and Targets](/hpc/compilation/flags/)
3. [Situational Optimizations](/hpc/compilation/situational/)
4. [Contract Programming](/hpc/compilation/contracts/)
5. [Precomputation](/hpc/compilation/precalc/)

* [Profiling](/hpc/profiling/)

1. [Instrumentation](/hpc/profiling/instrumentation/)
2. [Statistical Profiling](/hpc/profiling/events/)
3. [Program Simulation](/hpc/profiling/simulation/)
4. [Machine Code Analyzers](/hpc/profiling/mca/)
5. [Benchmarking](/hpc/profiling/benchmarking/)
6. [Getting Accurate Results](/hpc/profiling/noise/)

* [Arithmetic](/hpc/arithmetic/)

1. [Floating-Point Numbers](/hpc/arithmetic/float/)
2. [IEEE 754 Floats](/hpc/arithmetic/ieee-754/)
3. [Rounding Errors](/hpc/arithmetic/errors/)
4. [Newton's Method](/hpc/arithmetic/newton/)
5. [Fast Inverse Square Root](/hpc/arithmetic/rsqrt/)
6. [Integer Numbers](/hpc/arithmetic/integer/)
7. [Integer Division](/hpc/arithmetic/division/)

* [Number Theory](/hpc/number-theory/)

1. [Modular Arithmetic](/hpc/number-theory/modular/)
2. [Binary Exponentiation](/hpc/number-theory/exponentiation/)
3. [Extended Euclidean Algorithm](/hpc/number-theory/euclid-extended/)
4. [Montgomery Multiplication](/hpc/number-theory/montgomery/)

* [External Memory](/hpc/external-memory/)

1. [Memory Hierarchy](/hpc/external-memory/hierarchy/)
2. [Virtual Memory](/hpc/external-memory/virtual/)
3. [External Memory Model](/hpc/external-memory/model/)
4. [External Sorting](/hpc/external-memory/sorting/)
5. [List Ranking](/hpc/external-memory/list-ranking/)
6. [Eviction Policies](/hpc/external-memory/policies/)
7. [Cache-Oblivious Algorithms](/hpc/external-memory/oblivious/)
8. [Spatial and Temporal Locality](/hpc/external-memory/locality/)

* [RAM & CPU Caches](/hpc/cpu-cache/)

1. [Memory Bandwidth](/hpc/cpu-cache/bandwidth/)
2. [Memory Latency](/hpc/cpu-cache/latency/)
3. [Cache Lines](/hpc/cpu-cache/cache-lines/)
4. [Memory Sharing](/hpc/cpu-cache/sharing/)
5. [Memory-Level Parallelism](/hpc/cpu-cache/mlp/)
6. [Prefetching](/hpc/cpu-cache/prefetching/)
7. [Alignment and Packing](/hpc/cpu-cache/alignment/)
8. [Pointer Alternatives](/hpc/cpu-cache/pointers/)
9. [Cache Associativity](/hpc/cpu-cache/associativity/)
10. [Memory Paging](/hpc/cpu-cache/paging/)
11. [AoS and SoA](/hpc/cpu-cache/aos-soa/)

* [SIMD Parallelism](/hpc/simd/)

1. [Intrinsics and Vector Types](/hpc/simd/intrinsics/)
2. [Moving Data](/hpc/simd/moving/)
3. [Reductions](/hpc/simd/reduction/)
4. [Masking and Blending](/hpc/simd/masking/)
5. [In-Register Shuffles](/hpc/simd/shuffling/)
6. [Auto-Vectorization and SPMD](/hpc/simd/auto-vectorization/)

* [Algorithms Case Studies](/hpc/algorithms/)

1. [Binary GCD](/hpc/algorithms/gcd/)
2. [Integer Factorization](/hpc/algorithms/factorization/)
3. [Argmin with SIMD](/hpc/algorithms/argmin/)
4. [Prefix Sum with SIMD](/hpc/algorithms/prefix/)
5. [Matrix Multiplication](/hpc/algorithms/matmul/)

* [Data Structures Case Studies](/hpc/data-structures/)

1. [Binary Search](/hpc/data-structures/binary-search/)
2. [Static B-Trees](/hpc/data-structures/s-tree/)
3. [Search Trees](/hpc/data-structures/b-tree/)
4. [Segment Trees](/hpc/data-structures/segment-trees/)

![](/icons/bars-solid.svg "open table of contents")
![](/icons/adjust-solid.svg "dark theme")
![](/icons/search-solid.svg "search")

Computer Architecture

![](/icons/print-solid.svg "print")
[![](/icons/edit-solid.svg "edit")](https://prose.io/#algorithmica-org/algorithmica/edit/master/content%2fenglish/hpc%2farchitecture%2f_index.md)
[![](/icons/github-brands.svg "view on github")](https://github.com/algorithmica-org/algorithmica/blob/master/content/english/hpc/architecture/_index.md)

# Computer Architecture

When I began learning how to optimize programs myself, one big mistake I made was to rely primarily on the empirical approach. Not understanding how computers really worked, I would semi-randomly swap nested loops, rearrange arithmetic, combine branch conditions, inline functions by hand, and follow all sorts of other performance tips I’ve heard from other people, blindly hoping for improvement.

Unfortunately, this is how most programmers approach optimization. Most texts about performance do not teach you to reason about software performance qualitatively. Instead they give you general advice about certain implementation approaches — and general performance intuition is clearly not enough.

It would have probably saved me dozens, if not hundreds of hours if I learned computer architecture *before* doing algorithmic programming. So, even if most people aren’t *excited* about it, we are going to spend the first few chapters studying how CPUs work and start with learning assembly.

[← Programming Languages](https://en.algorithmica.org/hpc/complexity/languages/)
[← ../Complexity Models](https://en.algorithmica.org/hpc/complexity/)

[Instruction Set Architectures →](https://en.algorithmica.org/hpc/architecture/isa/)
[../Instruction-Level Parallelism →](https://en.algorithmica.org/hpc/pipelining/)

Copyright 2021–2022 Sergey Slotin