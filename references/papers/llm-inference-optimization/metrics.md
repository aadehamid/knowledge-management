[Skip to main content](#main-content)

![country_code](https://www.nvidia.com/content/dam/1x1-00000000.png)
Back to top

`Ctrl`+`K`

[![NVIDIA NIM LLMs Benchmarking - Home](_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)](index.html)

Choose version

Search
`Ctrl`+`K`

Search
`Ctrl`+`K`

[![NVIDIA NIM LLMs Benchmarking - Home](_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)
![NVIDIA NIM LLMs Benchmarking - Home](_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

NVIDIA NIM LLMs Benchmarking](index.html)

Choose version

Table of Contents

Benchmarking Guide

* [Overview](overview.html)
* Metrics
* [Parameters and Best Practices](parameters.html)
* [Using AIPerf to Benchmark](step-by-step.html)
* [Benchmarking LoRA Models](benchmarking-lora.html)

Performance

* [Benchmarks](performance.html)

* Metrics

[Is this page helpful?](https://surveys.hotjar.com/4904bf71-6484-47a7-83ff-4715cceabdb5)

# Metrics[#](#metrics "Link to this heading")

This section describes some of the common LLM inference metrics. Note that there can be variations in the benchmarking results between different tools. The following diagram illustrates some of the widely used LLM inference metrics.

[![_images/image3.png](_images/image3.png)](_images/image3.png)

Figure 1. Overview of popular LLM inference performance metrics.[#](#id1 "Link to this image")

## Time to First Token (TTFT)[#](#time-to-first-token-ttft "Link to this heading")

This metric shows how long a user needs to wait before seeing the model’s output. This is the time it takes from submitting the query to receiving the first token (if the response is not empty).

[![_images/image7.png](_images/image7.png)](_images/image7.png)

Figure 2: TTFT - Time to First Token including both the tokenization and de-tokenization steps for the first output token.[#](#id2 "Link to this image")

Note

Both NVIDIA GenAI-Perf and LLMPerf benchmarking tools disregard the initial responses that have no content or a content with empty string (no token present). This is because the TTFT measurement is meaningless when the first response has no token in it.

Time to first token generally includes both request queuing time, prefill time and network latency. The longer the prompt, the larger the TTFT. This is because the attention mechanism requires the whole input sequence to compute and create the so-called key-value cache (aka.[KV-cache](https://medium.com/%40joaolages/kv-caching-explained-276520203249)), from which point the iterative generation loop can begin. Additionally, a production application can have several requests in progress, therefore one request’s prefill phase may overlap with another request’s generation phase.

Note

Traditional web service benchmarking tools such as K6 can also provide TTFT, via timing events in the HTTP request.

## End-to-End Request Latency (e2e\_latency)[#](#end-to-end-request-latency-e2e-latency "Link to this heading")

This metric indicates how long it takes from submitting a query to receiving the full response, including the performance of your queueing/batching mechanisms and network latencies, as demonstrated in Figure 3.

[![_images/image8.png](_images/image8.png)](_images/image8.png)

Figure 3. End-to-end Request latency[#](#id3 "Link to this image")

Note

In streaming mode, the de-tokenization step can be done multiple times when partial results are returned to the user.

For an individual request, the end-to-end request latency is the time difference between the request sent and the final token received. Therefore:

\[e2e\\_latency = TTFT + Generation\\_time\]

Note

Generation\_time is the duration from the first token received to the final token received, as depicted in Figure 1. In addition, GenAI-Perf removes the last [done] signal or empty response, so they don’t get included in the e2e latency.

## Inter-token Latency (ITL)[#](#inter-token-latency-itl "Link to this heading")

This is defined as the average time between consecutive tokens and is also known as time per output token (TPOT).

[![_images/image4.png](_images/image4.png)](_images/image4.png)

[![_images/image9.png](_images/image9.png)](_images/image9.png)

Figure 4: ITL - latency between successive token generations.[#](#id4 "Link to this image")

Although this seems a simple and straightforward definition, there are some intricate decisions in which the different benchmarking tools take into account when collecting the metric. Questions such as, should this average calculation include the time to first token (TTFT) or not? NVIDIA genAI-perf does not, while LLMPerf does include this quantity.

GenAI-Perf defines ITL as follows:

\[\frac{e2e\\_ latency\ - \ TTFT}{Total\\_ output\\_ tokens\ - \ 1}\]

The equation used for this metric does not include the first token (hence subtracts 1 in the denominator). This is done in order to have ITL as a characteristic of only the decoding part of the request processing.

It is important to note that with longer output sequences, the KV cache grows and hence the memory cost. The cost of attention computation also grows: for each new token, this cost is linear in the length of the input + output sequence so far (but this computation is generally not compute-bound). Consistent inter-token latencies signifies an efficient memory management, better memory bandwidth as well as efficient attention computation.

## Tokens Per Second (TPS)[#](#tokens-per-second-tps "Link to this heading")

**Total TPS per system** represents the total output tokens per seconds throughput, accounting for all the requests happening simultaneously. As the number of requests increases, the total TPS per system increases, until it reaches a saturation point for all the available GPU compute resources, beyond which it might decrease.

Given the following timeline of the entire benchmark with **n** total requests.

![_images/image10.png](_images/image10.png)

Figure 5: Timeline of events in a benchmarking run[#](#id5 "Link to this image")

where

> * Li : End-to-end latency of i-th request
> * T\_start : start of benchmark
> * Tx : timestamp of the first request
> * Ty : timestamp of the last response of the last request
> * T\_end : end of benchmark

GenAI-perf defines the TPS as total output tokens divided by the end-to-end latency between the **first request** and the **last response** of the last request.

\[\frac{Total\\_ output\\_ tokens}{Ty\ - \ Tx}\]

Note that LLM-perf defines TPS as the total output tokens divided by the **entire benchmark duration**.

\[\frac{Total\\_ output\\_ tokens}{T\_{end}\ - \ T\_{start}}\]

As such, it also includes the following overheads into the metric: (1) Input prompt generation; (2) Request preparation and (3) Storing the responses. In our observation, these overheads in the single concurrency scenario can sometimes account for 33% of the entire benchmark duration.

Note that the previous calculation is done in a batch fashion and is not a live running metric. In addition, GenAI-perf uses a sliding window technique to find stable measurements. This means that the given measurements are from a representative subset of the fully-completed requests, meaning, the “warming up” and “cooling down” requests are not included when calculating the metrics.

**TPS per user** represents throughput from a single user perspective, and defined as (Output sequence length)/(e2e\_latency) for each user’s request, which asymptotically approaches 1/ITL as the output sequence length increases. Note that as the number of concurrent requests increases in the system, the total TPS for the whole system increases, while TPS per user decreases as latency becomes worse.

## Requests Per Second (RPS)[#](#requests-per-second-rps "Link to this heading")

This is the average number of requests that can be successfully completed by the system in a 1-second period. It is calculated as:

\[RPS\ = \ \frac{total\\_ completed\\_ requests}{Ty\ - \ Tx}\]

[previous

Overview](overview.html "previous page")
[next

Parameters and Best Practices](parameters.html "next page")

On this page

* [Time to First Token (TTFT)](#time-to-first-token-ttft)
* [End-to-End Request Latency (e2e\_latency)](#end-to-end-request-latency-e2e-latency)
* [Inter-token Latency (ITL)](#inter-token-latency-itl)
* [Tokens Per Second (TPS)](#tokens-per-second-tps)
* [Requests Per Second (RPS)](#requests-per-second-rps)

[![NVIDIA](_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg)
![NVIDIA](_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/)
|
[Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/)
|
[Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/)
|
[Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/)
|
[Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/)
|
[Product Security](https://www.nvidia.com/en-us/product-security/)
|
[Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2024-2026, NVIDIA Corporation.

Last updated on Apr 01, 2026.