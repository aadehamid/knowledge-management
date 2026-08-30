* Product

  + [**Bento Inference Platform**

    Full control without the complexity. Self-host anywhere. Serve any model. Optimize for performance.

    Book a Demo](https://www.bentoml.com/contact)
  + [**BentoML Open-Source**

    The most flexible way to serve AI/ML models and custom inference pipelines in production

    GitHub](https://github.com/bentoml)
* [Pricing](https://www.modular.com/pricing?utm_source=bentoml_pricing)
* [Docs](https://docs.bentoml.com/)
* Learn

  + [Blog](/blog)
  + [LLM Inference Handbook](/llm)
  + [LLM Performance Explorer](/llm-perf)
  + [Featured Examples](https://docs.bentoml.com/en/latest/examples/overview.html)
* [Customers](/customers)

[Log In](https://cloud.bentoml.com)

[Sign UpSign Up](https://console.modular.com/signup?utm_source=bentoml_signup)

* Product

  + [Bento Inference Platform](https://www.bentoml.com/contact)
  + [BentoML Open-Source](https://github.com/bentoml)
* [Pricing](https://www.modular.com/pricing?utm_source=bentoml_pricing)
* [Docs](https://docs.bentoml.com/)
* Learn

  + [Blog](/blog)
  + [LLM Inference Handbook](/llm)
  + [LLM Performance Explorer](/llm-perf)
  + [Featured Examples](https://docs.bentoml.com/en/latest/examples/overview.html)
* [Customers](/customers)

[Log In](https://cloud.bentoml.com)[Sign Up](https://console.modular.com/signup?utm_source=bentoml_signup)

EngineeringEngineering

# The Shift to Distributed LLM Inference: 3 Key Technologies Breaking Single-Node Bottlenecks

Explore 3 key strategies — prefill/decode disaggregation, KV cache utilization-aware load balancing, and prefix-aware routing — to optimize distributed LLM inference at scale.

### Authors

![Bo Jiang](/_next/image?url=https%3A%2F%2Fadmin.bentoml.com%2Fuploads%2Fbo_daf117c7b6.jpg&w=3840&q=75)

Bo Jiang

![Sherlock Xu](/_next/image?url=https%3A%2F%2Fadmin.bentoml.com%2Fuploads%2Fsherlock_a89966bd17.png&w=3840&q=75)

Sherlock Xu

### Last Updated

June 11, 2025

### Share

![](/_next/image?url=https%3A%2F%2Fadmin.bentoml.com%2Fuploads%2Fdistributed_llm_inference_4ef1647434.png&w=3840&q=100)

The landscape of LLM inference is rapidly evolving, with a clear shift toward distributed serving.

What’s happening? Single-node GPU optimizations like dynamic batching, paged attention, and CUDA tweaks are starting to show limitations as LLM inference scales. Larger models like DeepSeek-R1 and tasks with longer context, such as reasoning or agentic use cases, stretch the limits even further.

Today, inference optimization is no longer just about squeezing more out of a single node, but rethinking how to distribute inference across a cluster of workers. If you’re working to deploy and scale LLM services, this is a trend you can’t afford to ignore. It heralds a new era, one that brings better resource allocation, smarter GPU usage, lower token latency, and reduced cost per generated token.

Leading AI teams and open-source communities are already pioneering distributed inference strategies. We’ve seen optimization efforts converge on three key areas:

* Prefill-decode (PD) disaggregation
* KV cache utilization-aware load balancing
* Prefix-aware routing

In this blog post, we’ll walk through each of these and highlight the active progress to address the challenges they present.

## PD disaggregation[#](#pd%20disaggregation)

To understand PD disaggregation, let’s start with how LLM inference actually works.

For transformer-based LLMs, every time you send a prompt, the model goes through two key steps:

* **Prefill**: Processes the entire sequence in parallel and store key and value vectors from the attention layers in a KV cache. This helps the model efficiently output new tokens later without recomputing everything from scratch. Because it’s handling all the tokens at once, prefill is compute-bound, but not too demanding on GPU memory.
* **Decode**: Generates the output tokens, one at a time, by reusing the KV cache built earlier. Different from prefill, decode requires fast memory access but lower compute.

![llm-inference-diagram.png](https://admin.bentoml.com/uploads/llm_inference_diagram_d1439d26f5.png)

Prefill and decode in LLM inference

For a long time, the standard way of doing inference was to run these two steps together. On the surface, this might seem straightforward.

In practice, you’ll often have multiple requests arriving at once. Each one has its own prefill and decode needs, but only one phase can run at a time. When the GPU is occupied with compute-heavy prefill tasks, decode tasks must wait, which increases ITL, and vice versa.

Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously.

![collocating-prefill-and-decode.png](https://admin.bentoml.com/uploads/collocating_prefill_and_decode_16b8fb3c90.png)

Latency increase by collocating prefill and decode. Source: [DistServe Paper](https://arxiv.org/pdf/2401.09670)

### Why disaggregation makes sense[#](#why%20disaggregation%20makes%20sense)

The idea of PD disaggregation is simple: separate these two very different tasks so they don’t get in each other’s way. Key benefits include:

* **Dedicated resource allocation**: Prefill and decode can be scheduled and scaled independently on different hardware. For example, if your workload has lots of prompt overlap (like multi-turn conversations or agentic workflows), it means much of your KV cache can be reused. As a result, there’s less compute demand on prefill and you can put more resources on decode.
* **Parallel execution**: Prefill and decode phases don’t interfere with each other anymore. You can run them more efficiently in parallel, which means better concurrency and throughput.
* **Independent tuning**: You can implement different optimization techniques (like tensor or pipeline parallelism) for prefill and decode to better meet your goals for TTFT and ITL.

Several open-source frameworks and projects are actively exploring PD disaggregation, including [SGLang](https://github.com/sgl-project/sglang/issues/4655), [vLLM](https://docs.vllm.ai/en/latest/features/disagg_prefill.html), [Dynamo](https://docs.nvidia.com/dynamo/latest/architecture/disagg_serving.html), and [llm-d](https://docs.google.com/document/d/1FNN5snmipaTxEA1FGEeSH7Z_kEqskouKD1XYhVyTHr8/edit?pli=1&tab=t.0).

### Disaggregation isn’t always a silver bullet[#](#disaggregation%20isn%E2%80%99t%20always%20a%20silver%20bullet)

As promising as PD disaggregation sounds, it’s not a one-size-fits-all fix.

* **Thresholds matter**: If your workload is too small, or your GPU setup isn’t tuned for this approach, performance can drop (by 20-30% in our tests).
* **Local prefill can be faster**: For shorter prompts or when the decode engine has a high prefix cache hit, running prefill locally on the decode worker is often faster and simpler.
* **Data transfer cost**: Disaggregation requires moving KV caches rapidly and reliably between prefill and decode workers. This means your solution must support fast, low-latency communication protocols that are both hardware- and network-agnostic. Unless the performance gains from disaggregation outweigh the data transfer cost, overall performance can actually degrade. Existing methods for data transfer for your reference: [NVIDIA Inference Xfer Library (NIXL)](https://github.com/ai-dynamo/nixl), CXL, NVMe-oF.

  ![kv-cache-transfer-process.png](https://admin.bentoml.com/uploads/kv_cache_transfer_process_e2f1ee49a6.png)

## KV cache utilization-aware load balancing[#](#kv%20cache%20utilization-aware%20load%20balancing)

For traditional web applications, load balancing is usually pretty simple. Requests are small, responses are quick, and any backend instance can handle any request equally well. Load balancers can use simple strategies like round-robin to distribute traffic evenly.

But things are completely different for LLM inference. A major factor here is the KV cache built during the prefill phase.

Traditional load balancers treat LLM workers like identical black boxes. They don’t see what’s going on inside each worker, including:

* How much GPU memory is consumed by the KV cache
* How long the request queue is

![kv-cache-util-lb.png](https://admin.bentoml.com/uploads/kv_cache_util_lb_adeb63e5f4.png)

When a load balancer can’t see these details, it starts making bad decisions, leading to:

* **Missed cache reuse**: New requests with similar prefixes can't leverage existing cached computations (more details in the next section).
* **Increased latency**: Conversations routed to wrong replicas lose their KV cache, requiring expensive re-computation.
* **Load imbalance**: Some workers process many long conversations while others remain idle.

The open-source community is already working on smarter solutions. For example, the [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) project uses an endpoint picker (EPP) to collect information on KV cache utilization, queue length, and LoRA adapters on each worker, and routes requests to the optimal replica for better inference.

## Prefix-aware routing[#](#prefix-aware%20routing)

The term "KV cache" originally described caching within a single inference request. As mentioned above, LLMs work autoregressively during decode as they output the next new token based on the previously generated tokens (i.e. reusing their KV cache). Without the KV cache, they need to recompute everything for the previous tokens in each decode step, which would be a huge waste of resources.

When extending this caching concept across multiple requests, it’s more accurate to call it **prefix caching.**

Imagine you have a chatbot system with a prompt like this:

```
You are a helpful AI writer. Please write in a professional manner.
```

This system prompt doesn’t change from one conversation to the next. When new messages come in, the model can reuse the stored prefix cache, only processing the new part of the prompt.

Here’s the challenge: How can a new request be routed to the worker that already has the right prefix cached? How does the router know what’s in each worker’s cache?

![prefix-caching-aware-routing.png](https://admin.bentoml.com/uploads/prefix_caching_aware_routing_be934fc6b1.png)

Different open-source projects are exploring their own approaches:

* **Worker-reported prefix status**

  [Dynamo](https://github.com/ai-dynamo/dynamo) has workers actively report which prefixes they’ve cached. The router then uses this real-time data to make smart routing decisions.
* **Router-predicted cache status**

  [SGLang](https://github.com/sgl-project/sglang) maintains an approximate radix tree for each worker based on past requests. This helps the router predict which worker is most likely to have the needed prefix, without constant updates from the workers.
* **Hybrid efforts**

  + The Gateway API Inference Extension project is [exploring multiple strategies to implement a routing algorithm on EPP](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/498):

    - **Prefix affinity consistent hashing**: Group requests with similar prefixes to the same worker.
    - **Approximate prefix cache on the router**: Let the router maintain an approximate lookup cache of the prefix caches on all the backend servers.
    - **Accurate prefix cache on the router**: Gather KV cache information reported by model servers.
  + The [llm-d](https://github.com/llm-d/llm-d) project uses a component called Inference Scheduler to implement filtering and scoring algorithms, and makes routing decisions based on a combination of factors like cache availability, prefill/decode status, SLA and load.

## Conclusion[#](#conclusion)

Distributed inference is becoming essential for deploying and scaling LLMs at larger scales. If an enterprise wants to truly optimize for metrics like latency or throughput, distributed LLM inference is the only real path forward. This goes far beyond what serverless API endpoints can achieve.

At Bento, we’re working to ensure our customers and users can tap into these latest LLM inference optimizations. As we’ve seen in our initial experiments, there’s no one-size-fits-all solution. The best approach depends on your specific workloads, models, and what matters most to you (e.g., latency, throughput, or cost).

We’ll be sharing more about distributed LLM inference with detailed benchmarks and best practices in future posts. If you want to collaborate or stay in the loop, feel free to reach out!

* Read our [LLM Inference Handbook](https://bentoml.com/llm/)
* [Schedule a call with our experts](https://www.modular.com/request-demo?utm_source=bentoml_blog) to discuss how these advanced inference optimizations could fit into your workloads.
* [Join our community forum](https://forum.modular.com/c/bento/31?utm_source=bentoml_blog) to connect with other builders and get the latest information on LLM inference.
* [Sign up for our unified inference platform](https://console.modular.com/signup?utm_source=bentoml_blog) to deploy and scale [cutting-edge open-source LLMs](https://www.bentoml.com/blog/navigating-the-world-of-open-source-large-language-models).

## Subscribe to our newsletter

Stay updated on AI infrastructure, inference techniques, and performance optimization.

Subscribe

### Products

[Bento Inference Platform](https://cloud.bentoml.com/)[BentoML Open-Source](https://github.com/bentoml/BentoML)[OpenLLM](https://github.com/bentoml/OpenLLM)[LLM-Optimizer](https://github.com/bentoml/llm-optimizer)[Comfy-Pack](https://github.com/bentoml/comfy-pack)

### Resources

[Documentation](https://docs.bentoml.com/)[Blog](/blog)[LLM Inference Handbook](/llm)[LLM Performance Explorer](/llm-perf)[Example Projects](https://docs.bentoml.com/en/latest/examples/overview.html)[AI Infrastructure Report](/2024-ai-infra-report)

### Company

[Careers](https://www.modular.com/company/careers)[Privacy Policy](/privacy)[LinkedIn](https://www.linkedin.com/company/modular-ai/)[X](https://x.com/Modular)

### Join our community

[GitHub](https://github.com/modular/modular)[Discord](https://discord.gg/modular)[Forum](https://forum.modular.com/c/bento/31)