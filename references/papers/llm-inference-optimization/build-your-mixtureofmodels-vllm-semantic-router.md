title: Build Your Mixture-of-Models | vLLM Semantic Router
description: Mixture-of-Models is a serving architecture for heterogeneous LLM inference. vLLM Semantic Router makes it practical to deploy.
author: vLLM Semantic Router Team
keywords: Mixture-of-Models runtime, preference-driven AI, open-source LLM router, multi-model routing, model orchestration, model selection, model cascade, Fusion API, micro-agent workflows, semantic router, policy-aware routing, vLLM

# Build Your Mixture-of-Models | vLLM Semantic Router

Mixture-of-Models

1 / 7

Routing vllm-sr/mom-v1-flash → flash recipe → Mistral

Entrypoint requests

1. 01 **Entrypoint** vllm-sr/mom-v1-flash
2. 02 **Recipe** flash recipe
3. 03 **Signals** Detect
4. 04 **Projections** Enrich
5. 05 **Decision** Match
6. 06 **Algorithm** Select
7. 07 **Plugin hooks** Enforce

Security & policy Observability & replay

Model pools

Closed models

- Claude
- ChatGPT
- Gemini

Open models

- Mistral
- DeepSeek
- GLM

Heterogeneous models

Fragmented today

Models specialize in different work.

With vLLM SR

Compose policy-specific model paths.

Fragmented today

GPU generations and accelerators differ in capacity and latency.

With vLLM SR

Route across heterogeneous compute.

Fragmented today

Inference spans edge, private, and cloud.

With vLLM SR

Route to configured local, private, or cloud backends.

Fragmented today

“Best” changes by user and workload.

With vLLM SR

Express preferences as routing policy.

Signals 20

20 signal families spanning request context, safety, intent, preference, and system state.

Algorithms 16

11 selection algorithms and 5 loopers for choosing, composing, and retrying model calls.

Papers 18

18 research papers spanning routing, systems, safety, and multimodality.

The user request is the raw source message before encoding.

Mixture-of-Models proof

## One Model API can beat frontier models {#mom-proof-title}

vLLM Semantic Router keeps the public surface as vllm-sr/auto, then coordinates closed, open, and hybrid model pools inside the serving layer.

Router-side collaboration

### The app calls one model. The router builds the team.

Route by task shape, risk, confidence, and model capability; run bounded collaboration; return one OpenAI-compatible response.

![vLLM Semantic Router routes heterogeneous closed and open model pools](https://vllm-semantic-router.com/img/mom-proof/architecture-router-dark.png)

Data sovereignty

## Keep regulated traffic on approved paths

Residency, locality, and authorization are hard constraints, not preferences. Ineligible paths are removed before ranking ever runs.

1. 01 **Request** Identity \+ context
2. 02 **Hard constraints** Residency · locality · auth
3. 03 **Eligible pool** Approved paths only
4. 04 **Rank** Quality · latency · cost

2026 / Paper POSITION PAPER

### vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models

vLLM Semantic Router Team

arXiv Technical Report

We introduce vLLM Semantic Router, a signal-driven decision routing framework for Mixture-of-Modality deployments that composes heterogeneous signals into deployment-specific routing policies across cost, privacy, latency, and safety constraints.

2026 / Paper VISION PAPER

### The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project

Huamin Chen, Xunzhuo Liu, Bowei He, Fuyuan Lyu, Yankai Chen, Xue Liu, Yuhan Liu, Junchen Jiang

arXiv Technical Report

We synthesize the project’s recent routing, fleet, multimodal, and governance results into the Workload-Router-Pool (WRP) architecture, connecting signal-driven routing to a full-stack inference optimization framework and outlining future research directions across workload, router, and pool design.

2026 / Paper

### Visual Confused Deputy: Exploiting and Defending Perception Failures in Computer-Using Agents

Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen

arXiv Technical Report

We formalize the visual confused deputy as a security failure mode in computer-using agents and introduce a dual-channel guardrail that independently checks click targets and action reasoning before execution.

2026 / Paper

### Outcome-Aware Tool Selection for Semantic Routers: Latency-Constrained Learning Without LLM Inference

Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu

arXiv Technical Report

We introduce Outcome-Aware Tool Selection (OATS), an offline embedding refinement method that improves semantic-router tool ranking under single-digit millisecond CPU budgets without adding serving-time model inference.

2026 / Paper

### Adaptive Vision-Language Model Routing for Computer Use Agents

Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen

arXiv Technical Report

We propose Adaptive VLM Routing (AVR), which estimates action difficulty and routes computer-use agent steps to the cheapest model that still satisfies a target reliability threshold.

2026 / Paper

### 98× Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router

Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen

arXiv Technical Report

We combine Flash Attention, prompt compression, and near-streaming body processing to cut routing latency from seconds to tens of milliseconds while keeping the router lightweight enough to share hardware with serving.

2026 / Paper

### inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference

Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu

arXiv Technical Report

We present a queueing-theory-grounded fleet planner and discrete-event simulator for sizing multi-pool LLM GPU fleets against P99 TTFT targets, without requiring hardware profiling runs up front.

2026 / Paper

### FleetOpt: Analytical Fleet Provisioning for LLM Inference with Compress-and-Route as Implementation Mechanism

Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu

arXiv Technical Report

We derive the minimum-cost two-pool LLM fleet directly from the workload CDF and P99 TTFT target, then use Compress-and-Route to make the optimal boundary deployable in practice.

2026 / Paper

### The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency

Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu

arXiv Technical Report

We derive the 1/W law showing that tokens per watt roughly halve whenever the serving context window doubles, making context-length routing topology a larger energy-efficiency lever than a pure GPU generation upgrade.

2026 / Paper

### Conflict-Free Policy Languages for Probabilistic ML Predicates: A Framework and Case Study with the Semantic Router DSL

Xunzhuo Liu, Hao Wu, Huamin Chen, Bowei He, Xue Liu

arXiv Technical Report

We show how probabilistic ML predicates in policy languages can silently co-fire on the same query, and implement conflict detection plus a softmax-based prevention mechanism in the Semantic Router DSL.

2026 / Paper

### From Inference Routing to Agent Orchestration: Declarative Policy Compilation with Cross-Layer Verification

Huamin Chen, Xunzhuo Liu, Bowei He, Xue Liu

arXiv Technical Report

We extend the Semantic Router DSL from stateless, per-request routing to multi-step agent workflows, emitting verified decision nodes for orchestration frameworks, Kubernetes artifacts, YANG/NETCONF payloads, and protocol-boundary gates from a single declarative source file.

2026 / Paper

### Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents

Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen

arXiv Technical Report

We show that conversational memory and retrieval-grounded routing let a lightweight 8B model recover most of a 235B model’s performance on persistent user-specific queries while cutting effective inference cost by 96%.

2026 / Paper RAG VERIFICATION

### Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems

Xunzhuo Liu, Bowei He, Xue Liu, Haichen Zhang, Huamin Chen

SIGIR 2026 Industry Track

We present a real-time verification component for long-document RAG that processes contexts up to 32K tokens, balancing latency and grounding coverage so interactive systems can detect unsupported answers without falling back to truncated checks.

2026 / Paper

### Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference

Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu

arXiv Technical Report

We propose token-budget-aware pool routing, which estimates each request’s total token budget using a self-calibrating bytes-per-token ratio and dispatches it to short or long vLLM pools to cut fleet cost while avoiding KV-cache failures.

2025 / Paper

### When to Reason: Semantic Router for vLLM

Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen

NeurIPS - MLForSys

We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial.

2025 / Paper

### Category-Aware Semantic Caching for Heterogeneous LLM Workloads

Chen Wang, Xunzhuo Liu, Yue Zhu, Alaa Youssef, Priya Nagpurkar, Huamin Chen

We present a category-aware semantic caching where similarity thresholds, TTLs, and quotas vary by query category, with a hybrid architecture separating in-memory HNSW search from external document storage.

2025 / Paper

### Semantic Inference Routing Protocol (SIRP)

Huamin Chen, Luay Jalil

Internet Engineering Task Force (IETF)

This document specifies the Semantic Inference Routing Protocol (SIRP), a framework for content-level classification and semantic routing in AI inference systems.

2025 / Paper

### Multi-Provider Extensions for Agentic AI Inference APIs

H. Chen, L. Jalil, N. Cocker

Internet Engineering Task Force (IETF) - Network Management Research Group

This document specifies multi-provider extensions for agentic AI inference APIs. Published: 20 October 2025. Intended Status: Informational. Expires: 23 April 2026.

Maintainer

### Xunzhuo Liu

@AMD

Maintainer

### Huamin Chen

@Microsoft

Maintainer

### Kun-Tai Wu

Software Engineer @Delta Electronics

Maintainer

### Aayush Saini

Senior Software Engineer, Data and AI @Red Hat

Committer

### FAUST

Cloud-native Open Source Contributor @Tongji University

Committer

### David Shrader

GTM Tech Lead @Google

Committer

### yangw

Cloud-native Engineer @DaoCloud

Committer

### Ramakrishnan Sathyavageeswaran

Computer Science Engineer @Intuit

Committer

### Teemu Kuusisto

SMTS, Silo AI @AMD

Committer

### Akshay Viswanathan

PMTS, Silo AI @AMD

Committer

### Theo Hsiung

Junior Software Engineer (Research Scientist) @Delta Electronics

Committer

### Wilson Wu

Cloud-Native / AI Engineer

Committer

### Yincheng Ren

Software Engineer @Meta

Committer

### cryo-zd

Individual Contributor

Committer

### Hao Wu

Individual Contributor
