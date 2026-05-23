* [![StartupHub.ai](/favicon-circle.svg)

  ![StartupHub.ai — AI Ecosystem Hub](/logo.svg)StartupHub.ai — AI Ecosystem Hub](/)

Discover

* [Home](/)
* [Search](/search)
* [Trending](/trending)
* [New AI Startups](/new-ai-startups)
* [Categories](/startup-categories)
* [Countries](/countries)
* [Funding Rounds](/recent-funding-rounds)
* [Rankings](/rankings)
* [News](/news)
* [News Rankings](/rankings/news)
* [Conferences](/conferences)
* [Watchlist](/watchlist)
* [Lists](/lists)

Intelligence

* [Market Analysis](/analysis)
* [Comparison](/comparison)
* [Claude's Corner](/claudes-corner)
* [Claude's Trades](/trader-claudes)

Tools

* [Market Map Maker

  NEW](/market-map-maker)
* [Visual TL;DR

  MCP](/visual-tldr)
* [YouTube to Article

  NEW](/youtube-to-article)
* [Email Validator

  MCP](/email-validator)
* [AI Agent Readiness](/agent-readiness)
* [Monitor

  NEW](/monitor)
* [API Docs](/api-docs)

Company

* [Pricing](/pricing)
* [Advertise](/advertise)
* [Publish Content](/editorial)
* [Affiliate Program](/affiliate)
* [About](/about)
* [Terms](/terms)
* [Privacy](/privacy)

Account

* Sign In

Toggle Sidebar

1. [Home](/)
- [AI News](/news)
- Maximizing Kv Cache Hit Rates Wekas Open Source Context Platform Engineering

1. [Home](/)
2. /
3. [AI News](/news)
4. /
5. [AI Video](/category/ai-video)
6. /
7. Maximizing KV-Cache Hit Rates: WEKA's Open-Source Context Platform Engineering

[AI Video](/category/ai-video)

[Preferred on Google](https://www.google.com/preferences/source?q=startuphub.ai "Make us preferred on Google")[Google News](https://news.google.com/publications/CAAqKAgKIiJDQklTRXdnTWFnOEtEWE4wWVhKMGRYQm9kV0l1WVdrb0FBUAE?hl=en-US&gl=US&ceid=US%3Aen "Follow us on Google News")

# Maximizing KV-Cache Hit Rates: WEKA's Open-Source Context Platform Engineering

Nov 25, 2025 at 7:46 AM4 min read

![Maximizing KV-Cache Hit Rates: WEKA's Open-Source Context Platform Engineering](https://cdn.startuphub.ai/storage/v1/object/public/entity-images/posts/1056921.webp)

"If I had to choose just one metric, I'd argue that the KV-cache hit rate is the single most important metric for a production-stage AI agent." This powerful statement, attributed to Manus AI and highlighted by Val Bercovici, WEKA's Chief AI Officer, sets the stage for a compelling discussion on optimizing AI agent performance. At the AIE CODE Summit, Bercovici, joined by Callan Fox, Principal Product Manager at WEKA, unveiled their company's new open-source Context Platform Engineering (CPE) toolkit, designed to tackle the pervasive "token anxiety" plaguing AI development and deployment. Their presentation offered sharp analysis and concrete insights into transforming abstract Service Level Agreement (SLA) requirements into actionable Service Level Objectives (SLOs) for AI agent inference platforms.

The core problem CPE addresses is the inherent inefficiency in how AI agents consume and manage context, particularly the Key-Value (KV) cache. This cache stores past computations, allowing an agent to recall information without re-processing it, directly impacting latency and cost. Without effective management, developers face "token anxiety"—the constant worry about hitting rate limits and incurring unnecessary expenses due to inefficient token usage. Fox elaborated on this, explaining that in the absence of robust CPE, teams often resort to "context financial engineering" or "prompt cache arbitrage," a precarious balancing act of predicting cache writes and reads across various token pricing tiers. This manual optimization is not only complex but often leads to suboptimal performance and escalating costs.

### Related startups

WEKA's solution centers on an open-source benchmarking and testing methodology, which they are actively encouraging the community to adopt and enhance. The toolkit features a sophisticated load generator, developed by Callan Fox, that allows users to configure agent swarms and sub-tasks with precise SLOs. This includes cycling through deterministic and random prompt cycles, alongside various model parallelism options, disaggregated or aggregated prefill and decode options, and critical memory tiering. The overarching goal is to engineer context platforms that dramatically simplify the achievement of maximum KV-cache hit rates, thereby mitigating token anxiety and making AI agent deployment more predictable and cost-effective.

A significant challenge in agentic workflows is the "user-agent cadence mismatch." Human feedback loops are inherently slow, operating in minutes or even hours, while AI agents iterate at a much higher cadence, often in seconds, performing numerous background computations. Many of these background tokens are, in principle, cacheable. However, without a transparent and optimized platform, these opportunities are lost, leading to redundant computations. This problem fundamentally boils down to inefficient "token storage." When a user subscribes to an AI service, they are, in essence, purchasing an "allotment of cache in the memory hierarchy," and the platform's ability to convert those SLA commitments into tangible SLOs through smart caching is paramount.

### Related Reading

* [Unlocking Subagents: Brian John's Hack for Codex CLI](https://www.startuphub.ai/ai-news/ai-video/2025/unlocking-subagents-brian-johns-hack-for-codex-cli/)
* [Agents are Robots Too: The Infrastructure Imperative for Next-Gen AI](https://www.startuphub.ai/ai-news/ai-video/2025/agents-are-robots-too-the-infrastructure-imperative-for-next-gen-ai/)
* [AI Agents: Year of Utility, Decade of Autonomy](https://www.startuphub.ai/ai-news/ai-video/2025/ai-agents-year-of-utility-decade-of-autonomy/)

The reality of caching by current inference providers often falls short. Fox illustrated this with a graph showing "yellow cache misses," representing tokens that are recomputed unnecessarily due to factors like limited cache size or short time-to-live (TTL) settings. Each cache miss translates directly into increased input token costs for API users, potentially tenfold, or for subscription users, it means hitting rate limits faster and experiencing degraded performance. The data reveals that shorter cache TTLs (e.g., 1 minute) lead to constant thrashing, where tokens are repeatedly loaded and dropped from the cache. Extending the TTL to 5 minutes or even an hour significantly improves cache hit rates, as the system can "ride out" downtimes in user conversations, maintaining context and reducing redundant GPU prefills. This underscores the necessity for robust memory tiers capable of holding large volumes of tokens for extended periods.

WEKA addresses these caching inefficiencies through its Augmented Memory Grid (AMG), which sits within a diverse landscape of memory tiers. While High Bandwidth Memory (HBM) and DRAM offer low latency, their capacity is often limited and tightly coupled with compute resources, making scalability challenging. Traditional storage (like POSIX file systems) provides capacity but suffers from high latency. WEKA AMG aims to bridge this gap, offering 1000x denser capacity than DRAM, with DRAM-like speeds, effectively acting as a "Token Warehouse." The CPE toolkit, by integrating with such advanced memory hierarchies, ensures "extreme retrieve performance to minimize time-to-first-token while saving recomputing tokens" and "extreme store performance to not block concurrency or have KV cache evicted before save." This comprehensive approach allows for enough capacity to support a larger number of concurrent users with prioritized context, maximizing the value derived from every token and fundamentally reshaping the economics and performance of AI agent deployment.

© 2025 [StartupHub.ai](/). All rights reserved. Do not enter, scrape, copy, reproduce, or republish this article in whole or in part. Use as input to AI training, fine-tuning, retrieval-augmented generation, or any machine-learning system is prohibited without written license. Substantially-similar derivative works will be pursued to the fullest extent of applicable copyright, database, and computer-misuse laws. See our [terms](/terms).

[#AI](/tag/ai)[#AI Agents](/tag/ai-agents)[#Callan Fox](/tag/callan-fox)[#KV Cache Optimization](/tag/kv-cache-optimization)[#Launch](/tag/launch)[#Open-Source](/tag/open-source)[#Val Bercovici](/tag/val-bercovici)[#WEKA](/tag/weka)

### AI Daily Digest

Get the most important AI news daily.

Continue with GoogleRegister with Email

![Google](https://www.google.com/images/branding/googleg/1x/googleg_standard_color_128dp.png)![Sequoia](/logos/sequoia.jpg)![OpenAI](https://cdn.oaistatic.com/assets/apple-touch-icon-mz9nytnj.webp)![a16z](/logos/a16z.png)

+40k readers