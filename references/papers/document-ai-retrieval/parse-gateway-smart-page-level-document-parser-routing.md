[Introducing ExtractBench, the most comprehensive document extraction benchmark. Learn More →](https://www.llamaindex.ai/blog/introducing-extractbench)

* Product

  [LlamaParse

  Industry-leading document processing](/llamacloud)

  + [Parse](/llamaparse)
  + [Extract](/llamaextract)
  + [Index](/llamacloud-index)

  [Open Source

  OSS repos trusted by millions of developers](/llamacloud)

  + [LiteParse](/liteparse)
  + [Workflows](/workflows)
  + [LlamaIndex](/llamaindex)
* Solutions

  Persona

  + [Engineering & R&D   Accelerate product development](/solutions/engineering)
  + [Administrative Operations   Streamline business processes](/solutions/administrative-operations)
  + [Financial Analysts   Build AI-powered financial models](/solutions/finance)

  Industry

  + [Insurance   Automate claims and underwriting](/industry/insurance)
  + [Finance   Power financial research](/industry/finance)
  + [Manufacturing   Optimize system uptime](/industry/manufacturing)
  + [Healthcare & Pharma   Accelerate clinical research](/industry/healthcare-pharma)

  Use cases

  + [Financial Due Diligence   Speed up compliance reviews](/use-cases/financial-due-diligence)
  + [Invoice Processing   Automate manual review](/use-cases/invoice-processing)
  + [Technical Document Search   Find answers in complex docs](/use-cases/technical-document-search)
  + [Customer Support   Instant, accurate responses](/use-cases/customer-support)
* [Docs](https://developers.llamaindex.ai/)
* Resources

  Resources

  + [Customer stories   See real-world success stories](/customers)

  [Jeppesen (a Boeing Company) Saves ~2,000 Engineering Hours with Unified Chat Framework](/customers/jeppesen-a-boeing-company-saves-2-000-engineering-hours-with-unified-chat-framework)
* Company

  + [About us   Our mission and story](/about)
  + [Careers   Join our growing team](/careers)
  + [Brand   Logos and brand guidelines](/brand)

  [View open roles at LlamaIndex](/careers)
* [Blog](/blog)
* [Pricing](/pricing)

* [Book a demo](/contact)
* Sign in

  + [Global](https://cloud.llamaindex.ai/)
  + [EU](https://cloud.eu.llamaindex.ai/)

* Product

  [LlamaParse

  Industry-leading document processing](/llamacloud)

  + [Parse](/llamaparse)
  + [Extract](/llamaextract)
  + [Index](/llamacloud-index)

  [Open Source

  OSS repos trusted by millions of developers](/llamacloud)

  + [LiteParse](/liteparse)
  + [Workflows](/workflows)
  + [LlamaIndex](/llamaindex)
* Solutions

  Persona

  + [Engineering & R&D   Accelerate product development](/solutions/engineering)
  + [Administrative Operations   Streamline business processes](/solutions/administrative-operations)
  + [Financial Analysts   Build AI-powered financial models](/solutions/finance)

  Industry

  + [Insurance   Automate claims and underwriting](/industry/insurance)
  + [Finance   Power financial research](/industry/finance)
  + [Manufacturing   Optimize system uptime](/industry/manufacturing)
  + [Healthcare & Pharma   Accelerate clinical research](/industry/healthcare-pharma)

  Use cases

  + [Financial Due Diligence   Speed up compliance reviews](/use-cases/financial-due-diligence)
  + [Invoice Processing   Automate manual review](/use-cases/invoice-processing)
  + [Technical Document Search   Find answers in complex docs](/use-cases/technical-document-search)
  + [Customer Support   Instant, accurate responses](/use-cases/customer-support)
* [Docs](https://developers.llamaindex.ai/)
* Resources

  + [Customer stories   See real-world success stories](/customers)

  [How Jeppesen (a Boeing Company) Saves ~2,000 Engineering Hours with Unified Chat Framework](/customers/jeppesen-a-boeing-company-saves-2-000-engineering-hours-with-unified-chat-framework)
* Company

  + [About us   Our mission and story](/about)
  + [Careers   Join our growing team](/careers)
  + [Brand   Logos and brand guidelines](/brand) [![](/_astro/header-careers.q9Hk8rsS_1RSkuD.webp)
  View open roles at LlamaIndex](/careers)
* [Blog](/blog)
* [Pricing](/pricing)

* [Book a demo](/contact)
* Sign in

  + [Global](https://cloud.llamaindex.ai/)
  + [EU](https://cloud.eu.llamaindex.ai/)

[← Back](/blog)

Jul 30, 2026

* [[ Product ]](/blog?tag=Product)

# Parse Gateway: Smart, Page-Level Document Parser Routing

By

![](https://cdn.sanity.io/images/7m9jw85w/production/f7648777f8669bbb5945705ced4827a158de933b-1080x1080.png?w=80)

Clelia Astra Bertelli

![](https://cdn.sanity.io/images/7m9jw85w/production/041943970563e1ba72230363f9417221430023d9-1200x676.png?w=1200)

You don’t always need the best parser

25

* [You don’t always need the best parser](#you-dont-always-need-the-best-parser)
* [Routing for everyone — including your agents](#routing-for-everyone-including-your-agents)
* [What this means for document processing](#what-this-means-for-document-processing)

Content

* [You don’t always need the best parser](#you-dont-always-need-the-best-parser)
* [Routing for everyone — including your agents](#routing-for-everyone-including-your-agents)
* [What this means for document processing](#what-this-means-for-document-processing)

Follow us on

25

The raise of agents has shifted the software industry’s focus from using *the* best model for every task to looking for the best combination of models that can carry out each specific task rapidly, reliably and at a relatively contained price.

The concept behind this shift is known as **routing**: while [early implementations](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/router_query_engine.py) were already circulating a few years ago, especially in the world of RAG. With the recent surge in token spend the idea has really taken off again.

Many players in the field have already started offering routing as a core part of their platform, such as OpenRouter with [Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/) and Sakana AI with [Fugu](https://sakana.ai/fugu/), and one pattern emerges from these services: a good orchestrator/dispatcher is the element that determines the success of a routing system when carrying out a task.

While most of the routing world is focused on language models powering agents, we decided to apply this same concept and learnings to something closer to us: **document parsing**, with our new [Parse Gateway](https://parse-gateway.dev) webapp.

## You don’t always need the best parser

Document processing pipelines often tend to oversee the parsing complexity needs for each file, let alone for each page in a file.

There are, instead, two widespread approaches:

* Use the best parser, sacrificing cost and latency in favor of output quality
* Use the fastest parser, losing output quality in favor of reduced latency and costs

These strategies are often “flat”, meaning that they apply to all documents that flow through the pipeline without any nuance or distinction. Even in those systems where some differentiation is applied, it is mostly applied at the file level, and humans or VLMs are the ones determining complexity and routing to the appropriate pipeline, implying higher costs and longer processing times that discourage companies from adopting this approach.

With Parse Gateway, we decided to take a different path, following the introduction, in LiteParse v2.2.0, of the `is_complex` functionality, which estimates the complexity of a document at the page level, determining the need for OCR and the reasons why more advanced parsing techniques might be needed, also informed by layout complexity signals.

The idea is simple: when a file gets uploaded, LiteParse estimates its complexity at the page level, and each page is routed to a LlamaParse tier based on why — and how severely — it needs more than a cheap text-only pass. Each reason `is_complex` surfaces implies a baseline tier:

| Reason | Explanation | Baseline Tier |
| --- | --- | --- |
| `no-text` | Almost no extractable native text, and no full-page raster behind it (a blank page, or a near-empty cover/divider). | Cost Effective |
| `scanned` | A single raster covers essentially the whole page and there is little or no extractable text behind it (a scanned/photographed page). | Agentic |
| `sparse-text` | Some real text, but it covers very little of the page. Typically a figure-heavy page with only thin captions. | Agentic |
| `embedded-images` | Substantial embedded raster figures sit alongside the native text. | Agentic |
| `garbled` | The native text decodes to garbage (broken cmap / Type3 char-code fallback), so the visible glyphs and the extracted text disagree. | Agentic Plus |
| `vector-text` | Text is painted as filled vector outlines, outside the text layer, so no native text items represent it. | Agentic Plus |

But we don't stop at the baseline. `is_complex` also returns the magnitude behind each reason: how little of the page is actually covered by text, how much of a garbled or vector-text page is affected, how many separate images are interleaved with the body text… and Parse Gateway uses these metrics to escalate a page past its baseline tier when the signal says the page is harder than the reason alone would suggest. A page with three or more reasons firing at once is escalated too, since problems compounding across dimensions (say, sparse text *and* embedded images *and* garbling on the same page) tend to be harder than any single reason in isolation. On top of that, layout complexity signals (multi-column reading order, ruled or borderless tables, dense figure coverage) are folded in independently: a page can need no OCR at all and still get bumped up a tier if its structure is complex enough to trip up a single-pass extraction.

If no OCR is needed and the layout is simple, the page is routed to LiteParse (which can output Markdown, as of v2.1.0). In this way, you don't parse an entire file with one parser, but you scatter its pages across different tiers based on how difficult each one actually is, driving down cost and latency for non-OCR pages (LiteParse runs in-process, for free), without losing accuracy for more complex pages that get routed through more capable LlamaParse tiers.

Here is an animation of the routing flow:

 ![](https://cdn.sanity.io/images/7m9jw85w/production/9157de7b0296b7d54b68f520d76f6c0eba4e5161-1200x720.png)

## Routing for everyone — including your agents

The intelligent routing available in the Parse Gateway isn't limited to the web app: we've also brought the same capabilities to our MCP server.

By adding `https://mcp.llamaindex.ai/mcp` (or `https://mcp.llamaindex.ai/parse/mcp` , if you just want the subset of tools specific for parsing) as an MCP server for your agent, you'll gain access to two additional tools:

* `estimateFileComplexity` — predicts whether a document requires full parsing or can be handled by LiteParse.
* `parseWithLiteParse` — lets your agent explicitly route compatible documents to LiteParse for lower latency and zero-cost, in-process parsing.

This enables agents to make parsing decisions automatically: they can first estimate a document's complexity and then choose the most appropriate parsing tier, striking the right balance between speed, cost, and extraction quality without requiring any hardcoded heuristics.

Under the hood, `estimateFileComplexity` uses the same algorithm that powers the Parse Gateway's `/is-complex` endpoint, ensuring routing decisions are consistent whether you're using the web interface or an MCP-powered agent.

## What this means for document processing

Complexity-based routing might be the missing link in your document processing pipelines: PDFs and other unstructured documents are not homogeneous blocks of pages, they often contain a mixture of pages with perfectly clear text, images, tables and scanned content.

In this sense, a one-size-fits-all approach inevitably brings along a set of tradeoffs that favor one vertex of the cost-latency-accuracy triangle, while losing ground on the others. Inferring the complexity of a page and parsing it with a dedicated tier is a first step towards a solution that embraces all three vertices, without noticeable sacrifices in any of them.

You can try out Parse Gateway in the web app demo, and find the code in the GitHub repository: <https://github.com/run-llama/parse-gateway>.

Let us know what you think!

Related articles

PortableText [components.type] is missing "undefined"

* [Introducing ExtractBench: The Most Comprehensive Benchmark for Data Extraction from Enterprise Documents](/blog/introducing-extractbench)

  Aug 11, 2026

  + [[ Product ]](/blog?tag=Product)
  + [ +1 ]

   [![](https://cdn.sanity.io/images/7m9jw85w/production/a43a9471bb27e008c02cec63f7e1e44080a9bd8a-1200x620.png)](/blog/introducing-extractbench)
* [Document OCR is Not Getting Commoditized](/blog/document-ocr-is-not-getting-commoditized)

  Aug 5, 2026

  + [[ Product ]](/blog?tag=Product)

   [![](https://cdn.sanity.io/images/7m9jw85w/production/d891ef4efcbbc97d5025b50b5ccbdfe8551a1f2a-2400x1260.png)](/blog/document-ocr-is-not-getting-commoditized)
* [Introducing liteparse-grpc: A gRPC Server for LiteParse](/blog/introducing-liteparse-grpc-a-grpc-server-for-liteparse)

  Jul 16, 2026

  + [[ Product ]](/blog?tag=Product)

   [![](https://cdn.sanity.io/images/7m9jw85w/production/690d9a232d95ea83f70bbef2f7ba16bb1806a94e-1200x676.png)](/blog/introducing-liteparse-grpc-a-grpc-server-for-liteparse)

## Start building your first document agent today

PortableText [components.type] is missing "undefined"

* [Sign up for free](https://cloud.llamaindex.ai/)
* [Book a demo](/contact)

## Build document agents that understand, reason, and act

* [Contact sales](/contact)
* [Sign up](https://cloud.llamaindex.ai/)

Explore AI Summary

### Solutions

* [Engineering & R&D](/solutions/engineering)
* [Administrative Operations](/solutions/administrative-operations)
* [Financial Analysts](/solutions/finance)
* [Developers](/)
* [Insurance](/industry/insurance)
* [Finance](/industry/finance)
* [Manufacturing](/industry/manufacturing)
* [Healthcare & Pharma](/industry/healthcare-pharma)
* [Finance Due Diligence](/use-cases/financial-due-diligence)
* [Invoice Processing](/use-cases/invoice-processing)
* [Technical Document Search](/use-cases/technical-document-search)
* [Customer Support](/use-cases/customer-support)

### Products

* [LlamaParse](/llamacloud)
  + [Parse](/llamaparse)
  + [Extract](/llamaextract)
  + [Index](/llamacloud-index)
* [LlamaIndex](/llamaindex)
* [Workflows](/workflows)

### Resources

* [Customer Stories](/customers)
* [Glossary](/glossary)
* [Applications](/services)
* [Insights](/insights)

### Company

* [Pricing](/pricing)
* [Blog](/blog)
* [About us](/about)
* [Careers](/careers)
* [Brand](/brand)
* [Trust center](https://security.llamaindex.ai/)

### Weekly newsletter

Get a weekly roundup of the latest news and insights on the world of LLMs and word on the newest features of
the LlamaIndex libraries.

  [![Compliance](/_astro/compliance.Ciny7Bnl_QyWAL.webp)](https://security.llamaindex.ai/)

© 2026 LlamaIndex

* [Privacy Notice](/legal/privacy-notice)
* [Terms of Service](/legal/terms-of-service)
* [Data Processing Addendum](https://powerforms.docusign.net/c2d2c960-b423-418b-8c0d-142d6d99960e?env=na4&acct=ee4e6249-cd9d-4111-89ef-aa930b7cbb40&accountId=ee4e6249-cd9d-4111-89ef-aa930b7cbb40)

×