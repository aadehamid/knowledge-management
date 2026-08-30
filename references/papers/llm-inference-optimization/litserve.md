title: GitHub - Lightning-AI/LitServe: A minimal Python framework for building custom AI inference servers with full control over logic, batching, and scaling.
description: A minimal Python framework for building custom AI inference servers with full control over logic, batching, and scaling. - Lightning-AI/LitServe

# GitHub - Lightning-AI/LitServe: A minimal Python framework for building custom AI inference servers with full control over logic, batching, and scaling.

[![Lightning](https://camo.githubusercontent.com/842979a74a8ced1e94ca327bca2ee1f63f4358e1d03c39264ea1c5676b00d432/68747470733a2f2f706c2d626f6c74732d646f632d696d616765732e73332e75732d656173742d322e616d617a6f6e6177732e636f6d2f6170702d322f6c735f62616e6e6572322e706e67){width=800px}](https://camo.githubusercontent.com/842979a74a8ced1e94ca327bca2ee1f63f4358e1d03c39264ea1c5676b00d432/68747470733a2f2f706c2d626f6c74732d646f632d696d616765732e73332e75732d656173742d322e616d617a6f6e6177732e636f6d2f6170702d322f6c735f62616e6e6572322e706e67)

[Quick start](#quick-start) • [Examples](#featured-examples) • [Features](#features) • [Performance](#performance) • [Hosting](#host-anywhere) • [Docs](https://lightning.ai/docs/litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)

Most serving tools (vLLM, etc..) are built for a single model type and enforce rigid abstractions. They work well until you need custom logic, multiple models, agents, or non standard pipelines. LitServe lets you write your own inference engine in Python. You define how requests are handled, how models are loaded, how batching and routing work, and how outputs are produced. LitServe handles performance, concurrency, scaling, and deployment. Use LitServe to build inference APIs, agents, chatbots, RAG systems, MCP servers, or multi model pipelines.

Run it locally, self host anywhere, or deploy with one click on [Lightning AI](https://lightning.ai/litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme).

Over 380,000 developers use [Lightning Cloud](https://lightning.ai/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme), the simplest way to run LitServe without managing infrastructure. Deploy with one command, get autoscaling GPUs, monitoring, and a free tier. No cloud setup required. Or self host anywhere.

Install LitServe via pip ([more options](https://lightning.ai/docs/litserve/home/install)):

```
pip install litserve
```

[Example 1](#inference-engine-example): Toy inference pipeline with multiple models.\
 [Example 2](#agent-example): Minimal agent to fetch the news (with OpenAI API).\
 ([Advanced examples](#featured-examples)):

```
import litserve as ls

# define the api to include any number of models, dbs, etc...
class InferenceEngine(ls.LitAPI):
    def setup(self, device):
        self.text_model = lambda x: x**2
        self.vision_model = lambda x: x**3

    def predict(self, request):
        x = request["input"]    
        # perform calculations using both models
        a = self.text_model(x)
        b = self.vision_model(x)
        c = a + b
        return {"output": c}

if __name__ == "__main__":
    # 12+ features like batching, streaming, etc...
    server = ls.LitServer(InferenceEngine(max_batch_size=1), accelerator="auto")
    server.run(port=8000)
```

Deploy for free to [Lightning cloud](#hosting-options) (or self host anywhere):

```
# Deploy for free with autoscaling, monitoring, etc...
lightning deploy server.py --cloud

# Or run locally (self host anywhere)
lightning deploy server.py
# python server.py
```

Test the server: Simulate an http request (run this on any terminal):

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 4.0}'
```

```
import re, requests, openai
import litserve as ls

class NewsAgent(ls.LitAPI):
    def setup(self, device):
        self.openai_client = openai.OpenAI(api_key="OPENAI_API_KEY")

    def predict(self, request):
        website_url = request.get("website_url", "https://text.npr.org/")
        website_text = re.sub(r'<[^>]+>', ' ', requests.get(website_url).text)

        # ask the LLM to tell you about the news
        llm_response = self.openai_client.chat.completions.create(
           model="gpt-3.5-turbo", 
           messages=[{"role": "user", "content": f"Based on this, what is the latest: {website_text}"}],
        )
        output = llm_response.choices[0].message.content.strip()
        return {"output": output}

if __name__ == "__main__":
    server = ls.LitServer(NewsAgent())
    server.run(port=8000)
```

Test it:

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"website_url": "https://text.npr.org/"}'
```

A few key benefits:

- **Deploy any pipeline or model**: Agents, pipelines, RAG, chatbots, image models, video, speech, text, etc...
- **No MLOps glue:** LitAPI lets you build full AI systems (multi-model, agent, RAG) in one place ().
- **Instant setup:** Connect models, DBs, and data in a few lines with `setup()` ().
- **Optimized:** autoscaling, GPU support, and fast inference included ().
- **Deploy anywhere:** self-host or one-click deploy with Lightning ().
- **FastAPI for AI:** Built on FastAPI but optimized for AI - 2× faster with AI-specific multi-worker handling ().
- **Expert-friendly:** Use vLLM, or build your own with full control over batching, caching, and logic ().

> ⚠️ Not a vLLM or Ollama alternative out of the box. LitServe gives you lower-level flexibility to build what they do (and more) if you need it.

Here are examples of inference pipelines for common model types and use cases.

```
Toy model:      Hello world
LLMs:           Llama 3.2, LLM Proxy server, Agent with tool use
RAG:            vLLM RAG (Llama 3.2), RAG API (LlamaIndex)
NLP:            Hugging face, BERT, Text embedding API
Multimodal:     OpenAI Clip, MiniCPM, Phi-3.5 Vision Instruct, Qwen2-VL, Pixtral
Audio:          Whisper, AudioCraft, StableAudio, Noise cancellation (DeepFilterNet)
Vision:         Stable diffusion 2, AuraFlow, Flux, Image Super Resolution (Aura SR),
                Background Removal, Control Stable Diffusion (ControlNet)
Speech:         Text-speech (XTTS V2), Parler-TTS
Classical ML:   Random forest, XGBoost
Miscellaneous:  Media conversion API (ffmpeg), PyTorch + TensorFlow in one API, LLM proxy server
```

[Browse 100\+ community-built templates](https://lightning.ai/studios?section=serving&utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)

Self-host with full control, or deploy with [Lightning AI](https://lightning.ai/?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) in seconds with autoscaling, security, and 99.995% uptime.\
 **Free tier included. No setup required. Run on your cloud**

```
lightning deploy server.py --cloud
```

> [!NOTE]+ deploy.mp4
> [\[Video\]](https://private-user-images.githubusercontent.com/3640001/427828144-ff83dab9-0c9f-4453-8dcb-fb9526726344.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODgwMzY5NTAsIm5iZiI6MTc4ODAzNjY1MCwicGF0aCI6Ii8zNjQwMDAxLzQyNzgyODE0NC1mZjgzZGFiOS0wYzlmLTQ0NTMtOGRjYi1mYjk1MjY3MjYzNDQubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDgyOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA4MjlUMjA1MDUwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NmVjMTZkMzAwMTE1ZjA2NDQwZjIwNjY5N2NjMzk0Yjc2ZWUxYTE4YzAzZjk2NzVkYzgwNGQ5ZDcwZDAzMWU2MSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPXZpZGVvJTJGbXA0In0.9KJ_NnfgsVVQ75SlRqlWzONXPfOrHTZ9beUuRQ7yJ_Y)

| [Feature](https://lightning.ai/docs/litserve/features?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | Self Managed | [Fully Managed on Lightning](https://lightning.ai/deploy?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) |
|----|----|----|
| Docker-first deployment | ✅ DIY | ✅ One-click deploy |
| Cost | ✅ Free (DIY) | ✅ Generous [free tier](https://lightning.ai/pricing?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) with pay as you go |
| Full control | ✅ | ✅ |
| Use any engine (vLLM, etc.) | ✅ | ✅ vLLM, Ollama, LitServe, etc. |
| Own VPC | ✅ (manual setup) | ✅ Connect your own VPC |
| [(2x)\+ faster than plain FastAPI](#performance) | ✅ | ✅ |
| [Bring your own model](https://lightning.ai/docs/litserve/features/full-control?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Build compound systems (1\+ models)](https://lightning.ai/docs/litserve/home?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [GPU autoscaling](https://lightning.ai/docs/litserve/features/gpu-inference?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Batching](https://lightning.ai/docs/litserve/features/batching?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Streaming](https://lightning.ai/docs/litserve/features/streaming?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Worker autoscaling](https://lightning.ai/docs/litserve/features/autoscaling?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Serve all models: (LLMs, vision, etc.)](https://lightning.ai/docs/litserve/examples?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Supports PyTorch, JAX, TF, etc...](https://lightning.ai/docs/litserve/features/full-control?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [OpenAPI compliant](https://www.openapis.org/) | ✅ | ✅ |
| [Open AI compatibility](https://lightning.ai/docs/litserve/features/open-ai-spec?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [MCP server support](https://lightning.ai/docs/litserve/features/mcp?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Asynchronous](https://lightning.ai/docs/litserve/features/async-concurrency?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅ | ✅ |
| [Authentication](https://lightning.ai/docs/litserve/features/authentication?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ❌ DIY | ✅ Token, password, custom |
| GPUs | ❌ DIY | ✅ 8\+ GPU types, H100s from $1.75 |
| Load balancing | ❌ | ✅ Built-in |
| Scale to zero (serverless) | ❌ | ✅ No machine runs when idle |
| Autoscale up on demand | ❌ | ✅ Auto scale up/down |
| Multi-node inference | ❌ | ✅ Distribute across nodes |
| Use AWS/GCP credits | ❌ | ✅ Use existing cloud commits |
| Versioning | ❌ | ✅ Make and roll back releases |
| Enterprise-grade uptime (99.95%) | ❌ | ✅ SLA-backed |
| SOC2 / HIPAA compliance | ❌ | ✅ Certified & secure |
| Observability | ❌ | ✅ Built-in, connect 3rd party tools |
| CI/CD ready | ❌ | ✅ Lightning SDK |
| 24/7 enterprise support | ❌ | ✅ Dedicated support |
| Cost controls & audit logs | ❌ | ✅ Budgets, breakdowns, logs |
| Debug on GPUs | ❌ | ✅ Studio integration |
| [20\+ features](https://lightning.ai/docs/litserve/features?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | - | - |

LitServe is designed for AI workloads. Specialized multi-worker handling delivers a minimum **2x speedup over FastAPI**.

Additional features like batching and GPU autoscaling can drive performance well beyond 2x, scaling efficiently to handle more simultaneous requests than FastAPI and TorchServe.

Reproduce the full benchmarks [here](https://lightning.ai/docs/litserve/home/benchmarks?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) (higher is better).

[![LitServe](https://camo.githubusercontent.com/9da4385acfcb260babf36f1d5818d8b107da5ee13b7fcabf0d866e199ae7867f/68747470733a2f2f706c2d626f6c74732d646f632d696d616765732e73332e75732d656173742d322e616d617a6f6e6177732e636f6d2f6170702d322f6c735f6368617274735f76362e706e67){width=1000px}](https://camo.githubusercontent.com/9da4385acfcb260babf36f1d5818d8b107da5ee13b7fcabf0d866e199ae7867f/68747470733a2f2f706c2d626f6c74732d646f632d696d616765732e73332e75732d656173742d322e616d617a6f6e6177732e636f6d2f6170702d322f6c735f6368617274735f76362e706e67)

These results are for image and text classification ML tasks. The performance relationships hold for other ML tasks (embedding, LLM serving, audio, segmentation, object detection, summarization etc...).

***💡 Note on LLM serving:*** For high-performance LLM serving (like Ollama/vLLM), integrate [vLLM with LitServe](https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme), use [LitGPT](https://github.com/Lightning-AI/litgpt?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme#deploy-an-llm), or build your custom vLLM-like server with LitServe. Optimizations like kv-caching, which can be done with LitServe, are needed to maximize LLM performance.

LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) - Let's make the world's most advanced AI inference engine.

💬 [Get help on Discord](https://discord.com/invite/MWAEvnC5fU) \
 📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
