title: Gemma 4 - How to Run Locally | Unsloth Documentation
description: Run Google’s new Gemma 4 models locally, including E2B, E4B, 26B A4B, and 31B.

# Gemma 4 - How to Run Locally | Unsloth Documentation

Gemma 4 is Google DeepMind’s new family of open models, including **12B**, **E2B**, **E4B**, **26B-A4B**, and **31B.** The multimodal, hybrid-thinking models support 140\+ languages, up to **256K context**, and have dense and MoE variants. Gemma 4 is Apache-2.0 licensed and can run on your local device.

**Gemma-4-12B** is new and features unified text, image and audio support. It runs on **8GB** RAM (4-bit) or 14GB (8-bit). **Gemma-4-E2B** and **E4B** also support image and audio. Run on **5GB RAM** (4-bit) or 15GB (full 16-bit) via GGUF, MLX or NVFP4 quants.

[Run Gemma 4](https://unsloth.ai/docs/models/gemma-4#run-gemma-4-tutorials) [Fine-tune Gemma 4](https://unsloth.ai/docs/models/gemma-4/train) [Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat) [Gemma 4 MTP](https://unsloth.ai/docs/models/mtp#gemma-4-mtp)

**Gemma-4-26B-A4B** runs on **18GB** (4-bit) or 28GB (8-bit). **Gemma-4-31B** needs **20GB RAM** (4-bit) or 34GB (8-bit).

You can now run all GGUFs, [MLX](https://unsloth.ai/docs/models/gemma-4#mlx-dynamic-quants) and fine-tune Gemma 4 in [Unsloth Studio](https://unsloth.ai/docs/models/gemma-4#unsloth-studio-guide) (see right).

[**QAT** variants](https://unsloth.ai/docs/models/gemma-4/qat) of Gemma 4 reduce memory requirements around 3x while preserving model quality.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FstfdTMsoBMmsbQsgQ1Ma%252Flandscape%2520clip%2520gemma.gif%3Falt%3Dmedia%26token%3Deec5f2f7-b97a-4c1c-ad01-5a041c3e4013&width=768&dpr=3&quality=100&sign=4c4aa3a765cc7bbddb8b66896c591ae6&sv=3){width=1280 height=720}

### Usage Guide {#usage-guide}

Gemma 4 excels at reasoning, coding, tool use, long-context and agentic workflows, and multimodal tasks. The smaller E2B and E4B variants are designed for phones and laptops, while the larger models target medium-high CPU /VRAM systems such as PCs with NVIDIA RTX GPUs.

**See Gemma 4:**  [**Performance benchmarks**](https://unsloth.ai/docs/models/gemma-4#official-gemma-benchmarks)  **and**  [**GGUF benchmarks**](https://unsloth.ai/docs/models/gemma-4#unsloth-gguf-benchmarks) **.**

**Should I pick 26B-A4B or 31B?**

- **26B-A4B** - balances speed and accuracy. Its MoE design makes it faster than 31B, with 4B active parameters. Pick it if RAM is limited and you are fine trading a bit of quality for speed.
- **31B** - currently the strongest Gemma 4 model. Pick it for maximum quality if you have enough memory and can accept slightly slower speeds.

### Hardware requirements {#hardware-requirements}

**Table: Gemma 4 Inference GGUF recommended hardware requirements** (units \= total memory: RAM \+ VRAM, or unified memory). You can use Gemma 4 on MacOS, NVIDIA RTX GPUs etc.

### Recommended Settings {#recommended-settings}

It is recommended to use Google's default Gemma 4 parameters:

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 64`

#### Thinking Mode {#thinking-mode}

Compared to older Gemma chat templates, Gemma 4 uses the standard `system`, `assistant`, and `user` roles and adds explicit thinking control.

**How to enable thinking:**

Add the token `<|think|>` at the **start of the system prompt**.

**Thinking enabled**

**Thinking disabled**

**Output behavior:**

When thinking is enabled, the model outputs its internal reasoning channel before the final answer.

When thinking is disabled, the larger models may still emit an **empty thought block** before the final answer.

**For example using "**What is the capital of France?":

**then it outputs with:**

**Multi-turn chat rule:**

For multi-turn conversations, **only keep the final visible answer in chat history**. Do **not** feed prior thought blocks back into the next turn.

**How to disable thinking:**

Note `llama-cli` might not work reliably, so use `llama-server` for disabling reasoning:

To [disable thinking / reasoning](https://unsloth.ai/docs/models/gemma-4#how-to-enable-or-disable-reasoning-and-thinking), use `--chat-template-kwargs '{"enable_thinking":false}'`

If you're on **Windows** Powershell, use: `--chat-template-kwargs "{\"enable_thinking\":false}"`

Use 'true' and 'false' interchangeably.

## Run Gemma 4 Tutorials {#run-gemma-4-tutorials}

Because Gemma 4 GGUFs comes in several sizes, the recommended starting point for the small models is 8-bit and the larger models is [**Dynamic**](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)  **4-bit**. [Gemma 4 GGUFs](https://huggingface.co/collections/unsloth/gemma-4) or [MLX](https://unsloth.ai/docs/models/gemma-4#mlx-dynamic-quants) or [NVFP4](https://unsloth.ai/docs/models/gemma-4#nvfp4-guide):

[🦥 Unsloth Desktop Guide](https://unsloth.ai/docs/models/gemma-4#unsloth-studio-guide) [🦙 Llama.cpp Guide](https://unsloth.ai/docs/models/gemma-4#llama.cpp-guide) [NVFP4 Guide](https://unsloth.ai/docs/models/gemma-4#nvfp4-guide)

### 🦥 Unsloth Guide {#unsloth-guide}

Gemma 4 can now be run and fine-tuned in [Unsloth Desktop](https://unsloth.ai/docs/new/studio), our new open-source desktop UI for local AI. Unsloth Studio lets you run models locally on **MacOS, Windows**, Linux and:

- Search, download, [run GGUFs](https://unsloth.ai/docs/new/studio#run-models-locally) and safetensor models
- Fast CPU \+ GPU inference via llama.cpp

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FpAGvwjGD0iVMZKBoyu7m%252Fgreeennn.png%3Falt%3Dmedia%26token%3Dd17a5528-8375-444c-9aff-f9e9f7903bcd&width=768&dpr=3&quality=100&sign=a3410511eddef55858443f963e911cfb&sv=3){width=2560 height=1600}

#### Launch Unsloth {#launch-unsloth}

**MacOS, Linux, WSL and Windows:**

Then open `http://127.0.0.1:8888` or your specific URL in your browser.

**Launch Unsloth securely with HTTPS and Cloudflare**

**NEW!** Unsloth now provides a secure way to launch Unsloth over HTTPS through a free Cloudflare tunnel. Use the below (works in Windows, Mac & Linux):

#### Search and download Gemma 4 {#search-and-download-gemma-4}

On first launch you will need to create a password to secure your account and sign in again.

Then go to the [Unsloth Chat](https://unsloth.ai/docs/new/studio/chat) tab and search for Gemma 4 in the search bar and download your desired model and quant. Unsloth supports the latest Gemma-4-12B Unified model.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FpYoNILI8NFMl8QaQlc7V%252FScreenshot%25202026-04-02%2520at%252010.37.32%25E2%2580%25AFPM.png%3Falt%3Dmedia%26token%3D18d5918e-4f71-4e0e-b8c9-464097389835&width=768&dpr=3&quality=100&sign=6dcafb02224917d3449a82f5064143ce&sv=3){width=1442 height=778}

#### Run Gemma 4 {#run-gemma-4}

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings. You can run GGUFs and MLX files.

For more information, you can view our [Unsloth Studio inference guide](https://unsloth.ai/docs/new/studio/chat).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FVrLgXwplAMcvkU4owjPk%252F26b%2520gif.gif%3Falt%3Dmedia%26token%3D8a569952-c152-435f-b815-c9f295619587&width=768&dpr=3&quality=100&sign=7fa1fbfd389701e0019e4820a0657ee0&sv=3){width=1280 height=1034}

### 🦙 Llama.cpp Guide {#llama.cpp-guide}

For this guide we will be utilizing Dynamic 4-bit for the 12B, 26B-A4B and 31B, and 8-bit for E2B and E4B. See: [Gemma 4 GGUF collection](https://huggingface.co/collections/unsloth/gemma-4)

For these tutorials, we will using [llama.cpp](llama.cpphttps://github.com/ggml-org/llama.cpp) for fast local inference, especially if you have a CPU.

Obtain the latest `llama.cpp` **on** [**GitHub here**](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

If you want to use `llama.cpp` directly to load models, you can follow commands below, according to each model. `UD-Q4_K_XL` is the quantization type. You can also download via Hugging Face (step 3). This is similar to `ollama run` . Use `export LLAMA_CACHE="folder"` to force `llama.cpp` to save to a specific location. There is no need to set context length as llama.cpp automatically uses the exact amount required.

To [disable thinking / reasoning](https://unsloth.ai/docs/models/gemma-4#how-to-enable-or-disable-reasoning-and-thinking), use: `--chat-template-kwargs '{"enable_thinking":false}'`

**Windows** Powershell: `--chat-template-kwargs "{\"enable_thinking\":false}"`

Use '`true`' and '`false`' interchangeably.

**12B:**

**26B-A4B:**

**31B:**

**E4B:**

**E2B:**

You can also download the model manually as well via the code below (after installing `pip install huggingface_hub`). You can choose `UD-Q4_K_XL` or other quantized versions like `Q8_0` . If downloads get stuck, see: [Hugging Face Hub, XET debugging](https://unsloth.ai/docs/basics/troubleshooting-and-faqs/hugging-face-hub-xet-debugging)

Then run the model in conversation mode (with vision `mmproj-F16`):

#### Llama-server deployment {#llama-server-deployment}

To deploy Gemma-4 on llama-server, use:

### MLX Dynamic Quants {#mlx-dynamic-quants}

We also uploaded dynamic 4bit and 8bit quants as a first trial for MacOS device! The MLX quants support **vision.**

To try them out use:

### **NVFP4 Guide** {#nvfp4-guide}

We uploaded [Dynamic NVFP4](https://unsloth.ai/docs/basics/nvfp4) Gemma 4 quants for faster 4-bit inference on NVIDIA Blackwell GPUs. are the hardware requirements for models which you can use including Gemma 4. Also see the overall speed boost you will achieve:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FZw91MhTauyOZGP5HT7bl%252Fimage.png%3Falt%3Dmedia%26token%3Dae0051a9-af39-4ef7-8ed2-9f8c1fff33fc&width=768&dpr=3&quality=100&sign=3f62a9bc3d97477caa2558c29a562b84&sv=3){width=3368 height=1218}

#### **vLLM Tutorial:** {#vllm-tutorial}

To run NVFP4 quants, see below for commands to run Gemma-4-26B-A4B in [vLLM](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide) and [SGLang](https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide) (you can change model name to `gemma-4-31B-it-NVFP4` etc.) Also do NOT select any MoE backend - leave vLLM to select it - for eg Marlin is 2.5x slower! See [Gemma 4](https://unsloth.ai/docs/models/gemma-4#marlin-vs-flashinfer-vs-cutlass-vs-cute-dsl)If you have a DGX Spark, see [Gemma 4](https://unsloth.ai/docs/models/gemma-4#dgx-spark-serving) you must use `--moe-backend flashinfer_b12x` or you will get much slower inference.

To install vLLM in a separate venv:

Then to serve the 26B MoE variant:

Change `unsloth/gemma-4-26B-A4B-it-NVFP4` to any [available NVFP4](https://unsloth.ai/docs/basics/nvfp4#overview) quant name!

To enable MTP / speculative decoding (faster decode but somewhat less throughput), use:

If you get Torchcodec issues, be sure to do the below then relaunch vllm.

#### **DGX Spark with NVFP4 quants** {#dgx-spark-with-nvfp4-quants}

To ensure DGX Spark has the correct kernels (or you will get **2x SLOWER inference**), first check:

which should NOT error out - if it did, please update vllm or reinstall via:

Then to serve in vLLM for DGX Spark:

If you get Torchcodec issues, be sure to do the below then relaunch vllm.

#### **SGLang Tutorial:** {#sglang-tutorial}

### Ollama Guide {#ollama-guide}

Ollama now supports Unsloth GGUFs well now. Use `curl -fsSL https://ollama.com/install.sh | sh` to install Ollama on Linux or `irm https://ollama.com/install.ps1 | iex` for Windows. To use a single quant file (under 50GB) use:

For multiple shards like larger BF16 shards do:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FjflAyMZg9ZaSmeUL7LlQ%252Fimage.png%3Falt%3Dmedia%26token%3Dddda55e7-0526-4fe0-99bc-d0cb4d05f7fa&width=768&dpr=3&quality=100&sign=b39ba82c3b171c5753abb48d85f26cc3&sv=3){width=2226 height=1054}

## Gemma 4 Best Practices {#gemma-4-best-practices}

### Prompting examples {#prompting-examples}

#### Simple reasoning prompt {#simple-reasoning-prompt}

#### OCR / document prompt {#ocr-document-prompt}

For OCR, use a **high visual token budget** like **560** or **1120**.

#### Multi-modal comparison prompt {#multi-modal-comparison-prompt}

#### Audio ASR prompt {#audio-asr-prompt}

#### Audio translation prompt {#audio-translation-prompt}

### Multi-modal Settings {#multi-modal-settings}

For best results with multimodal prompts, put multimodal content first:

- Put **image and/or audio before text**.
- For video, pass a sequence of frames first, then the instruction.

#### Audio and video limits {#audio-and-video-limits}

- **Audio** is available on **12B**, **E2B** and **E4B** only.
- Audio supports a maximum of **30 seconds**.
- Video supports a maximum of **60 seconds** assuming **1 frame per second** processing.

#### Audio prompt templates {#audio-prompt-templates}

**ASR prompt**

**Speech translation prompt**

## 📊 Benchmarks {#benchmarks}

### Unsloth GGUF Benchmarks {#unsloth-gguf-benchmarks}

We conducted Mean KL Divergence benchmarks for Gemma 4 GGUFs across providers to help you pick the best quant (lower is better). 

- KL Divergence puts all Unsloth GGUFs on the SOTA Pareto frontier 
- KLD shows how well a quantized model matches the original BF16 output distribution, indicating retained accuracy. 

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FtRVN97QDzO0Pq7SscC7x%252Fgemma%2520426b%2520bench.png%3Falt%3Dmedia%26token%3D80b4da76-efe9-4554-8e31-cca6494d456c&width=768&dpr=3&quality=100&sign=98f2ee7401325debc39c30aeebe70db1&sv=3){width=10153 height=5944} 26B A4B - KLD benchmarks (lower is better)

### Official Gemma Benchmarks {#official-gemma-benchmarks}

**Text/Code Benchmarks**

**Vision Benchmarks**

**Audio Benchmarks**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FfKaFMy7LHQYNKpfsf7Zy%252Fgemma%25204%2520banner.png%3Falt%3Dmedia%26token%3D8bd8d0e0-ccb6-4ded-b99b-2c8a18370ae5&width=768&dpr=3&quality=100&sign=dea7ab3d7585d31986b2f8d8a8c64eb8&sv=3){width=2560 height=900}
