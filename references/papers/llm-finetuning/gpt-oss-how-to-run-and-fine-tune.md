title: gpt-oss: How to Run Guide | Unsloth Documentation
description: Run & fine-tune OpenAI's new open-source models!

# gpt-oss: How to Run Guide | Unsloth Documentation

OpenAI releases '**gpt-oss-120b'** and '**gpt-oss-20b'**, two SOTA open language models under the Apache 2.0 license. Both 128k context models outperform similarly sized open models in reasoning, tool use, and agentic tasks. You can now run & fine-tune them locally with Unsloth!

[Run gpt-oss-20b](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-20b) [Run gpt-oss-120b](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-120b) [Fine-tune gpt-oss](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#fine-tuning-gpt-oss-with-unsloth)

> [**Fine-tune**](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#fine-tuning-gpt-oss-with-unsloth) **gpt-oss-20b for free with our** [**Colab notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb)

Trained with [RL](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), **gpt-oss-120b** rivals o4-mini and **gpt-oss-20b** rivals o3-mini. Both excel at function calling and CoT reasoning, surpassing o1 and GPT-4o.

For best performance, make sure your total available memory (unified mem \+ VRAM \+ system RAM) exceeds the size of the quantized model file you’re downloading. If it doesn’t, llama.cpp can still run via SSD/HDD offloading, but inference will be slower.

#### **gpt-oss - Unsloth GGUFs:** {#gpt-oss-unsloth-ggufs}

**Includes Unsloth's** [**chat template fixes**](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss) **. For best results, use our uploads & train with Unsloth!**

## 📜Unsloth fixes for gpt-oss {#unsloth-fixes-for-gpt-oss}

OpenAI released a standalone parsing and tokenization library called [Harmony](https://github.com/openai/harmony) which allows one to tokenize conversations to OpenAI's preferred format for gpt-oss.

Inference engines generally use the jinja chat template instead and not the Harmony package, and we found some issues with them after comparing with Harmony directly. If you see below, the top is the correct rendered form as from Harmony. The below is the one rendered by the current jinja chat template. There are quite a few differences!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-9b377044965ac55a125d6c703ec1c50555157266%252FScreenshot%25202025-08-08%2520at%252008-19-49%2520Untitled151.ipynb%2520-%2520Colab.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=4960ad0256d5a615d003711a2237a4dc&sv=3){width=2650 height=930}

We also made some functions to directly allow you to use OpenAI's Harmony library directly without a jinja chat template if you desire - you can simply parse in normal conversations like below:

Then use the `encode_conversations_with_harmony` function from Unsloth:

The harmony format includes multiple interesting things:

1. `reasoning_effort = "medium"` You can select low, medium or high, and this changes gpt-oss's reasoning budget - generally the higher the better the accuracy of the model.
2. `developer_instructions` is like a system prompt which you can add.
3. `model_identity` is best left alone - you can edit it, but we're unsure if custom ones will function.

We find multiple issues with current jinja chat templates (there exists multiple implementations across the ecosystem):

1. Function and tool calls are rendered with `tojson`, which is fine it's a dict, but if it's a string, speech marks and other **symbols become backslashed**.
2. There are some **extra new lines** in the jinja template on some boundaries.
3. Tool calling thoughts from the model should have the `analysis`  **tag and not**  `final`  **tag**.
4. Other chat templates seem to not utilize `<|channel|>final` at all - one should use this for the final assistant message. You should not use this for thinking traces or tool calls.

Our chat templates for the GGUF, our BnB and BF16 uploads and all versions are fixed! For example when comparing both ours and Harmony's format, we get no different characters:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4c42f3d83194ea2fbe436670a550e1b6f148f4cd%252FScreenshot%25202025-08-08%2520at%252008-20-00%2520Untitled151.ipynb%2520-%2520Colab.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=26aa7c8c84eb898d07b45dfac3e50807&sv=3){width=2650 height=930}

### 🔢 Precision issues {#precision-issues}

We found multiple precision issues in Tesla T4 and float16 machines primarily since the model was trained using BF16, and so outliers and overflows existed. MXFP4 is not actually supported on Ampere and older GPUs, so Triton provides `tl.dot_scaled` for MXFP4 matrix multiplication. It upcasts the matrices to BF16 internally on the fly.

We made a [MXFP4 inference notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_%2820B%29-Inference.ipynb) as well in Tesla T4 Colab!

We found if you use float16 as the mixed precision autocast data-type, you will get infinities after some time. To counteract this, we found doing the MoE in bfloat16, then leaving it in either bfloat16 or float32 precision. If older GPUs don't even have bfloat16 support (like T4), then float32 is used.

We also change all precisions of operations (like the router) to float32 for float16 machines.

## 🖥️ **Running gpt-oss** {#running-gpt-oss}

Below are guides for the [20B](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-20b) and [120B](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-120b) variants of the model.

The `gpt-oss` models from OpenAI include a feature that allows users to adjust the model's "reasoning effort." This gives you control over the trade-off between the model's performance and its response speed (latency) which by the amount of token the model will use to think.

The `gpt-oss` models offer three distinct levels of reasoning effort you can choose from:

- **Low**: Optimized for tasks that need very fast responses and don't require complex, multi-step reasoning.
- **Medium**: A balance between performance and speed.
- **High**: Provides the strongest reasoning performance for tasks that require it, though this results in higher latency.

### ⚙️ Recommended Settings {#recommended-settings}

OpenAI recommends these inference settings for both models:

`temperature=1.0`, `top_p=1.0`, `top_k=0`

- **Temperature of 1.0**
- Top\_K \= 0 (or experiment with 100 for possible better results)
- Top\_P \= 1.0
- Recommended minimum context: 16,384
- Maximum context length window: 131,072

**Chat template:**

The end of sentence/generation token: EOS is `<|return|>`

### Run gpt-oss-20B {#run-gpt-oss-20b}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-920b641670a166258845bbe8152999983b1e68af%252Fgpt-oss-20b.svg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=f77afe54012a9afa31cf69e60671f5bd&sv=3){width=1000 height=200}

To achieve inference speeds of 6\+ tokens per second for our Dynamic 4-bit quant, have at least **14GB of unified memory** (combined VRAM and RAM) or **14GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. GGUF Link: [unsloth/gpt-oss-20b-GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.

You can run the model on Google Colab, Docker, LM Studio or llama.cpp for now. See below:

> **You can run gpt-oss-20b for free with our** [**Google Colab notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_%2820B%29-Inference.ipynb)

#### 🦥 Unsloth Studio Guide {#unsloth-studio-guide}

For this tutorial, we will be using [Unsloth Studio](https://docs.unsloth.ai/docs/new/studio), which is our new web UI for running and training LLMs. With Unsloth Studio, you can run models locally on **Mac, Windows**, and Linux and:

- Search, download, [run GGUFs](https://docs.unsloth.ai/docs/new/studio#run-models-locally) and safetensor models
- **Compare** models **side-by-side**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FFeQ0UUlnjXkDdqhcWglh%252Fskinny%2520studio%2520chat.png%3Falt%3Dmedia%26token%3Dc2ee045f-c243-4024-a8e4-bb4dbe7bae79&width=768&dpr=3&quality=100&sign=3913590d30f444735fdee2650732ae17&sv=3){width=1615 height=1178}

#### Install Unsloth {#install-unsloth}

Run in your terminal:

**MacOS, Linux, WSL:**

**Windows PowerShell:**

#### Launch Unsloth {#launch-unsloth}

**MacOS, Linux, WSL, Windows:**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fd1yMMNa65Ccz50Ke0E7r%252FScreenshot%25202026-03-17%2520at%252012.32.38%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D9369cfe7-35b1-4955-b8cb-42f7ecb43780&width=768&dpr=3&quality=100&sign=205b569c15fb8423b66ae9895a3d20a8&sv=3){width=2738 height=1328}

**Then open**  `http://localhost:8888`  **in your browser.**

#### Search and download gpt-oss-20b {#search-and-download-gpt-oss-20b}

On first launch you will need to create a password to secure your account and sign in again later. You’ll then see a brief onboarding wizard to choose a model, dataset, and basic settings. You can skip it at any time.

Then go to the [Unsloth Chat](https://docs.unsloth.ai/docs/new/studio/chat) tab and search for gpt-oss in the search bar and download your desired model and quant.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F24OqTOq1fYvW3oZ5WlNG%252FScreenshot%25202026-03-20%2520at%25201.35.19%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3Db1e463ee-33a2-4e77-b725-d715b03f5d28&width=768&dpr=3&quality=100&sign=782dac8176eee119c120386f970a40d4&sv=3){width=1680 height=1132}

#### Run gpt-oss-20b {#run-gpt-oss-20b-1}

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.

For more information, you can view our [Unsloth Studio inference guide](https://docs.unsloth.ai/docs/new/studio/chat).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FXPQGEEr1YoKofrTatAKK%252Ftoolcallingif.gif%3Falt%3Dmedia%26token%3D25d68698-fb13-4c46-99b2-d39fb025df08&width=768&dpr=3&quality=100&sign=803722bc09915f7a6a588bff82625529&sv=3){width=800 height=551}

#### 🐋 Docker: Run gpt-oss-20b Tutorial {#docker-run-gpt-oss-20b-tutorial}

If you already have Docker desktop, all you need to do is run the command below and you're done:

#### ✨ Llama.cpp: Run gpt-oss-20b Tutorial {#llama.cpp-run-gpt-oss-20b-tutorial}

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

1. You can directly pull from Hugging Face via:

### Run gpt-oss-120b: {#run-gpt-oss-120b}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4f6fc3b98363be32b7c7cf07c713947cb1bd9444%252Fgpt-oss-120b.svg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=619abe9a0ac50c2227f65a15582fb141&sv=3){width=1000 height=200}

To achieve inference speeds of 6\+ tokens per second for our 1-bit quant, we recommend at least **66GB of unified memory** (combined VRAM and RAM) or **66GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. GGUF Link: [unsloth/gpt-oss-120b-GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.

#### 🦥 Unsloth Studio Guide {#unsloth-studio-guide-1}

For this tutorial, we will be using [Unsloth Studio](https://docs.unsloth.ai/docs/new/studio), which is our new web UI for running and training LLMs. With Unsloth Studio, you can run models locally on **Mac, Windows**, and Linux and:

- Search, download, [run GGUFs](https://docs.unsloth.ai/docs/new/studio#run-models-locally) and safetensor models
- **Compare** models **side-by-side**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FFeQ0UUlnjXkDdqhcWglh%252Fskinny%2520studio%2520chat.png%3Falt%3Dmedia%26token%3Dc2ee045f-c243-4024-a8e4-bb4dbe7bae79&width=768&dpr=3&quality=100&sign=3913590d30f444735fdee2650732ae17&sv=3){width=1615 height=1178}

#### Install Unsloth {#install-unsloth-1}

**MacOS, Linux, WSL:**

**Windows PowerShell:**

#### Setup Unsloth Studio (one time) {#setup-unsloth-studio-one-time}

Setup automatically installs Node.js (via nvm), builds the frontend, installs all Python dependencies, and builds llama.cpp with CUDA support.

#### Launch Unsloth {#launch-unsloth-1}

**MacOS, Linux, WSL:**

**Windows Powershell:**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fd1yMMNa65Ccz50Ke0E7r%252FScreenshot%25202026-03-17%2520at%252012.32.38%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D9369cfe7-35b1-4955-b8cb-42f7ecb43780&width=768&dpr=3&quality=100&sign=205b569c15fb8423b66ae9895a3d20a8&sv=3){width=2738 height=1328}

**Then open**  `http://localhost:8888`  **in your browser.**

#### Search and download gpt-oss-120b {#search-and-download-gpt-oss-120b}

On first launch you will need to create a password to secure your account and sign in again later. You’ll then see a brief onboarding wizard to choose a model, dataset, and basic settings. You can skip it at any time.

Then go to the [Unsloth Chat](https://docs.unsloth.ai/docs/new/studio/chat) tab and search for gpt-oss in the search bar and download your desired model and quant.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FgKxB8OhIRVgT29vCGFEg%252FScreenshot%25202026-03-20%2520at%25201.34.07%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D9950d667-4ff1-463e-b017-601d5e8e38a5&width=768&dpr=3&quality=100&sign=6cc224540257641d8a7e73f5fe5485ca&sv=3){width=1686 height=1128}

#### Run gpt-oss-120b {#run-gpt-oss-120b-1}

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.

For more information, you can view our [Unsloth Studio inference guide](https://docs.unsloth.ai/docs/new/studio/chat).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FXPQGEEr1YoKofrTatAKK%252Ftoolcallingif.gif%3Falt%3Dmedia%26token%3D25d68698-fb13-4c46-99b2-d39fb025df08&width=768&dpr=3&quality=100&sign=803722bc09915f7a6a588bff82625529&sv=3){width=800 height=551}

#### 📖 Llama.cpp: Run gpt-oss-120b Tutorial {#llama.cpp-run-gpt-oss-120b-tutorial}

For gpt-oss-120b, we will specifically use Llama.cpp for optimized inference.

If you want a **full precision unquantized version**, use our `F16` versions!

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.
2. You can directly use llama.cpp to download the model but I normally suggest using `huggingface_hub` To use llama.cpp directly, do:
3. Or, download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..
4. Run the model in conversation mode and try any prompt.
5. Edit `--threads -1` for the number of CPU threads, `--ctx-size` 262114 for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity. More options discussed [here](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#improving-generation-speed).

### 🛠️ Improving generation speed {#improving-generation-speed}

If you have more VRAM, you can try offloading more MoE layers, or offloading whole layers themselves.

Normally, `-ot ".ffn_.*_exps.=CPU"` offloads all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.

The [latest llama.cpp release](https://github.com/ggml-org/llama.cpp/pull/14363) also introduces high throughput mode. Use `llama-parallel`. Read more about it [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/parallel). You can also **quantize the KV cache to 4bits** for example to reduce VRAM / RAM movement, which can also make the generation process faster.

## 🦥 Fine-tuning gpt-oss with Unsloth {#fine-tuning-gpt-oss-with-unsloth}

[**Aug 28 update**](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training#new-saving-to-gguf-vllm-after-gpt-oss-training) **:** You can now export/save your QLoRA fine-tuned gpt-oss model to llama.cpp, vLLM, HF etc.

We also introduced [Unsloth Flex Attention](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training#introducing-unsloth-flex-attention-support) which enables **>8× longer context lengths**, **>50% less VRAM usage** and **>1.5× faster training** vs. all implementations. [Read more here](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training#introducing-unsloth-flex-attention-support)

Unsloth gpt-oss fine-tuning is 1.5x faster, uses 70% less VRAM, and supports 10x longer context lengths. gpt-oss-20b QLoRA training fits on a 14GB VRAM, and gpt-oss-120b works on 65GB VRAM.

- **QLoRA requirements:** gpt-oss-20b \= 14GB VRAM • gpt-oss-120b \= 65GB VRAM.
- **BF16 LoRA requirements:** gpt-oss-20b \= 44GB VRAM • gpt-oss-120b \= 210GB VRAM.

Read our step-by-step tutorial for fine-tuning gpt-oss:

[Tutorial: How to Fine-tune gpt-oss](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)

You can now export/save your QLoRA fine-tuned gpt-oss model to llama.cpp, vLLM, HF etc.

Free Unsloth notebooks to fine-tune gpt-oss:

### Reinforcement Learning (GRPO) {#reinforcement-learning-grpo}

Unsloth now supports RL for gpt-oss! We made two notebooks, for more details, read our specific blog for gpt-oss RL: [gpt-oss RL](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning)

### 💾**NEW: Saving to GGUF, vLLM after gpt-oss training** {#new-saving-to-gguf-vllm-after-gpt-oss-training}

You can now QLoRA fine-tune gpt-oss and directly save, export, or merge the model to **llama.cpp**, **vLLM**, or **HF** - not just Unsloth. We will be releasing a free notebook hopefully soon.

Previously, any QLoRA fine-tuned gpt-oss model was restricted to running in Unsloth. We’ve removed that limitation by introducing **on-demand dequantization of MXFP4** base models (like gpt-oss) during the LoRA merge process. This makes it possible to **export your fine-tuned model in bf16 format**.

After fine-tuning your gpt-oss model, you can now merge it into a 16-bit format with a **single command**:

If you prefer to merge the model and push to the hugging-face hub directly instead, you could do so using:

### 💡Making efficient gpt-oss fine-tuning work {#making-efficient-gpt-oss-fine-tuning-work}

We found that while MXFP4 is highly efficient, it does not natively support training with gpt-oss. To overcome this limitation, we implemented custom training functions specifically for MXFP4 layers through mimicking it via `Bitsandbytes` NF4 quantization.

We utilized OpenAI's Triton Kernels library directly to allow MXFP4 inference. For finetuning / training however, the MXFP4 kernels do not yet support training, since the backwards pass is not yet implemented. We're actively working on implementing it in Triton! There is a flag called `W_TRANSPOSE` as mentioned [here](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py#L39), which should be implemented. The derivative can be calculated by the transpose of the weight matrices, and so we have to implement the transpose operation.

If you want to train gpt-oss with any library other than Unsloth, you’ll need to upcast the weights to bf16 before training. This approach, however, **significantly increases** both VRAM usage and training time by as much as **300% more memory usage**! **ALL other training methods will require a minimum of 65GB VRAM to train the 20b model while Unsloth only requires 14GB VRAM (-80%).**

As both models use MoE architecture, the 20B model selects 4 experts out of 32, while the 120B model selects 4 out of 128 per token. During training and release, weights are stored in MXFP4 format as `nn.Parameter` objects, not as `nn.Linear` layers, which complicates quantization, especially since MoE/MLP experts make up about 19B of the 20B parameters.

To enable `BitsandBytes` quantization and memory-efficient fine-tuning, we converted these parameters into `nn.Linear` layers. Although this slightly slows down operations, it allows fine-tuning on GPUs with limited memory, a worthwhile trade-off.

### Datasets fine-tuning guide {#datasets-fine-tuning-guide}

Though gpt-oss supports only reasoning, you can still fine-tune it with a non-reasoning [dataset](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide), but this may affect its reasoning ability. If you want to maintain its reasoning capabilities (optional), you can use a mix of direct answers and chain-of-thought examples. Use at least 75% reasoning and 25% non-reasoning in your dataset to make the model retain its reasoning capabilities.

Our gpt-oss-20b Conversational notebook uses OpenAI's example which is Hugging Face's Multilingual-Thinking dataset. The purpose of using this dataset is to enable the model to learn and develop reasoning capabilities in these four distinct languages.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4d648159c0ba6d62d5c9b5cd519767f764e5faab%252Fwider%2520gptoss%2520image.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=601ea27f7e3738ddae465eab9055212a&sv=3){width=2560 height=963}
