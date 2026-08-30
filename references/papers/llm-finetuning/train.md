title: Gemma 4 Fine-tuning Guide | Unsloth Documentation
description: Train Gemma 4 by Google with Unsloth.

# Gemma 4 Fine-tuning Guide | Unsloth Documentation

You can now train Google's [Gemma 4](https://unsloth.ai/docs/models/gemma-4) 12B, E2B, E4B, 26B-A4B and 31B with [**Unsloth**](https://github.com/unslothai/unsloth). Unsloth supports all vision, text, audio and RL fine-tuning for Gemma 4.

- Unsloth trains Gemma 4 **\~1.5x faster** with **\~60% less VRAM** than FA2 setups (no accuracy loss)
- Gemma 4 E2B trains on **8GB VRAM**. E4B requires 10GB VRAM.

[Quickstart](https://unsloth.ai/docs/models/gemma-4/train#quickstart) [Bug Fixes \+ Tips](https://unsloth.ai/docs/models/gemma-4/train#bug-fixes--tips)

Fine-tune Gemma 4 via our **free** **Google Colab notebooks**:

- Gemma 4 E2B LoRA works on 8-10GB VRAM. E4B LoRA requires 17GB VRAM.
- **31B QLoRA works with 22GB** and 26B-A4B LoRA needs >40GB
- **Exporting**/saving models to GGUF etc. and full fine-tuning **(FFT)** works as well.

### 🐛 Bug fixes \+ Tips {#bug-fixes--tips}

If you see **Gemma-4 E2B and E4B having a loss of 13-15, this is perfectly normal** - this is a common quirk of multimodal models. This also happened on Gemma-3N, Llama Vision, Mistral vision models and more.

**Gemma 26B and 31B have lower loss at 1-3 or lower. Vision will be 2x higher so 3-5**

#### 🍇Gradient accumulation might inflate your losses {#gradient-accumulation-might-inflate-your-losses}

If you see losses higher than 13-15 (like 100 or 300) most likely gradient accumulation is not being accounted properly - we have **fixed this as part of Unsloth and Unsloth Studio.**

To read more about gradient accumulation see our gradient accumulation bug fix blog: [https://unsloth.ai/blog/gradient](https://unsloth.ai/blog/gradient)

#### ⁉️IndexError on Gemma-4 31B and 26B-A4B inference {#indexerror-on-gemma-4-31b-and-26b-a4b-inference}

You might see this error when doing inference with 31B and 26B:

The culprit is below:

Where Gemma-4 31B and 26B-A4B ship with `num_kv_shared_layers = 0`. In Python, `-0 == 0`, so `layer_types[:-0]` collapses to `layer_types[:0] == []`. The cache is built with zero layer slots and the very first attention forward crashes inside `Cache.update`.

#### ⛔ `use_cache = True` generation was gibberish for E2B, E4B {#use_cache-true-generation-was-gibberish-for-e2b-e4b}

[See issue](https://github.com/huggingface/transformers/issues/45242) "\[Gemma 4\] `use_cache=False` corrupts attention computation, producing garbage logits #45242"

Gemma-4 E2B and E4B share KV state across layers (`num_kv_shared_layers = 20` and `18`). The cache is the only place where early layers stash KV for later layers to reuse. When `use_cache=False` (as every QLoRA tutorial sets, and as `gradient_checkpointing=True` forces), `Gemma4TextModel.forward` skips cache construction, so the KV-shared layers fall through to recomputing K and V locally from the current hidden states. The logits become garbage and training loss diverges.

**Before (** `unsloth/gemma-4-E2B-it` **, prompt "What is 1\+1?"):**

**After our fix:**

#### 📻Audio float16 overflow {#audio-float16-overflow}

`Gemma4AudioAttention` uses `config.attention_invalid_logits_value = -1e9` in a `masked_fill` call. On fp16 (Tesla T4), -1e9 overflows the fp16 max of 65504, causing:

This was due to `self.config.attention_invalid_logits_value` :

#### 💡Tips for Gemma-4 {#tips-for-gemma-4}

1. If you want to **preserve reasoning** ability, you can mix reasoning-style examples with direct answers (keep a minimum of 75% reasoning). Otherwise you can emit it fully. Use `gemma-4` for the non thinking chat-template and `gemma-4-thinking` for the thinking variant. Use the thinking one for the larger 26B and 31B ones, and the non thinking one for the small ones. 
2. 
To enable thinking mode, use `enable_thinking = True / False` in `tokenizer.apply_chat_template
`Thinking enabled:Will print `<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n
`Thinking disabled:Will print `<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`3. Gemma 4 is powerful for multilingual fine-tuning as it supports 140 languages.
4. It is recommended to train **E4B QLoRA** rather than **E2B LoRA** as the E4B is bigger and the quantization accuracy difference is miniscule. Gemma 4 E4B LoRA is even better.
5. After fine-tuning, you can export to [GGUF](https://unsloth.ai/docs/models/gemma-4/train#saving-export-your-fine-tuned-model) (for llama.cpp/Unsloth/Ollama/etc.)

### ⚡Quickstart {#quickstart}

#### 🦥 Unsloth Studio Guide {#unsloth-studio-guide}

Gemma 4 can be run and fine-tuned in [Unsloth Studio](https://unsloth.ai/docs/new/studio), our new open-source web UI for local AI.

With Unsloth Studio, you can run models locally on **MacOS, Windows**, Linux and train NVIDIA GPUs. Intel, MLX and AMD training support coming this month.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FpZlhqoILYOzznGpbudUk%252Funsloth%2520studio%2520gemma%2520graphic.png%3Falt%3Dmedia%26token%3D75e41585-e363-45cf-a87e-4d02960766ed&width=768&dpr=3&quality=100&sign=5ee76ba21854e7b39a37eca480e61807&sv=3){width=2288 height=1200}

#### Install Unsloth {#install-unsloth}

Run in your terminal:

**MacOS, Linux, WSL:**

**Windows PowerShell:**

**Installation will be quick and take approx 1-2 mins.**

#### Launch Unsloth {#launch-unsloth}

**MacOS, Linux, WSL and Windows:**

**Then open**  `http://localhost:8888`  **in your browser.**

#### Train Gemma 4 {#train-gemma-4}

On first launch you will need to create a password to secure your account and sign in again later. You’ll then see a brief onboarding wizard to choose a model, dataset, and basic settings. You can skip it at any time.

Search for Gemma 4 in the search bar and select your desired model and dataset. Next, adjust your hyperparameters, context length as desired.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FpZlhqoILYOzznGpbudUk%252Funsloth%2520studio%2520gemma%2520graphic.png%3Falt%3Dmedia%26token%3D75e41585-e363-45cf-a87e-4d02960766ed&width=768&dpr=3&quality=100&sign=5ee76ba21854e7b39a37eca480e61807&sv=3){width=2288 height=1200}

#### Monitor training progress {#monitor-training-progress}

After you click start training, you will be able to monitor and observe the training progress of the model. The training loss should be steadily decreasing. Once done, the model will be automatically saved.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FeBrnu9zxARIkhOHzd0pq%252FScreenshot%25202026-04-07%2520at%25205.53.32%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3Ddae77231-5020-4e8c-b2b8-cc49a98a9edf&width=768&dpr=3&quality=100&sign=2987aa0bd1f1e06c4ee3348b909a7192&sv=3){width=2382 height=1314}

#### Export your fine-tuned model {#export-your-fine-tuned-model}

Once done, Unsloth Studio allows you to export the model to GGUF, safetensor etc formats.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FBtpx58zCdrOD4zB4DPSC%252FScreenshot%25202026-04-07%2520at%25206.12.41%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D05f05af2-5f7f-4b91-9c99-21d6a9b04935&width=768&dpr=3&quality=100&sign=b36dcc0e9217926c1037f0b7153a67fb&sv=3){width=2294 height=1352}

#### Compare fine-tuned model vs original model {#compare-fine-tuned-model-vs-original-model}

Click on `Compare Mode` to compare the LoRA adapter and the original model.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fvm2CBSg7QBkKwTMKutyr%252FScreenshot%25202026-04-07%2520at%25206.14.50%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D8c9c159f-9d5b-4468-8984-681d19ebc427&width=768&dpr=3&quality=100&sign=ecdc00ca4046a1dce16257cc16ac9a78&sv=3){width=2276 height=1418}

#### 🦥 Unsloth Core (code-based) Guide {#unsloth-core-code-based-guide}

We made free notebooks for Gemma 4:

And for reinforcement learning (RL): [E2B **(RL GRPO)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_%28E2B%29_Reinforcement_Learning_Sudoku_Game.ipynb)

We also made notebooks for the larger Gemma 4 models but they need A100:

Below is a standalone Gemma-4-26B-A4B-it text SFT recipe. This is text only - see also our [vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning) section for more details.

**Loader example for MoE (bf16 LoRA):**

Once loaded, you’ll attach LoRA adapters and train similarly to the SFT example above.

### Reinforcement Learning (RL) {#reinforcement-learning-rl}

You can now train Gemma 4 with RL, GSPO, GRPO etc with [our free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_%284B%29_Vision_GRPO.ipynb).

Gemma 4 E2B RL works on 9GB.

The notebook's goal is to make Gemma 4 learn to solve Sudoku puzzles using [GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#from-rlhf-ppo-to-grpo-and-rlvr).

The model will devise a strategy to fill in empty cells, and we'll reward it for correct placements and completing valid puzzles.

You can run Gemma 4 RL with Unsloth even though it is not supported by vLLM, by setting `fast_inference=False` when loading the model:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fw08jKlXDLJji37JZPYNx%252Fgemma%25204%2520rl%2520sodoku%2520nb.png%3Falt%3Dmedia%26token%3D379c607d-60b1-4b35-bcdd-bf6a86911f85&width=768&dpr=3&quality=100&sign=c3029995f041f3dc999914f23554a435&sv=3){width=2108 height=2281}

### MoE fine-tuning (26B-A4B) {#moe-fine-tuning-26b-a4b}

The **26B-A4B** model is the speed / quality middle ground in the Gemma 4 lineup. Since it is an **MoE** model with only a subset of parameters active per token, a conservative fine-tuning approach is:

- use **LoRA** rather than full fine-tuning
- prefer **16-bit / bf16 LoRA** if memory allows
- start with shorter contexts and smaller ranks first
- scale up only after the pipeline is stable

If your goal is the highest quality and you have more memory, use **31B** instead.

### Multimodal fine-tuning (E2B / E4B) {#multimodal-fine-tuning-e2b-e4b}

Because **E2B** and **E4B** support **image** and **audio**, they are the main Gemma 4 variants for multimodal fine-tuning.

- load the multimodal model with `FastVisionModel`
- keep `finetune_vision_layers = False` first
- fine-tune only the language, attention, and MLP layers
- enable vision or audio layers later if your task needs it

#### Gemma 4 Multimodal LoRA example: {#gemma-4-multimodal-lora-example}

#### Image example format {#image-example-format}

Remember: for Gemma 4 multimodal prompts, put the image **before** the text instruction.

#### Audio example format {#audio-example-format}

Audio is for **E2B / E4B** only. Keep clips short and task-specific.

### Saving / export fine-tuned model {#saving-export-fine-tuned-model}

You can view our specific inference / deployment guides for [Unsloth Studio](https://unsloth.ai/docs/new/studio/export), [llama.cpp](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [vLLM](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide), [llama-server](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint), [Ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama) or [SGLang](https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide).

#### Save to GGUF {#save-to-gguf}

Unsloth supports saving directly to GGUF:

Or push GGUFs to Hugging Face:

If the exported model behaves worse in another runtime, Unsloth flags the most common cause: **wrong chat template / EOS token at inference time** (you must use the same chat template you trained with).

For more details read our inference guides:

### Gemma 4 data best practices {#gemma-4-data-best-practices}

Gemma 4 has a few formatting details you need to keep in mind.

#### 1. Use standard chat roles {#id-1.-use-standard-chat-roles}

Gemma 4 uses the standard:

- `system`
- `user`
- `assistant`

This means your SFT dataset should be written in regular chat format rather than older Gemma-specific role formats.

#### 2. Thinking mode is explicit {#id-2.-thinking-mode-is-explicit}

If you want to preserve thinking-style behavior during SFT:

- keep the format consistent
- decide whether you want to train on **visible thought blocks** or on **final answers only**
- do **not** mix multiple incompatible thought formats in the same dataset

For most production assistants, the simplest setup is to fine-tune on the **final visible answer only**.

#### 3. Multi-turn rule {#id-3.-multi-turn-rule}

For multi-turn conversations, only keep the **final visible answer** in the conversation history. Do **not** feed earlier thought blocks back into later turns.
