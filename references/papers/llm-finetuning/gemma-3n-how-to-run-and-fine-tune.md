title: Gemma 3n: How to Run & Fine-tune | Unsloth Documentation
description: Run Google's new Gemma 3n locally with Dynamic GGUFs on llama.cpp, Ollama, Open WebUI and fine-tune with Unsloth!

# Gemma 3n: How to Run & Fine-tune | Unsloth Documentation

Google’s Gemma 3n multimodal model handles image, audio, video, and text inputs. Available in 2B and 4B sizes, it supports 140 languages for text and multimodal tasks. You can now run and fine-tune **Gemma-3n-E4B** and **E2B** locally using [Unsloth](https://github.com/unslothai/unsloth).

> **Fine-tune Gemma 3n with our** [**free Colab notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_%284B%29-Conversational.ipynb)

Gemma 3n has **32K context length**, 30s audio input, OCR, auto speech recognition (ASR), and speech translation via prompts.

[Running Tutorial](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune#running-gemma-3n) [Fine-tuning Tutorial](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune#fine-tuning-gemma-3n-with-unsloth) [Fixes \+ Technical Analysis](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune#fixes-for-gemma-3n)

**Unsloth Gemma 3n (Instruct) uploads with optimal configs:**

**See all our Gemma 3n uploads including base and more formats in** [**our collection here**](https://huggingface.co/collections/unsloth/gemma-3n-685d3874830e49e1c93f9339) **.**

## 🖥️ Running Gemma 3n {#running-gemma-3n}

Currently Gemma 3n is only supported in **text format** for inference.

### ⚙️ Official Recommended Settings {#official-recommended-settings}

According to the Gemma team, the official recommended settings for inference:

`temperature = 1.0, top_k = 64, top_p = 0.95, min_p = 0.0`

- Temperature of 1.0
- Top\_K of 64
- Min\_P of 0.00 (optional, but 0.01 works well, llama.cpp default is 0.1)
- Top\_P of 0.95
- Repetition Penalty of 1.0. (1.0 means disabled in llama.cpp and transformers)
- 
Chat template:- Chat template with `\n`newlines rendered (except for the last)

llama.cpp an other inference engines auto add a \<bos> - DO NOT add TWO \<bos> tokens! You should ignore the \<bos> when prompting the model!

### 🦙 Tutorial: How to Run Gemma 3n in Ollama {#tutorial-how-to-run-gemma-3n-in-ollama}

Please re download Gemma 3N quants or remove the old ones via Ollama since there are some bug fixes. You can do the below to delete the old file and refresh it:

1. Install `ollama` if you haven't already!

1. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

### 📖 Tutorial: How to Run Gemma 3n in llama.cpp {#tutorial-how-to-run-gemma-3n-in-llama.cpp}

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

1. If you want to use `llama.cpp` directly to load models, you can do the below: (:Q4\_K\_XL) is the quantization type. You can also download via Hugging Face (point 3). This is similar to `ollama run`

1. **OR** download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose Q4\_K\_M, or other quantized versions (like BF16 full precision).

1. Run the model.
2. Edit `--threads 32` for the number of CPU threads, `--ctx-size 32768` for context length (Gemma 3 supports 32K context length!), `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.
3. For conversation mode:

1. For non conversation mode to test Flappy Bird:

Remember to remove \<bos> since Gemma 3N auto adds a \<bos>!

## 🦥 Fine-tuning Gemma 3n with Unsloth {#fine-tuning-gemma-3n-with-unsloth}

Gemma 3n, like [Gemma 3](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes-for-gemma-3), had issues running on **Flotat16 GPUs such as Tesla T4s in Colab**. You will encounter NaNs and infinities if you do not patch Gemma 3n for inference or finetuning. [More information below](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune#infinities-and-nan-gradients-and-activations).

We also found that because Gemma 3n's unique architecture reuses hidden states in the vision encoder it poses another interesting quirk with [Gradient Checkpointing described below](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune#gradient-checkpointing-issues)

**Unsloth is the only framework which works in float16 machines for Gemma 3n inference and training.** This means Colab Notebooks with free Tesla T4 GPUs also work! Overall, Unsloth makes Gemma 3n training 1.5x faster, 50% less VRAM and 4x longer context lengths.

Our free Gemma 3n Colab notebooks default to fine-tuning text layers. If you want to fine-tune vision or audio layers too, be aware this will require much more VRAM - beyond the 15GB free Colab or Kaggle provides. You *can* still fine-tune all layers including audio and vision and Unsloth also lets you fine-tune only specific areas, like just vision. Simply adjust as needed:

#### 🏆Bonus Content {#bonus-content}

We also heard you guys wanted a **Vision notebook for Gemma 3 (4B)** so here it is:

## 🐛Fixes for Gemma 3n {#fixes-for-gemma-3n}

### ✨GGUF issues & fixes {#gguf-issues-and-fixes}

Thanks to discussions from [Michael](https://github.com/mxyng) from the Ollama team and also [Xuan](https://x.com/ngxson) from Hugging Face, there were 2 issues we had to fix specifically for GGUFs:

1. The `add_shared_kv_layers` parameter was accidentally encoded in `float32` which is fine, but becomes slightly complicated to decode on Ollama's side - a simple change to `uint32` solves the issue. [Pull request](https://github.com/ggml-org/llama.cpp/pull/14450) addressing this issue.
2. 
The `per_layer_token_embd` layer should be Q8\_0 in precision. Anything lower does not function properly and errors out in the Ollama engine - to reduce issues for our community, we made this all Q8\_0 in all quants - unfortunately this does use more space.As an [update](https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/discussions/4), [Matt](https://huggingface.co/WBB2500) mentioned we can also use Q4\_0, Q4\_1, Q5\_0, Q5\_1 for the embeddings - and we confirmed it does also work in Ollama! This means once again the smaller 2, 3 and 4bit quants are smaller in size, and don't need Q8\_0!
## ♾️Infinities and NaN gradients and activations {#infinities-and-nan-gradients-and-activations}

Gemma 3n just like Gemma 3 has issues on FP16 GPUs (e.g., Tesla T4s in Colab).

Our previous fixes for Gemma 3 is [discussed here](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune). For Gemma 3, we found that activations exceed float16's maximum range of **65504.**

**Gemma 3N does not have this activation issue, but we still managed to encounter infinities!**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-3f1aa0661c7919f8ad830fcbdf85a074d6a54bdf%252FGemma%25203%2520activation.webp%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=c0e1d84943ec2c00b18a073bca3b218b&sv=3){width=2304 height=1144}

To get to the bottom of these infinities, we plotted the absolute maximum weight entries for Gemma 3N, and we see the below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e9b01a20a0a1cfcc41ef47cb29c5188d52d6a79d%252Foutput2.webp%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=e38b3707ce8b09434497e677ca9ddca6&sv=3){width=1888 height=937}

We find that the green crosses are the Conv2D convolutional weights. We can see that the magnitude of Conv2D layers is much larger on average.

Below is a table for Conv2D weights which have large magnitudes. Our hypothesis is that during a Conv2D operation, large weights multiply and sum together, and **unfortunately by chance exceed float16's maximum range of 65504.** Bfloat16 is fine, since it's maximum range is 10\^38.

### 🎇Solution to infinities {#solution-to-infinities}

The naive solution is to `upcast` all Conv2D weights to float32 (if bfloat16 isn't available). But that would increase VRAM usage. To tackle this, we instead make use of `autocast` on the fly to upcast the weights and inputs to float32, and so we perform the accumulation in float32 as part of the matrix multiplication itself, without having to upcast the weights.

Unsloth is the only framework that enables Gemma 3n inference and training on float16 GPUs, so Colab Notebooks with free Tesla T4s work!

### 🏁Gradient Checkpointing issues {#gradient-checkpointing-issues}

We found Gemma 3N's vision encoder to be quite unique as well since it re-uses hidden states. This unfortunately limits the usage of [Unsloth's gradient checkpointing](https://unsloth.ai/blog/long-context), which could have reduced VRAM usage significantly. since it cannot be applied to Vision encoder.

However, we still managed to leverage **Unsloth's automatic compiler** to optimize Gemma 3N!

### 🌵Large losses during finetuning {#large-losses-during-finetuning}

We also found losses are interestingly very large during the start of finetuning - in the range of 6 to 7, but they do decrease over time quickly. We theorize this is either because of 2 possibilities:

1. There might be some implementation issue, but this is unlikely since inference seems to work.
2. **Multi-modal models always seem to exhibit this behavior** - we found Llama 3.2 Vision's loss starts at 3 or 4, Pixtral at 8 or so, and Qwen 2.5 VL also 4 ish. Because Gemma 3N includes audio as well, it might amplify the starting loss. But this is just a hypothesis. We also found quantizing Qwen 2.5 VL 72B Instruct to have extremely high perplexity scores of around 30 or so, but the model interestingly performs fine.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a37a8ce8ff2cfc3873a9f78acee3744c778692dc%252Foutput%283%29.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=1e47e11484ce6851fa77709f92c4f32e&sv=3){width=1979 height=1180}

## 🛠️ Technical Analysis {#technical-analysis}

### Gemma 3n : MatFormer {#gemma-3n-matformer}

So what is so special about Gemma 3n you ask? It is based on [Matryoshka Transformer or MatFormer](https://arxiv.org/abs/2310.07707) architecture meaning that each transformer layer/block embeds/nests FFNs of progressively smaller sizes. Think of it like progressively smaller cups put inside one another. The training is done so that at inference time you can choose the size you want and get the most of the performance of the bigger models.

There is also Per Layer Embedding which can be cached to reduce memory usage at inference time. So the 2B model (E2B) is a sub-network inside the 4B (aka 5.44B) model that is achieved by both Per Layer Embedding caching and skipping audio and vision components focusing solely on text.

The MatFormer architecture, typically is trained with exponentially spaced sub-models aka of sizes `S`, `S/2, S/4, S/8` etc in each of the layers. So at training time, inputs are randomly forwarded through one of the said sub blocks giving every sub block equal chance to learn. Now the advantage is, at inference time, if you want the model to be 1/4th of the original size, you can pick `S/4` sized sub blocks in each layer.

You can also choose to **Mix and Match** where you pick say, `S/4` sized sub block of one layer, `S/2` sized sub block of another layer and `S/8` sized sub block of another layer. In fact, you can change the sub models you pick based on the input itself if you fancy so. Basically its like choose your own kind of structure at every layer. So by just training a model of one particular size, you are creating exponentially many models of smaller sizes. No learning goes waste. Pretty neat huh.
