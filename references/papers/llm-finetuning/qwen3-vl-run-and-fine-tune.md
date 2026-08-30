title: Qwen3-VL: How to Run Guide | Unsloth Documentation
description: Learn to fine-tune and run Qwen3-VL locally with Unsloth.

# Qwen3-VL: How to Run Guide | Unsloth Documentation

Qwen3-VL is Qwen’s new vision models with **instruct** and **thinking** versions. The 2B, 4B, 8B and 32B models are dense, while 30B and 235B are MoE. The 235B thinking LLM delivers SOTA vision and coding performance rivaling GPT-5 (high) and Gemini 2.5 Pro. Qwen3-VL has vision, video and OCR capabilities as well as 256K context (can be extended to 1M). [Unsloth](https://github.com/unslothai/unsloth) supports **Qwen3-VL fine-tuning and** [**RL**](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl). Train Qwen3-VL (8B) for free with our [notebooks](https://docs.unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune#fine-tuning-qwen3-vl).

[Running Qwen3-VL](https://docs.unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune#running-qwen3-vl) [Fine-tuning Qwen3-VL](https://docs.unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune#fine-tuning-qwen3-vl)

## 🖥️ **Running Qwen3-VL** {#running-qwen3-vl}

To run the model in llama.cpp, vLLM, Ollama etc., here are the recommended settings:

### ⚙️ Recommended Settings {#recommended-settings}

Qwen recommends these settings for both models (they're a bit different for Instruct vs Thinking):

Qwen3-VL also used the below settings for their benchmarking numbers, as mentioned [on GitHub](https://github.com/QwenLM/Qwen3-VL/tree/main?tab=readme-ov-file#generation-hyperparameters).

Instruct Settings:

```
export greedy='false'
export seed=3407
export top_p=0.8
export top_k=20
export temperature=0.7
export repetition_penalty=1.0
export presence_penalty=1.5
export out_seq_length=32768
```

Thinking Settings:

```
export greedy='false'
export seed=1234
export top_p=0.95
export top_k=20
export temperature=1.0
export repetition_penalty=1.0
export presence_penalty=0.0
export out_seq_length=40960
```

### 🐛Chat template bug fixes {#chat-template-bug-fixes}

At Unsloth, we care about accuracy the most, so we investigated why after the 2nd turn of running the Thinking models, llama.cpp would break, as seen below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-37356b40688b10a85c927e1d432739a15bb33682%252Fimage.webp%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=d75098100a7a9f8e95c98c4c48252324&sv=3){width=2771 height=1084}

The error code:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Value is not callable: null at row 63, column 78:
            {%- if '</think>' in content %}
                {%- set reasoning_content = ((content.split('</think>')|first).rstrip('\n').split('<think>')|last).lstrip('\n') %}
                                                                             ^
```

We have successfully fixed the Thinking chat template for the VL models so we re-uploaded all Thinking quants and Unsloth's quants. They should now all work after the 2nd conversation - **other quants will fail to load after the 2nd conversation.**

### **Qwen3-VL Unsloth uploads**: {#qwen3-vl-unsloth-uploads}

Qwen3-VL is now supported for GGUFs by llama.cpp as of 30th October 2025, so you can run them locally!

### 📖 Llama.cpp: Run Qwen3-VL Tutorial {#llama.cpp-run-qwen3-vl-tutorial}

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-9bf7ec93680f889d7602e5f56a8d677d6a58ae6a%252Funsloth%2520made%2520with%2520love.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=277f332e5047561c4e1d638ed99e63b9&sv=3){width=1015 height=379}

1. Let's download this image

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4b30cc86b2c75edf95ee1ec6fe0c51fb30afd6c0%252F8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=ed97357ddad50625e683c418322c954b&sv=3){width=1000 height=600}

1. Then, let's use llama.cpp's auto model downloading feature, try this for the 8B Instruct model:

1. Once in, you will see the below screen:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-636dfd126430a8a8c91ef6d248b007daa34561c5%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=79b8677ab1a111cf0af150c5cd42fb86&sv=3){width=2805 height=1384}

1. Load up the image via `/image PATH` ie `/image unsloth.png` then press ENTER

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-7525265b8ef19c7fd17cca64d1b64ffe1959c2d1%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=aa731168aba505497f474ade435dde52&sv=3){width=912 height=286}

1. When you hit ENTER, it'll say "unsloth.png image loaded"

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2c996efe3373ae7f05bfec4d214524768624a6a8%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=e5b2eaa49930475b63ea431a2c9c1240&sv=3){width=816 height=307}

1. Now let's ask a question like "What is this image?":

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-62bd79e094c7daad6a8f021194aa0e67ef96f9a5%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=cef1f6b7912b33dae864e6dee8ebc4fa&sv=3){width=2834 height=565}

1. Now load in picture 2 via `/image picture.png` then hit ENTER and ask "What is this image?"

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-317cc2c7e41765ff466d357d14d506115f3262b6%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=7f8dbfd0c959002cf16a42733151b77e&sv=3){width=2848 height=1214}

1. And finally let's ask how are both images are related (it works!)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e323226293156ac17708836c635c6df3ab2b9ca3%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=e2bd88f3b21ef6895ebfc0d7de09fad3&sv=3){width=2842 height=626}

1. You can also download the model via (after installing `pip install huggingface_hub hf_transfer` ) HuggingFace's `snapshot_download` which is useful for large model downloads, **since llama.cpp's auto downloader might lag.** You can choose Q4\_K\_M, or other quantized versions.

1. Run the model and try any prompt. **For Instruct:**

1. **For Thinking**:

### 🪄Running Qwen3-VL-235B-A22B and Qwen3-VL-30B-A3B {#running-qwen3-vl-235b-a22b-and-qwen3-vl-30b-a3b}

For Qwen3-VL-235B-A22B, we will use llama.cpp for optimized inference and a plethora of options.

1. We're following similar steps to above however this time we'll also need to perform extra steps because the model is so big.
2. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..
3. Run the model and try a prompt. Set the correct parameters for Thinking vs. Instruct.

**Instruct:**

**Thinking:**

1. Edit, `--ctx-size 16384` for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

**Use**  `--fit on`  **introduced 15th Dec 2025 for maximum usage of your GPU and CPU.**

Optionally, use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

### 🐋 Docker: Run Qwen3-VL {#docker-run-qwen3-vl}

If you already have Docker desktop, to run Unsloth's models from Hugging Face, run the command below and you're done:

Or you can run Docker's uploaded Qwen3-VL models:

## 🦥 **Fine-tuning Qwen3-VL** {#fine-tuning-qwen3-vl}

Unsloth supports fine-tuning and reinforcement learning (RL) Qwen3-VL including the larger 32B and 235B models. This includes support for fine-tuning for video and object detection. As usual, Unsloth makes Qwen3-VL models train 1.7x faster with 60% less VRAM and 8x longer context lengths with no accuracy degradation. We made two Qwen3-VL (8B) training notebooks which you can train free on Colab:

**Saving Qwen3-VL to GGUF now works as llama.cpp just supported it!**

If you want to use any other Qwen3-VL model, just change the 8B model to the 2B, 32B etc. one.

The goal of the GRPO notebook is to make a vision language model solve maths problems via RL given an image input like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-fe1591d4378d19fa5115f61680d60356846807f5%252Four_new_3_datasets.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=dbb89fa374f7fe36de7f6d79a8655ce9&sv=3){width=2414 height=1363}

This Qwen3-VL support also integrates our latest update for even more memory efficient \+ faster RL including our [Standby feature](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl#unsloth-standby), which uniquely limits speed degradation compared to other implementations. You can read more about how to train vision LLMs with RL with our [VLM GRPO guide](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl).

### Multi-image training {#multi-image-training}

In order to fine-tune or train Qwen3-VL with multi-images the most straightforward change is to swap

with:

Using map kicks in dataset standardization and arrow processing rules which can be strict and more complicated to define.
