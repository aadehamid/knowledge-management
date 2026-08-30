title: Vision Reinforcement Learning \(VLM RL\) | Unsloth Documentation
description: Train Vision/multimodal models via GRPO and RL with Unsloth!

# Vision Reinforcement Learning (VLM RL) | Unsloth Documentation

Unsloth now supports vision/multimodal RL with [Qwen3-VL](https://docs.unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune), [Gemma 3](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune) and more. Due to Unsloth's unique [weight sharing](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#what-unsloth-offers-for-rl) and custom kernels, Unsloth makes VLM RL **1.5–2× faster,** uses **90% less VRAM**, and enables **15× longer context** lengths than FA2 setups, with no accuracy loss. This update also introduces Qwen's [GSPO](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl#gspo-rl) algorithm.

Unsloth can train Qwen3-VL-8B with GSPO/GRPO on a free Colab T4 GPU. Other VLMs work too, but may need larger GPUs. Gemma requires newer GPUs than T4 because vLLM [restricts to Bfloat16](https://docs.unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes), thus we recommend NVIDIA L4 on Colab. Our notebooks solve numerical math problems involving images and diagrams:

- **Qwen-3 VL-8B** (vLLM inference)**:** [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_%288B%29-Vision-GRPO.ipynb)
- **Gemma-3-4B** (Unsloth inference): [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29-Vision-GRPO.ipynb)

We have also added vLLM VLM integration into Unsloth natively, so all you have to do to use vLLM inference is enable the `fast_inference=True` flag when initializing the model. Special thanks to [Sinoué GAD](https://github.com/unslothai/unsloth/pull/2752) for providing the [first notebook](https://github.com/GAD-cell/vlm-grpo/blob/main/examples/VLM_GRPO_basic_example.ipynb) that made integrating VLM RL easier!

This VLM support also integrates our latest update for even more memory efficient \+ faster RL including our [Standby feature](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl#unsloth-standby), which uniquely limits speed degradation compared to other implementations.

```
os.environ['UNSLOTH_VLLM_STANDBY'] = '1' # To enable memory efficient GRPO with vLLM
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_seq_length = 16384, #Must be this large to fit image in context
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
```

It is also important to note, that vLLM does not support LoRA for vision/encoder layers, thus set `finetune_vision_layers = False` when loading a LoRA adapter. However you CAN train the vision layers as well if you use inference via transformers/Unsloth.

```
# Add LoRA adapter to the model for parameter efficient fine tuning
model = FastVisionModel.get_peft_model(
    model,

    finetune_vision_layers     = False,# fast_inference doesn't support finetune_vision_layers yet :(
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)
```

## 🦋Qwen 2.5 VL Vision RL Issues and Quirks {#qwen-2.5-vl-vision-rl-issues-and-quirks}

During RL for Qwen 2.5 VL, you might see the following inference output:

This was [reported](https://github.com/QwenLM/Qwen2.5-VL/issues/759) as well in Qwen2.5-VL-7B-Instruct output unexpected results "addCriterion". In fact we see this as well! We tried both non Unsloth, bfloat16 and float16 machines and other things, but it appears still. For example item 165 ie `train_dataset[165]` from the [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) dataset is below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-61a659529171fcc10ed6398a15912b21d6b1a076%252FUntitled.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=15ecf4a3e929b88e24fed9eb82bcc3a6&sv=3){width=512 height=512}

And then we get the above gibberish output. One could add a reward function to penalize the addition of addCriterion, or penalize gibberish outputs. However, the other approach is to train it for longer. For example only after 60 steps ish do we see the model actually learning via RL:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-5f34f66f0ac6508fd28343b16592c59b889ec5ca%252Fimage.webp%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=9bf926afc432c6fb9399826ab6cddc37&sv=3){width=1859 height=548}

Forcing `<|assistant|>` during generation will reduce the occurrences of these gibberish results as expected since this is an Instruct model, however it's still best to add a reward function to penalize bad generations, as described in the next section.

## 🏅Reward Functions to reduce gibberish {#reward-functions-to-reduce-gibberish}

To penalize `addCriterion` and gibberish outputs, we edited the reward function to penalize too much of `addCriterion` and newlines.

## 🏁GSPO Reinforcement Learning {#gspo-reinforcement-learning}

This update in addition adds GSPO ([Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)) which is a variant of GRPO made by the Qwen team at Alibaba. They noticed that GRPO implicitly results in importance weights for each token, even though explicitly advantages do not scale or change with each token.

This lead to the creation of GSPO, which now assigns the importance on the sequence likelihood rather than the individual token likelihoods of the tokens. The difference between these two algorithms can be seen below, both from the GSPO paper from Qwen and Alibaba:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-45d743dd5dcd590626777ce09cfab61808aa8c24%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=7cbe2a19911c4f2aa6122e0ec2b4d696&sv=3){width=967 height=260}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-ee755850cbe17482ce240dde227d55c62e9a3e64%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=993d4564ff42ccb1d8fd204671c71e4b&sv=3){width=891 height=302}

In Equation 1, it can be seen that the advantages scale each of the rows into the token logprobs before that tensor is sumed. Essentially, each token is given the same scaling even though that scaling was given to the entire sequence rather than each individual token. A simple diagram of this can be seen below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-b3c944808a15dde0a7ff45782f9f074993304bf1%252FCopy%2520of%2520GSPO%2520diagram%2520%281%29.jpg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=9bd5ff0634d760603b8c4f0dd743338d&sv=3){width=572 height=314} GRPO Logprob Ratio row wise scaled with advantages

Equation 2 shows that the logprob ratios for each sequence is summed and exponentiated after the Logprob ratios are computed, and only the resulting now sequence ratios get row wise multiplied by the advantages.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-62fc5b50921e79cce155d2794201c9b96faf941e%252FGSPO%2520diagram%2520%281%29.jpg%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=849c131ad5be2e813dd116959ab77e54&sv=3){width=626 height=667} GSPO Sequence Ratio row wise scaled with advantages

Enabling GSPO is simple, all you need to do is set the `importance_sampling_level = "sequence"` flag in the GRPO config.

Overall, Unsloth now with VLM vLLM fast inference enables for both 90% reduced memory usage but also 1.5-2x faster speed with GRPO and GSPO!

If you'd like to read more about reinforcement learning, check out out RL guide:

[Reinforcement Learning](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)

***Authors:***  *A huge thank you to* [*Keith*](https://www.linkedin.com/in/keith-truongcao-7bb84a23b/) *and* [*Datta*](https://www.linkedin.com/in/datta0/) *for contributing to this article!*
