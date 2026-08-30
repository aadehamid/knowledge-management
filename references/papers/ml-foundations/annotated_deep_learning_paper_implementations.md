title: GitHub - labmlai/annotated\_deep\_learning\_paper\_implementations: 🧑‍🏫 60+ Implementations/tutorials of deep learning papers with side-by-side notes 📝; including transformers \(original, xl, switch, feedback, vit, ...\), optimizers \(adam, adabelief, sophia, ...\), gans\(cyclegan, stylegan2, ...\), 🎮 reinforcement learning \(ppo, dqn\), capsnet, distillation, ... 🧠
description: 🧑‍🏫 60+ Implementations/tutorials of deep learning papers with side-by-side notes 📝; including transformers \(original, xl, switch, feedback, vit, ...\), optimizers \(adam, adabelief, sophia, ...\), gans\(cyclegan, stylegan2, ...\), 🎮 reinforcement learning \(ppo, dqn\), capsnet, distillation, ... 🧠 - labmlai/annotated\_deep\_learning\_paper\_implementations

# GitHub - labmlai/annotated\_deep\_learning\_paper\_implementations: 🧑‍🏫 60\+ Implementations/tutorials of deep learning papers with side-by-side notes 📝; including transformers (original, xl, switch, feedback, vit, ...), optimizers (adam, adabelief, sophia, ...), gans(cyclegan, stylegan2, ...), 🎮 reinforcement learning (ppo, dqn), capsnet, distillation, ... 🧠

[![Twitter](https://camo.githubusercontent.com/9b939d1a01c1db43909095fcd24a61f7f1a729df5a52360c0235d246787241b3/68747470733a2f2f696d672e736869656c64732e696f2f747769747465722f666f6c6c6f772f6c61626d6c61693f7374796c653d736f6369616c)](https://twitter.com/labmlai)

This is a collection of simple PyTorch implementations of neural networks and related algorithms. These implementations are documented with explanations,

[The website](https://nn.labml.ai/index.html) renders these as side-by-side formatted notes. We believe these would help you understand these algorithms better.

[![Screenshot](https://camo.githubusercontent.com/c922d65eb1a33dfc48a86cb6e902cccef1e54ec343cd290eb2c8bf45010f7d68/68747470733a2f2f6e6e2e6c61626d6c2e61692f64716e2d6c696768742e706e67)](https://camo.githubusercontent.com/c922d65eb1a33dfc48a86cb6e902cccef1e54ec343cd290eb2c8bf45010f7d68/68747470733a2f2f6e6e2e6c61626d6c2e61692f64716e2d6c696768742e706e67)

We are actively maintaining this repo and adding new implementations almost weekly. [![Twitter](https://camo.githubusercontent.com/9b939d1a01c1db43909095fcd24a61f7f1a729df5a52360c0235d246787241b3/68747470733a2f2f696d672e736869656c64732e696f2f747769747465722f666f6c6c6f772f6c61626d6c61693f7374796c653d736f6369616c)](https://twitter.com/labmlai) for updates.

- [JAX implementation](https://nn.labml.ai/transformers/jax_transformer/index.html)
- [Multi-headed attention](https://nn.labml.ai/transformers/mha.html)
- [Triton Flash Attention](https://nn.labml.ai/transformers/flash/index.html)
- [Transformer building blocks](https://nn.labml.ai/transformers/models.html)
- [Transformer XL](https://nn.labml.ai/transformers/xl/index.html)
    - [Relative multi-headed attention](https://nn.labml.ai/transformers/xl/relative_mha.html)
- [Rotary Positional Embeddings](https://nn.labml.ai/transformers/rope/index.html)
- [Attention with Linear Biases (ALiBi)](https://nn.labml.ai/transformers/alibi/index.html)
- [RETRO](https://nn.labml.ai/transformers/retro/index.html)
- [Compressive Transformer](https://nn.labml.ai/transformers/compressive/index.html)
- [GPT Architecture](https://nn.labml.ai/transformers/gpt/index.html)
- [GLU Variants](https://nn.labml.ai/transformers/glu_variants/simple.html)
- [kNN-LM: Generalization through Memorization](https://nn.labml.ai/transformers/knn)
- [Feedback Transformer](https://nn.labml.ai/transformers/feedback/index.html)
- [Switch Transformer](https://nn.labml.ai/transformers/switch/index.html)
- [Fast Weights Transformer](https://nn.labml.ai/transformers/fast_weights/index.html)
- [FNet](https://nn.labml.ai/transformers/fnet/index.html)
- [Attention Free Transformer](https://nn.labml.ai/transformers/aft/index.html)
- [Masked Language Model](https://nn.labml.ai/transformers/mlm/index.html)
- [MLP-Mixer: An all-MLP Architecture for Vision](https://nn.labml.ai/transformers/mlp_mixer/index.html)
- [Pay Attention to MLPs (gMLP)](https://nn.labml.ai/transformers/gmlp/index.html)
- [Vision Transformer (ViT)](https://nn.labml.ai/transformers/vit/index.html)
- [Primer EZ](https://nn.labml.ai/transformers/primer_ez/index.html)
- [Hourglass](https://nn.labml.ai/transformers/hour_glass/index.html)

- [Generate on a 48GB GPU](https://nn.labml.ai/neox/samples/generate.html)
- [Finetune on two 48GB GPUs](https://nn.labml.ai/neox/samples/finetune.html)
- [LLM.int8()](https://nn.labml.ai/neox/utils/llm_int8.html)

- [Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/index.html)
- [Denoising Diffusion Implicit Models (DDIM)](https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html)
- [Latent Diffusion Models](https://nn.labml.ai/diffusion/stable_diffusion/latent_diffusion.html)
- [Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/index.html)

- [Original GAN](https://nn.labml.ai/gan/original/index.html)
- [GAN with deep convolutional network](https://nn.labml.ai/gan/dcgan/index.html)
- [Cycle GAN](https://nn.labml.ai/gan/cycle_gan/index.html)
- [Wasserstein GAN](https://nn.labml.ai/gan/wasserstein/index.html)
- [Wasserstein GAN with Gradient Penalty](https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html)
- [StyleGAN 2](https://nn.labml.ai/gan/stylegan/index.html)

- [Graph Attention Networks (GAT)](https://nn.labml.ai/graphs/gat/index.html)
- [Graph Attention Networks v2 (GATv2)](https://nn.labml.ai/graphs/gatv2/index.html)

Solving games with incomplete information such as poker with CFR.

- [Kuhn Poker](https://nn.labml.ai/cfr/kuhn/index.html)

- [Proximal Policy Optimization](https://nn.labml.ai/rl/ppo/index.html) with [Generalized Advantage Estimation](https://nn.labml.ai/rl/ppo/gae.html)
- [Deep Q Networks](https://nn.labml.ai/rl/dqn/index.html) with with [Dueling Network](https://nn.labml.ai/rl/dqn/model.html), [Prioritized Replay](https://nn.labml.ai/rl/dqn/replay_buffer.html) and Double Q Network.

- [Adam](https://nn.labml.ai/optimizers/adam.html)
- [AMSGrad](https://nn.labml.ai/optimizers/amsgrad.html)
- [Adam Optimizer with warmup](https://nn.labml.ai/optimizers/adam_warmup.html)
- [Noam Optimizer](https://nn.labml.ai/optimizers/noam.html)
- [Rectified Adam Optimizer](https://nn.labml.ai/optimizers/radam.html)
- [AdaBelief Optimizer](https://nn.labml.ai/optimizers/ada_belief.html)
- [Sophia-G Optimizer](https://nn.labml.ai/optimizers/sophia.html)

- [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)
- [Layer Normalization](https://nn.labml.ai/normalization/layer_norm/index.html)
- [Instance Normalization](https://nn.labml.ai/normalization/instance_norm/index.html)
- [Group Normalization](https://nn.labml.ai/normalization/group_norm/index.html)
- [Weight Standardization](https://nn.labml.ai/normalization/weight_standardization/index.html)
- [Batch-Channel Normalization](https://nn.labml.ai/normalization/batch_channel_norm/index.html)
- [DeepNorm](https://nn.labml.ai/normalization/deep_norm/index.html)

- [PonderNet](https://nn.labml.ai/adaptive_computation/ponder_net/index.html)

- [Evidential Deep Learning to Quantify Classification Uncertainty](https://nn.labml.ai/uncertainty/evidence/index.html)

- [Fuzzy Tiling Activations](https://nn.labml.ai/activations/fta/index.html)

- [Greedy Sampling](https://nn.labml.ai/sampling/greedy.html)
- [Temperature Sampling](https://nn.labml.ai/sampling/temperature.html)
- [Top-k Sampling](https://nn.labml.ai/sampling/top_k.html)
- [Nucleus Sampling](https://nn.labml.ai/sampling/nucleus.html)

- [Zero3 memory optimizations](https://nn.labml.ai/scaling/zero3/index.html)

```
pip install labml-nn
```
