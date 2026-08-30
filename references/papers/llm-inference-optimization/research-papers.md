title: GitHub - Mog9/research-papers: Research implementations focused on inference efficiency and model optimization. Includes custom Triton kernels, LoRA, knowledge distillation pipelines, and more.
description: Research implementations focused on inference efficiency and model optimization. Includes custom Triton kernels, LoRA, knowledge distillation pipelines, and more. - Mog9/research-papers

# GitHub - Mog9/research-papers: Research implementations focused on inference efficiency and model optimization. Includes custom Triton kernels, LoRA, knowledge distillation pipelines, and more.

A collection of implementations for various machine learning research papers, focusing on high-performance kernels and efficient architectures.

High-performance Triton kernel implementation of Rotary Position Embeddings. Fuses the rotation logic into a single GPU pass, achieving a 5.33x speedup over PyTorch. Includes correctness validation and benchmarking suite.

Implementation of Low-Rank Adaptation for efficient fine-tuning of large language models. Focuses on injecting trainable rank decomposition matrices into existing layers to reduce parameter count and memory usage during training.

Implementation of Knowledge Distillation techniques to compress large teacher models into smaller student models. Focuses on transferring soft-label information and intermediate representations to maintain accuracy with reduced compute requirements.

Implementation of Residual Learning frameworks (ResNet style). Focuses on skip connections to mitigate the vanishing gradient problem, enabling the training of significantly deeper neural networks with improved convergence.
