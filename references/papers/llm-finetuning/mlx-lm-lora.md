title: GitHub - Goekdeniz-Guelmez/mlx-lm-lora: Train Large Language Models on MLX.
description: Train Large Language Models on MLX. Contribute to Goekdeniz-Guelmez/mlx-lm-lora development by creating an account on GitHub.

# GitHub - Goekdeniz-Guelmez/mlx-lm-lora: Train Large Language Models on MLX.

 [![logo](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/raw/main/logos/mlx_lm_lora.png){width=100%}](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/logos/mlx_lm_lora.png) 

[![image](https://camo.githubusercontent.com/63b25b83b5b9fd75aec7b9e3f934b967208a831fe2fe9052cfc846c436aab5a7/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6d6c782d6c6d2d6c6f72612e737667)](https://pypi.python.org/pypi/mlx-lm-lora)

With MLX-LM-LoRA you can, train Large Language Models locally on Apple Silicon using MLX. Training works with all models supported by [MLX-LM](https://github.com/ml-explore/mlx-lm), including:

- Llama
- Mistral
- Qwen
- Gemma
- OLMo, OLMoE
- MiniCPM, MiniCPM3
- and more...

**Training Types:**

- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **Full-precision**: Train all model parameters
- **Quantized training**: QLoRA with 4-bit, 6-bit, or 8-bit quantization
- **Quantization Aware Training (QAT)**: Apply quantization projection during training for SFT, DPO, and ORPO

**Training Algorithms:**

- **SFT**: Supervised Fine-Tuning
- **DPO**: Direct Preference Optimization
- **FTPO / Antidoom**: Final-token preference optimization for repairing repetition loops
- **CPO**: Contrastive Preference Optimization
- **ORPO**: Odds Ratio Preference Optimization
- **GRPO**: Group Relative Policy Optimization
- **GSPO**: Group Sequence Policy Optimization
- **Dr. GRPO**: Dr. Group Relative Policy Optimization
- **DAPO**: Decoupled Clip and Dynamic Sampling Policy Optimization
- **Online DPO**: Online Direct Preference Optimization
- **XPO**: Extended Preference Optimization
- **RLHF Reinforce KL**: Reinforced Reinforcement Learning from Human Feedback (with KL regularization)
- **PPO**: Proximal policy Optimization

**Quantization Aware Training (QAT):**

- Enable QAT for SFT, DPO, and ORPO with minimal post-update quantization projection.
- Supports 4-16 bit, group or per-tensor, and configurable start/interval.
- Use QAT to simulate quantization effects during training for better quantized model performance.

**Training Your Custom Preference Model:**

- You can now train a custom preference model for online preference training

> ---
> Head over to the examples repository for every notebook, YAML config, and walkthrough, including:
> - 🧪 **Fine-Tuning (Simple)** — LoRA on a standard SFT dataset
> - 🧠 **Fine-Tuning (Detailed)** — Full model weights for supervised fine-tuning
> - ⚖️ **ORPO Training** — Monolithic preference optimization
> - 📈 **DPO Training** — Direct preference optimization
> - 👥 **GRPO Training** — Group-based reinforcement training
> - 📄 **YAML configuration** — Example config file
> - …and more being added over time!
> ⭐ **Star the examples repo** to bookmark it: [https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks)

- [Install](#install)
- [Quick Start](#quick-start)
- [Training Methods](#training-methods)
    - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
    - [Contrastive Preference Optimization (CPO)](#contrastive-preference-optimization-cpo)
    - [Odds Ratio Preference Optimization (ORPO)](#odds-ratio-preference-optimization-orpo)
    - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
    - [Group Sequence Policy Optimization (GSPO)](#group-sequence-policy-optimization-gspo)
    - [Decoupled Reward Group Relative Policy Optimization (Dr. GRPO)](#decoupled-reward-group-relative-policy-optimization-dr-grpo)
    - [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](#decoupled-clip-and-dynamic-sampling-policy-optimization-dapo)
    - [Online DPO](#online-dpo)
    - [eXtended Preference Optimization (XPO)](#extended-preference-optimization-xpo)
    - [Reinforcement Learning from Human Feedback Reinforce (RLHF Reinforce)](#reinforced-reinforcement-learning-from-human-feedback-with-kl)
    - [Proximal Policy Optimization](#proximal-policy-optimization)
- [Other Features](#other-features)
    - [Examples Repository (moved)](#-examples-have-moved-)
    - [Training Your Custom Preference Model](#training-your-custom-preference-model)
- [Configuration](#configuration)
- [Dataset Formats](#dataset-formats)
- [Memory Optimization](#memory-optimization)
- [Evaluation & Generation](#evaluation--generation)
- [Performance Comparison](#performance-comparison)

---

```
pip install -U mlx-lm-lora
```

The main command is `mlx_lm_lora.train`. To see all options:

```
mlx_lm_lora.train --help
```

Basic training command:

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--data mlx-community/wikisql \
--iters 600
```

You can specify a YAML config with `-c`/`--config`:

```
mlx_lm_lora.train --config /path/to/config.yaml
```

Command-line flags will override corresponding values in the config file.

---

QAT projects trainable weights onto a quantized grid after each optimizer update, simulating quantization effects during training. This improves quantized model performance and robustness.

**Supported for:** SFT, DPO, ORPO

**QAT Flags:**

- `--qat-enable`    Enable QAT projection during training
- `--qat-bits`     Bit-width for QAT (default: 8)
- `--qat-group-size`  Group size for QAT (default: 64, 0\=per-tensor)
- `--qat-mode`     QAT mode (default: affine)
- `--qat-start-step`  Start QAT after this optimizer step (default: 1)
- `--qat-interval`   Apply QAT every N optimizer steps (default: 1)

**Example (SFT):**

```
mlx_lm_lora.train \
  --model <model> \
  --train \
  --train-mode sft \
  --data <data> \
  --qat-enable \
  --qat-bits 4 \
  --qat-group-size 64 \
  --qat-start-step 1 \
  --qat-interval 1
```

**Example (DPO):**

```
mlx_lm_lora.train \
  --model <model> \
  --train \
  --train-mode dpo \
  --data <data> \
  --qat-enable \
  --qat-bits 4
```

**Example (ORPO):**

```
mlx_lm_lora.train \
  --model <model> \
  --train \
  --train-mode orpo \
  --data <data> \
  --qat-enable \
  --qat-bits 8 \
  --qat-group-size 32
```

Standard instruction tuning using prompt-completion pairs.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode sft \
--sft-loss-type dft \
--data mlx-community/hermes-3 \
--batch-size 4 \
--learning-rate 1e-5 \
--iters 1000
```

**Key Parameters:**

- `--train-type`: Choose `lora` (default), `dora`, or `full`
- `--mask-prompt`: Apply loss only to assistant responses
- `--sft-loss-type`: SFT loss function - `nll` (default), memory-bounded `chunked_nll`, or dynamic fine-tuning loss `dft`
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--gradient-accumulation-steps`: Accumulate gradients over multiple steps

**Dataset Format:**

```
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"prompt": "Explain quantum computing", "completion": "Quantum computing uses..."}
{"text": "Complete text for language modeling"}
```

---

Train models using preference pairs without a separate reward model.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode dpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--dpo-cpo-loss-type sigmoid \
--reference-model-path Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
```

**Key Parameters:**

- `--beta`: KL penalty strength (default: 0.1)
- `--dpo-cpo-loss-type`: Loss function - `sigmoid`, `hinge`, `ipo`, or `dpop`
- `--delta`: Margin for hinge loss (default: 50.0)
- `--reference-model-path`: Reference model path (uses main model if not specified)

**Dataset Format:**

```
{"prompt": "User question", "chosen": "Good response", "rejected": "Bad response"}
{"system": "You are helpful", "prompt": "Question", "chosen": "Good", "rejected": "Bad"}
```

---

Repair reasoning doom loops using preference rows generated by Liquid AI's [Antidoom](https://github.com/Liquid4All/antidoom) pipeline:

```
mlx_lm_lora.train \
--model your/model \
--train \
--train-mode ftpo \
--data ./antidoom_data \
--learning-rate 1e-5 \
--lambda-mse-target 0.05 \
--tau-mse-target 1.0 \
--lambda-mse 0.4 \
--clip-epsilon-logits 2.0
```

Place Antidoom rows in `train.jsonl` (and optionally `valid.jsonl` and `test.jsonl`). Each row must contain `context_with_chat_template`, `rejected_decoded`, and `multi_chosen_decoded`. FTPO updates only the next-token distribution after the supplied context and uses a frozen copy of `--model` as the reference unless `--reference-model-path` is provided.

---

Variant of DPO designed for machine translation and other structured tasks.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode cpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--dpo-cpo-loss-type sigmoid
```

**Key Parameters:** Same as DPO. Uses identical dataset format to DPO.

---

Monolithic preference optimization without requiring a reference model.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode orpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--reward-scaling 1.0
```

**Key Parameters:**

- `--beta`: Temperature for logistic function (default: 0.1)
- `--reward-scaling`: Reward scaling factor (default: 1.0)

**Dataset Format:**

```
{"prompt": "Question", "chosen": "Good response", "rejected": "Bad response"}
{"prompt": "Question", "chosen": "Good", "rejected": "Bad", "preference_score": 8.0}
{"prompt": "Question", "chosen": {"messages": [...]}, "rejected": {"messages": [...]}}
```

---

Generate multiple responses per prompt and learn from their relative quality.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--data mlx-community/gsm8k \
--group-size 4 \
--epsilon 1e-4 \
--max-completion-length 512 \
--temperature 0.8 \
--reward-functions "accuracy_reward,format_reward" \
--reward-weights "[0.7, 0.3]"
```

**Key Parameters:**

- `--group-size`: Number of generations per prompt (default: 4)
- `--epsilon`: Numerical stability constant (default: 1e-4)
- `--max-completion-length`: Max generation length (default: 512)
- `--temperature`: Sampling temperature (default: 0.8)
- `--reward-functions`: Comma-separated reward function names
- `--reward-functions-file`: Path to custom reward functions file
- `--reward-weights`: JSON list of weights for each reward function
- `--grpo-loss-type`: Loss variant - `grpo`, `bnpo`, or `dr_grpo`

**Dataset Format:**

```
{"prompt": "Math problem", "answer": "42"}
{"prompt": "Question", "answer": "Response", "system": "You are helpful"}
{"prompt": "Question", "answer": "Response", "type": "math"}
```

**Custom Reward Functions:** Create a Python file with reward functions:

```
# my_rewards.py
from mlx_lm_lora.reward_functions import register_reward_function

@register_reward_function()
def my_custom_reward(prompt, completion, reference_answer, **kwargs):
    """Custom reward function"""
    # Your logic here
    return score  # float between 0 and 1
```

Then use: `--reward-functions-file ./my_rewards.py --reward-functions "my_custom_reward"`

---

GSPO extends GRPO with importance sampling at token or sequence level for improved sample efficiency.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--grpo-loss-type grpo \
--importance-sampling-level token \
--group-size 4 \
--epsilon 1e-4 \
--temperature 0.8
```

**Key Parameters:**

- `--importance-sampling-level`: Choose `token` (default) or `sequence`
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

Dr. GRPO decouples the reward computation from the policy optimization for more stable training.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--grpo-loss-type dr_grpo \
--group-size 4 \
--epsilon 1e-4 \
--temperature 0.8
```

**Key Parameters:**

- `--grpo-loss-type dr_grpo`: Enables Dr. GRPO variant
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

DAPO uses dual epsilon values for more flexible clipping in policy optimization.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--epsilon 1e-4 \
--epsilon-high 1e-2 \
--group-size 4 \
--temperature 0.8
```

**Key Parameters:**

- `--epsilon`: Lower bound for clipping (default: 1e-4)
- `--epsilon-high`: Upper bound for clipping (uses epsilon value if not specified)
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

Online preference optimization using a judge model or human feedback.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode online_dpo \
--data ./online_data \
--judge mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit \
--alpha 1e-5
```

**Key Parameters:**

- `--judge`: Judge model ID or "human" for human feedback
- `--alpha`: Learning rate for online updates (default: 1e-5)
- `--judge-config`: Additional configuration for judge model

**Dataset Format:**

```
{"prompt": [{"role": "user", "content": "Question"}]}
{"messages": [{"role": "user", "content": "Question"}]}
```

---

XPO extends online DPO with additional preference learning mechanisms.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode xpo \
--data ./xpo_data \
--judge mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit \
--alpha 1e-5 \
--beta 0.1
```

**Key Parameters:**

- `--judge`: Judge model ID or "human"
- `--alpha`: Online learning rate (default: 1e-5)
- `--beta`: KL penalty strength (default: 0.1)
- `--judge-config`: Additional judge configuration

**Dataset Format:** Same as Online DPO

---

Full RLHF REINFORCE pipeline with reward model and policy optimization Ziegler style.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode rlhf-reinforce \
--data Goekdeniz-Guelmez/ultrafeedback-prompt-flat \
--judge mlx-community/reward-model \
--alpha 1e-5 \
--beta 0.1
```

**Key Parameters:**

- `--judge`: Reward model ID
- `--alpha`: Policy learning rate (default: 1e-5)
- `--beta`: KL penalty strength (default: 0.1)

**Dataset Format:** Same as Online DPO

---

Full PPO pipeline with reward model and policy optimization.

```
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode ppo \
--data Goekdeniz-Guelmez/ultrafeedback-prompt-flat \
--judge mlx-community/reward-model \
--epsilon 0.2
```

**Key Parameters:**

- `--judge`: Reward model ID
- `--epsilon`: The Epsilon for numerical stability (default: 0.2)

**Dataset Format:** Same as Online DPO

---

This feature adds a second training stage on top of the judge (preference) stage. A reward model thats scores the policy’s generations and the policy is updated with a KL‑penalised PPO‑style loss.

1. Collect preference data → judge‑mode (online DPO) → reward model
2. Run RLHF (policy optimisation) using the reward model → final policy

```
python -m mlx_lm_lora.train_judge \
--model Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1 \
--train-type full \
--optimizer adamw \
--steps-per-report 1 \
--iters 50 \
--max-seq-length 1024 \
--adapter-path /Users/Goekdeniz.Guelmez@computacenter.com/Library/CloudStorage/OneDrive-COMPUTACENTER/Desktop/test \
--data mlx-community/Human-Like-DPO \
--gradient-accumulation-steps 1
```

**Dataset Format:** Same as DPO (with `prompt`, `chosen`, and `rejected` pairs).

---

```
# Model and data
--model <model_path>              # Model path or HF repo
--data <data_path>                # Dataset path or HF dataset name
--train-type lora                 # lora, dora, or full
--train-mode sft                  # sft, dpo, cpo, orpo, grpo, etc.

# Training schedule
--batch-size 4                    # Batch size
--iters 1000                      # Training iterations
--epochs 3                        # Training epochs (ignored if iters set)
--learning-rate 1e-5              # Learning rate
--gradient-accumulation-steps 1   # Gradient accumulation

# Model architecture
--num-layers 16                   # Layers to fine-tune (-1 for all)
--max-seq-length 2048            # Maximum sequence length

# LoRA parameters
--lora-parameters '{"rank": 8, "dropout": 0.0, "scale": 10.0}'

# Optimization
--optimizer adam                  # adam, adamw, qhadam, muon
--lr-schedule cosine             # Learning rate schedule
--grad-checkpoint                # Enable gradient checkpointing

# Quantization

# Quantization Aware Training (QAT)

QAT projects trainable weights onto a quantized grid after each optimizer update, simulating quantization effects during training. This improves quantized model performance and robustness. QAT is supported for SFT, DPO, and ORPO.

**QAT Flags:**

- `--qat-enable`    Enable QAT projection during training
- `--qat-bits`     Bit-width for QAT (default: 8)
- `--qat-group-size`  Group size for QAT (default: 64, 0=per-tensor)
- `--qat-mode`     QAT mode (default: affine)
- `--qat-start-step`  Start QAT after this optimizer step (default: 1)
- `--qat-interval`   Apply QAT every N optimizer steps (default: 1)

See [QAT section above](#quantization-aware-training-qat) for usage examples.
--load-in-4bits                  # 4-bit quantization
--load-in-6bits                  # 6-bit quantization  
--load-in-8bits                  # 8-bit quantization

# Quantization Aware Training (QAT)
--qat-enable                      # Enable QAT projection during training
--qat-bits 4                      # Bit-width for QAT (default: 8)
--qat-group-size 64               # Group size for QAT (default: 64, 0=per-tensor)
--qat-mode affine                 # QAT mode (default: affine)
--qat-start-step 1                # Start QAT after this optimizer step (default: 1)
--qat-interval 1                  # Apply QAT every N optimizer steps (default: 1)

# Monitoring
--steps-per-report 10            # Steps between loss reports
--steps-per-eval 200             # Steps between validation
--val-batches 25                 # Validation batches (-1 for all)
--wandb project_name             # WandB logging

# Checkpointing
--adapter-path ./adapters        # Save/load path for adapters
--save-every 100                 # Save frequency
--resume-adapter-file <path>     # Resume from checkpoint
--fuse                           # Fuse and save trained model
```

**Preference Optimization Methods:**

**DPO/CPO:**

```
--beta 0.1                        # KL penalty strength
--dpo-cpo-loss-type sigmoid       # sigmoid, hinge, ipo, dpop
--delta 50.0                      # Margin for hinge loss
--reference-model-path <path>     # Reference model path
```

**ORPO:**

```
--beta 0.1                        # Temperature parameter
--reward-scaling 1.0              # Reward scaling factor
```

**Group-Based Methods:**

**GRPO (Base):**

```
--group-size 4                    # Generations per prompt
--epsilon 1e-4                    # Numerical stability constant
--temperature 0.8                 # Sampling temperature
--max-completion-length 512       # Max generation length
--reward-functions "func1,func2"  # Comma-separated reward functions
--reward-functions-file <path>    # Custom reward functions file
--reward-weights "[0.5, 0.5]"    # JSON list of reward weights
--grpo-loss-type grpo             # grpo, bnpo, dr_grpo
```

**GSPO (GRPO \+ Importance Sampling):**

```
--importance-sampling-level token # token (default) or sequence
# Plus all GRPO parameters
```

**Dr. GRPO (Decoupled Rewards):**

```
--grpo-loss-type dr_grpo         # Enable Dr. GRPO variant
# Plus all GRPO parameters
```

**DAPO (Dynamic Clipping):**

```
--epsilon 1e-4                   # Lower bound for clipping
--epsilon-high 1e-2              # Upper bound for clipping
# Plus all GRPO parameters
```

**Online Methods:**

**Online DPO:**

```
--judge <model_id>               # Judge model or "human"
--alpha 1e-5                     # Online learning rate
--beta 0.1                       # KL penalty strength
--judge-config '{}'              # Additional judge configuration
```

**XPO (Extended Preference Optimization):**

```
--judge <model_id>               # Judge model or "human"
--alpha 1e-5                     # Online learning rate
--beta 0.1                       # KL penalty strength
--judge-config '{}'              # Judge configuration
# Plus additional XPO-specific parameters
```

**RLHF Reinforce:**

```
--judge <reward_model_id>        # Reward model
--alpha 1e-5                     # Policy learning rate
--beta 0.1                       # KL penalty strength
--group-size 4                   # Samples for policy optimization
--judge-config '{}'              # Reward model configuration
```

**PPO:**

```
--judge <reward_model_id>        # Reward model
--alpha 1e-5                     # Policy learning rate
--epsilon 0.2                    # Numerical stability value
--group-size 4                   # Samples for policy optimization
--judge-config '{}'              # Reward model configuration
```

---

Place JSONL files in a directory:

```
data/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

```
mlx_lm_lora.train --data "Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1" --train
```

Configure custom field names:

```
--text-feature "content"          # For text datasets
--chat-feature "conversation"     # For chat datasets
--prompt-feature "question"       # For prompt-completion
--completion-feature "answer"     # For prompt-completion
--chosen-feature "preferred"      # For preference datasets
--rejected-feature "dispreferred" # For preference datasets
--system-feature "instruction"    # For system messages
```

**SFT - Chat Format:**

```
{"messages": [
  {"role": "system", "content": "You are helpful"},
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "content": "4"}
]}
```

**SFT - Completion Format:**

```
{"prompt": "What is 2+2?", "completion": "2+2 equals 4"}
```

**SFT - Text Format:**

```
{"text": "The complete text for language modeling"}
```

**DPO/CPO Format:**

```
{"prompt": "Explain AI", "chosen": "AI is artificial intelligence", "rejected": "AI is magic"}
```

**ORPO Format:**

```
{"prompt": "What is AI?", "chosen": "Good explanation", "rejected": "Bad explanation", "preference_score": 0.8}
```

**GRPO Format:**

```
{"prompt": "Solve: 2+2=?", "answer": "4", "system": "You are a math tutor"}
```

**RLHF (Online DPO, XPO, RLHF Reinforced, PPO) Format:**

```
{"prompt": [{"role": "user", "content": "Question"}]}
```

or:

```
{"prompt": "Question"}
```

---

Use quantized models to reduce memory usage:

```
# 4-bit quantization (most memory efficient)
mlx_lm_lora.train --model <model> --load-in-4bits --train

# 6-bit quantization (balanced)
mlx_lm_lora.train --model <model> --load-in-6bits --train

# 8-bit quantization (higher quality)
mlx_lm_lora.train --model <model> --load-in-8bits --train
```

```
# Reduce batch size
--batch-size 1

# Train fewer layers
--num-layers 8

# Enable gradient checkpointing
--grad-checkpoint

# Linear and hybrid recurrent models (Qwen3.5/Next, Kimi, Mamba, etc.) are
# detected automatically and use checkpointed linear blocks and smaller SSM
# blocks during training.

# Reduce sequence length
--max-seq-length 1024

# Use gradient accumulation
--gradient-accumulation-steps 4 --batch-size 1
```

```
# Smaller LoRA rank
--lora-parameters '{"rank": 4, "dropout": 0.1, "scale": 10.0}'

# Train specific layers only
--num-layers 8
```

---

Evaluate on test set:

```
mlx_lm_lora.train \
--model <model_path> \
--adapter-path <adapter_path> \
--data <data_path> \
--test \
--test-batches 500
```

Use `mlx-lm` for generation with trained adapters:

```
mlx_lm.generate \
--model <model_path> \
--adapter-path <adapter_path> \
--prompt "Your prompt here" \
--max-tokens 100 \
--temperature 0.7
```

Merge LoRA weights into base model:

```
mlx_lm_lora.train \
--model <model_path> \
--adapter-path <adapter_path> \
--fuse
```

---

```
--lr-schedule cosine              # Cosine annealing
--lr-schedule linear              # Linear decay
--lr-schedule constant            # Constant rate
```

```
--optimizer adam                  # Adam optimizer
--optimizer adamw                 # AdamW with weight decay
--optimizer qhadam               # Quasi-hyperbolic Adam
--optimizer muon                 # Muon optimizer
```

List available reward functions:

```
mlx_lm_lora.train --list-reward-functions
```

Use multiple reward functions:

```
--reward-functions "accuracy_reward,format_reward,length_reward" \
--reward-weights "[0.5, 0.3, 0.2]"
```

```
--wandb my_project_name
```

---

| Method | Type | Reference Model | Judge Model | Multiple Generations | Key Benefit |
|----|----|----|----|----|----|
| SFT | Supervised | ❌ | ❌ | ❌ | Simple, fast training |
| DPO | Preference | ✅ | ❌ | ❌ | No reward model needed |
| CPO | Preference | ✅ | ❌ | ❌ | Better for structured tasks |
| ORPO | Preference | ❌ | ❌ | ❌ | Monolithic optimization |
| GRPO | Policy | ❌ | ❌ | ✅ | Group-based learning |
| GSPO | Policy | ❌ | ❌ | ✅ | Importance sampling |
| Dr. GRPO | Policy | ❌ | ❌ | ✅ | Decoupled rewards |
| DAPO | Policy | ❌ | ❌ | ✅ | Dynamic clipping |
| Online DPO | Online RL | ✅ | ✅ | ✅ | Real-time feedback |
| XPO | Online RL | ✅ | ✅ | ✅ | Extended preferences |
| RLHF Reinforce | Online RL | ✅ | ✅ | ✅ | Full RL pipeline |
| PPO | Online RL | ✅ | ✅ | ✅ | Full RL pipeline |

---

```
# SFT
mlx_lm_lora.train --model <model> --train-mode sft --data <data>

# DPO
mlx_lm_lora.train --model <model> --train-mode dpo --data <data> --beta 0.1

# CPO
mlx_lm_lora.train --model <model> --train-mode cpo --data <data> --beta 0.1

# ORPO
mlx_lm_lora.train --model <model> --train-mode orpo --data <data> --beta 0.1
```

```
# GRPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> --group-size 4

# GSPO (GRPO with importance sampling)
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--importance-sampling-level token --group-size 4

# Dr. GRPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--grpo-loss-type dr_grpo --group-size 4

# DAPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--epsilon 1e-4 --epsilon-high 1e-2 --group-size 4
```

```
# Online DPO
mlx_lm_lora.train --model <model> --train-mode online_dpo --data <data> \
--judge <judge_model> --alpha 1e-5

# XPO
mlx_lm_lora.train --model <model> --train-mode xpo --data <data> \
--judge <judge_model> --alpha 1e-5

# RLHF Reinforce
mlx_lm_lora.train --model <model> --train-mode rlhf-reinforce --data <data> \
--judge <reward_model> --alpha 1e-5 --group-size 4

# PPO
mlx_lm_lora.train --model <model> --train-mode ppo --data <data> \
--judge <reward_model> --epsilon 0.2 --group-size 4
```

---

1. **Out of Memory**: Reduce batch size, use quantization, enable gradient checkpointing
2. **Slow Training**: Increase batch size, reduce validation frequency
3. **Poor Quality**: Increase LoRA rank, train more layers, check data quality
4. **Convergence Issues**: Adjust learning rate, try different optimizers

| Model Size | Recommended Settings |
|----|----|
| 1-3B | `--batch-size 4 --num-layers 16` |
| 7B | `--batch-size 2 --num-layers 8 --load-in-8bits` |
| 13B\+ | `--batch-size 1 --num-layers 4 --load-in-4bits --grad-checkpoint` |

---

```
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./my_data
train_type: lora
train_mode: sft
batch_size: 4
learning_rate: 1e-5
iters: 1000
lora_parameters:
  rank: 8
  dropout: 0.0
  scale: 10.0
```

```
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./preference_data
train_mode: dpo
beta: 0.1
dpo_cpo_loss_type: sigmoid
batch_size: 2
learning_rate: 5e-6
iters: 500
```

```
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./grpo_data
train_mode: grpo
group_size: 4
temperature: 0.8
reward_functions: "accuracy_reward,format_reward"
reward_weights: [0.7, 0.3]
max_completion_length: 512
```

---

To measure performance on your hardware with MLX-LM-LoRA:

```
# SFT with speed/memory reporting
mlx_lm_lora.train \
  --model Goekdeniz-Guelmez/JOSIE-1.1-4B-Instruct \
  --data mlx-community/wikisql \
  --train --train-mode sft \
  --batch-size 4 --iters 100 \
  --steps-per-report 10
```

Monitor output for:

- `it/s` (iterations per second)
- `peak_memory` (in GB)
- `tokens/sec` (throughput)

---

Below is a comparison of iteration speed and memory usage across different training libraries my (MLX-LM-LoRA), [Unsloth](https://github.com/unslothai/unsloth), [mlx-tune](https://github.com/ARahim3/mlx-tune). Benchmarks are approximate and depend on hardware, model size, and configuration.

**Test Configuration:**

- **Hardware**: M4 Pro (24GB unified memory) vs. NVIDIA A100 (80GB VRAM)
- **Settings**: All LoRA layers trained, batch size of 1, max context length of 4096, 100 training steps
- **Quantization**: No quantization for Qwen/Qwen3-0.6B, 4-bit quantization for Qwen/Qwen3-8B

| Model Size | Training Mode | MLX-LM-LoRA | Unsloth | mlx-tune |
|----|----|----|----|----|
|  |  | **(Apple Silicon)** | **(NVIDIA GPU)** | **(Apple Silicon)** |
|  |  | **Speed / Memory** | **Speed / Memory** | **Speed / Memory** |
| **Qwen/Qwen3-0.6B** | SFT | \~4.7 it/s<br>\~2-2 GB | \~2.7 it/s<br>\~1-2 GB VRAM | \~0.6 it/s<br>\~4-6 GB |
| **Qwen/Qwen3-0.6B** | ORPO | \~4.5 it/s<br>\~2-4 GB | \~2.4 it/s<br>\~2-8 GB VRAM | OOM |
| **Qwen/Qwen3-0.6B** | GRPO | \~0.02 it/s<br>\~9-20 GB | \~0.04 it/s<br>\~76-80 GB VRAM | OOM |
| **Qwen/Qwen3-8B** | SFT | \~4.1 it/s<br>\~6-10 GB | \~1.3 it/s<br>\~10-16 GB VRAM | \~0.07 it/s<br>\~8-18 GB |

**MLX-LM-LoRA (Apple Silicon - Native MLX)**

- ✅ **Comprehensive**: 12 training algorithms (SFT, DPO, CPO, ORPO, GRPO, GSPO, Dr. GRPO, DAPO, Online DPO, XPO, RLHF, PPO)
- ✅ **Custom Preference Models**: Built-in judge training for online preference workflows
- ✅ **Unified Memory**: Access to full system RAM (up to 512GB on Ultra)
- ✅ **Moderate Speed**: Optimized MLX implementation with native Apple Silicon support
- ✅ **CLI-First**: Simple command-line, and notebook interface with YAML config support
- ⚠️ **Apple Only**: Requires Apple Silicon (M1/M2/M3/M4)

**Unsloth (NVIDIA GPU - CUDA/Triton)**

- ✅ **Fastest**: Highly optimized Triton kernels for NVIDIA GPUs
- ✅ **Production Ready**: Battle-tested, widely used in industry
- ✅ **Memory Efficient**: Custom CUDA kernels minimize VRAM usage
- ✅ **Rich Ecosystem**: Seamless integration with Hugging Face, TRL, PEFT
- ⚠️ **NVIDIA Only**: Requires CUDA-compatible GPU (doesn't work on Apple Silicon)
- ⚠️ **VRAM Limited**: Constrained by GPU VRAM (24-80GB typical)

**mlx-tune (Apple Silicon - MLX with Unsloth API)**

- ✅ **API Compatible**: Drop-in replacement for Unsloth code on Apple Silicon
- ✅ **Unified Memory**: Same memory advantages as MLX-LM-LoRA
- ✅ **Portability Focus**: Write once on Mac, deploy on CUDA
- ✅ **Vision Models**: VLM fine-tuning support (Qwen3.5, etc.)
- ⚠️ **Limited Methods**: Fewer training algorithms than MLX-LM-LoRA
- ⚠️ **Wrapper Library**: Built on top of MLX, adds abstraction layer
- ⚠️ **Moderate Speed**: Similar to MLX-LM-LoRA (both use MLX backend)

---

 [![MacPaw](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/raw/main/logos/macpaw.png){width=200}](https://macpaw.com) [![TypeFox](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/raw/main/logos/typefox.png){width=200}](https://typefox.io) [![Computacenter](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/raw/main/logos/cc.webp){width=200}](https://www.computacenter.com) 

MLX-LM-LoRA is also beeing used by researchers, engineers, and other profesionals by `Apple`, `IBM`, `Bosch`, `Red Hat`, `Daimler Truck`, and `Mercedes-Benz Group`.

> **Is you or your team using MLX-LM-LoRA?** I'd love to hear from you! Feel free to reach out and I'll add your logo here too. 🚀

---

[![Alt](https://camo.githubusercontent.com/0acc3784fbc727fff00d0a6974a7236a77a7ae225b90ea68c7f326ac5705c7ff/68747470733a2f2f7265706f62656174732e6178696f6d2e636f2f6170692f656d6265642f643665393431663635613864616266353833343565396365383363323363383162353539376264322e737667 "Repobeats analytics image")](https://camo.githubusercontent.com/0acc3784fbc727fff00d0a6974a7236a77a7ae225b90ea68c7f326ac5705c7ff/68747470733a2f2f7265706f62656174732e6178696f6d2e636f2f6170692f656d6265642f643665393431663635613864616266353833343565396365383363323363383162353539376264322e737667)

---

[    ![Star History Chart](https://camo.githubusercontent.com/efd793ee37553c75e303a560c03889d9c9138ea631a9fb2aabae217c966c96d8/68747470733a2f2f6170692e737461722d686973746f72792e636f6d2f63686172743f7265706f733d476f656b64656e697a2d4775656c6d657a2f6d6c782d6c6d2d6c6f726126747970653d64617465266c6567656e643d746f702d6c656674)  ](https://www.star-history.com/?repos=Goekdeniz-Guelmez%2Fmlx-lm-lora&type=date&legend=top-left)

---

```
@software{gülmez2025mlxlmlora,
  author = {Gökdeniz Gülmez},
  title = {{MLX-LM-LoRA}: Train LLMs on Apple silicon with MLX and the Hugging Face Hub},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-lm-lora},
  version = {0.1.0},
  year = {2025},
}
```
