title: Teaching NVIDIA Nemotron 3.5 Lightning to Route Code Reviews
description: CodeRabbit, NVIDIA, and Baseten post-trained Nemotron 3.5 Lightning for routing code reviews, improving agreement and reducing estimated inference cost.

# Teaching NVIDIA Nemotron 3.5 Lightning to Route Code Reviews

Every CodeRabbit review begins with a routing decision. Our system evaluates code changes and converts the model output into a review configuration. Because routing is one of our highest-volume model tasks, it was a good place to test whether a smaller model could handle a real part of the review process.

[NVIDIA Nemotron 3.5 Lightning](https://blogs.nvidia.com/blog/nemotron-lightning-switchyard-rtx-dgx) stood out because of its code responses and was small enough to run efficiently in smaller infrastructure. We wanted to see whether post-training could teach CodeRabbit's routing rules.

To answer that question, CodeRabbit worked with NVIDIA and Baseten on a two-stage post-training run.

- Supervised fine-tuning (SFT) on examples of correct routing decisions.
- Reinforcement learning with verifiable rewards (RLVR), where CodeRabbit's routing policy scored each answer.

The project brought together [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 3.5 Lightning model and post-training frameworks, including NVIDIA [NeMo AutoModel](https://github.com/nvidia-nemo/automodel) and NVIDIA [NeMo RL](https://github.com/nvidia-nemo/rl); Baseten's managed H100 Training Jobs and A100 Dedicated Inference; and CodeRabbit's code review data and evaluation.

## The results {#heading-the-results}

- Post-training Nemotron 3.5 Lightning was remarkably easy, requiring under three hours of experimentation and costing less than $100.
- The post-trained model achieved higher accuracy than the previous GPT-class model by about 4%.
- It also reduced estimated inference costs by about 50%.

On a frozen 1,000-task evaluation, exact route agreement increased from 75.8% for the GPT baseline model to 80.4% after SFT and 80.7% after SFT plus RLVR. Output agreement measured by Cohen's kappa reached 0.544, up from 0.461 after SFT.

![Diagram showing how candidate models assign complexity tags to a code change and CodeRabbit's scorer converts those tags into a review configuration.](https://www.coderabbit.ai/content/assets/nemotron-lightning-routing/review-complexity-routing.png)

*Figure 1. Each candidate model assigns complexity tags to a code change, and CodeRabbit's scorer converts those tags into a review configuration.*

## The experiment {#heading-the-experiment}

We started with 39,566 examples from public repositories. We split the data by repository to prevent train and evaluation leakage. After filtering and exact tokenizer checks, the main SFT set contained 9,996 examples and the frozen eval dataset contained 1,000.

We used knowledge distillation to create the training examples. A stronger model generated a reference routing decision for each code change. We filtered those outputs and used the remaining examples to teach NVIDIA Nemotron 3.5 Lightning how CodeRabbit routes a review.

## Stage 1: Distilled supervised fine-tuning {#heading-stage-1-distilled-supervised-fine-tuning}

We began with supervised fine-tuning. Using NVIDIA [NeMo AutoModel](https://github.com/nvidia-nemo/automodel), a PyTorch-native open training library, on Baseten's managed NVIDIA H100 Training Jobs, we trained lightweight rank-8 LoRA adapters for one epoch. We tested three learning rates. The 2e-4 run recorded the lowest validation loss at all five checkpoints, so we selected its step-1,249 adapter for the next stage.

![Line chart showing training and validation loss during supervised fine-tuning across three learning rates.](https://www.coderabbit.ai/content/assets/nemotron-lightning-routing/sft-loss.png)

*Graph 1. Training and validation loss during supervised fine-tuning. Lower is better.*

On the frozen 1,000-task evaluation, supervised fine-tuning increased exact route agreement from 75.8% to 80.4%. Output agreement, measured with Cohen's kappa, increased from 0.429 to 0.461. This gave us the checkpoint we carried into reward training.

With the supervised checkpoint selected, we moved to reward training to improve its decisions on the hardest examples.

## Stage 2: Refining decisions with verifiable rewards {#heading-stage-2-refining-decisions-with-verifiable-rewards}

Using the NVIDIA NeMo RL open library for efficient post-training on Baseten infrastructure with NVIDIA H100 GPUs, we ran RLVR with GRPO on top of the tuned output from Stage 1, after merging Stage 1's LoRA into the base model's weights. We trained on 910 selected prompts, evaluated every 25 steps, and stopped when validation plateaued.

## Results {#heading-results}

Cohen's kappa measures agreement beyond chance; higher is better. Here, it shows how closely the model's outputs match the reference outputs.

Relative to SFT-only, SFT plus RLVR added \+0.0835 kappa and \+0.3 percentage points of route agreement. The paired confidence interval for the route delta crossed zero, so we treat that route result as non-regression rather than a statistically decisive improvement. The output-agreement gain is the clearer RLVR result.

| Candidate | Output agreement (Cohen's kappa) | Exact route agreement |
|----|----|----|
| Baseline model | 0.429 | 75.8% |
| Nemotron 3.5 Lightning \+ SFT | 0.461 | 80.4% |
| Nemotron 3.5 Lightning \+ SFT \+ RLVR | 0.544 | 80.7% |

The final model completed all 1,000 requests, with no empty or abnormally short outputs. This rules out obvious collapse while leaving sustained-load and production-traffic checks for the next phase.

## Inference on an NVIDIA A100 system {#heading-inference-on-an-nvidia-a100-system}

One advantage of Nemotron 3.5 Lightning is its compact size, which can be hosted on an NVIDIA A100 system instead of an H100.

For serving, Baseten Dedicated Inference loaded the combined SFT plus RLVR output as one rank-16 LoRA over the original NVIDIA Nemotron 3.5 Lightning model. The final policy ran through vLLM on one 80 GiB A100.

Measured throughput was 314.82 aggregate output tokens per second with eight concurrent requests.

## Peak serving economics {#heading-peak-serving-economics}

For the same eval workload, we compared inference cost for the OpenAI model against the tuned model.

- At OpenAI's [public prices](https://developers.openai.com/api/docs/models/gpt-5.4-nano), the total cost was $2.34.
- Baseten's [public resource table](https://docs.baseten.co/deployment/resources) lists one NVIDIA A100 at $0.06667 per minute. At the measured peak aggregate rate, the total ended up being $1.16.
- Estimated savings were $1.18, or 50.4%.

The tuned model also generated 63.4% fewer output tokens than the baseline model on this task.

![Bar chart comparing peak serving cost for the same 1,000 tasks: GPT baseline at $2.34 and Nemotron on Baseten A100 at $1.16.](https://www.coderabbit.ai/content/assets/nemotron-lightning-routing/peak-serving-cost.png)

*Peak serving cost for the same 1,000 tasks, showing a 50.4% potential reduction at saturated throughput.*

## Conclusion {#heading-conclusion}

This experiment showed that NVIDIA Nemotron 3.5 Lightning responds well to fine-tuning: SFT delivered a clear quality gain, and RLVR improved output agreement while preserving route accuracy. NVIDIA's Nemotron 3.5 Lightning model and training stack, combined with Baseten's managed training and serving infrastructure, made the experiment practical end to end. The result makes Lightning a strong candidate for other narrow, high-volume parts of CodeRabbit's pipeline, and we are excited to keep exploring those opportunities with NVIDIA and Baseten.
