title: Introducing Stable LM 2 1.6B — Stability AI
description: Today, we are introducing our first language model from the new Stable LM 2 series: the 1.6 billion parameter base model and an instruction-tuned version.

# Introducing Stable LM 2 1.6B — Stability AI

 Jan 19 

**Key Takeaways:**

- [Stable LM 2 1.6B](https://huggingface.co/stabilityai/stablelm-2-1_6b) is a state-of-the-art 1.6 billion parameter small language model trained on multilingual data in English, Spanish, German, Italian, French, Portuguese, and Dutch.
- This model's compact size and speed lower hardware barriers, allowing more developers to participate in the generative AI ecosystem.
- In addition to the pre-trained and instruction-tuned version, we release the last checkpoint before the pre-training cooldown. We include optimizer states to facilitate developers in fine-tuning and experimentation. Data details will be provided in the upcoming technical report.
- Stable LM 2 1.6B can be used now both commercially and non-commercially with a [Stability AI Membership](https://stability.ai/membership) & you can test the model on [Hugging Face](https://huggingface.co/spaces/stabilityai/stablelm-2-1_6b-zephyr).[\
 \
](https://huggingface.co/spaces/stabilityai/stablelm-2-1_6b-zephyr)

Today, we are introducing our first language model from the new Stable LM 2 series: the 1.6 billion parameter [base model](https://huggingface.co/stabilityai/stablelm-2-1_6b) and an [instruction-tuned](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b) version. The base model is trained on approximately 2 trillion tokens for two epochs, incorporating multilingual data in English, Spanish, German, Italian, French, Portuguese, and Dutch. We leveraged recent algorithmic advancements in language modeling to strike a favorable balance between speed and performance, enabling fast experimentation and iteration with moderate resources. 

Data details will also be available with this release so that the open community can reproduce similarly performant models. Along with this, for the first time, we are releasing the final [pre-training checkpoint](https://huggingface.co/stabilityai/stablelm-2-1_6b/blob/global_step420000/README.md) before the cooldown, including the optimizer states, to help developers smoothly continue pre-training and fine-tuning their data – as some recent pre-trained models may be harder to fine-tune due to late-stage optimizations. In the upcoming days, we will share a comprehensive technical report that explores and describes the data mix and training procedure we followed.

**Model Performance**

We compare Stable LM 2 1.6B to other popular small language models such as Microsoft’s Phi-1.5 (1.3B) & Phi-2 (2.7B), TinyLlama 1.1B, or Falcon 1B. It outperforms models under 2B on most tasks, and even some larger ones, while offering compact size and speed when tested with few-shot performance across general benchmarks outlined in the Open LLM Leaderboard.

Thanks to explicit training on multilingual text, performance on [translated versions](https://arxiv.org/abs/2307.16039) of ARC Challenge, HellaSwag, TruthfulQA, MMLU, and LAMBADA show Stable LM 2 1.6B exceeding other models by a considerable margin.

*0-shot average accuracy performance on Okapi translated benchmarks and multilingual LAMBADA. Note that LAMBADA does not include Dutch and Portuguese.*

According to [MT Bench](https://paperswithcode.com/dataset/mt-bench) results, Stable LM 2 1.6B shows competitive performance, matching or even surpassing significantly larger models.

*(grading on a scale of 1 to 10)*

By releasing one of the most powerful small language models to date and providing complete transparency on its training details, we aim to empower developers and model creators to experiment and iterate quickly. It is important to note that, due to the nature of small, low-capacity language models, Stable LM 2 1.6B may similarly exhibit common issues such as high hallucination rates or potential toxic language. We ask the community to keep this in mind when building their applications and take appropriate measures to ensure they are developing responsibly.

\

**For Commercial and Non-Commercial Use**

Stable LM 2 1.6B, the first in a series of Stable LM models, is part of the [Stability AI Membership](https://stability.ai/membership). The membership has three distinct membership tiers, ensuring that anyone, from individuals to enterprises, can benefit from this technology. You can learn more [here](https://stability.ai/membership). 

Stay updated on our progress by signing up for our newsletter, and learn more about commercial applications by contacting us here. 

Follow us on,[ Instagram](https://www.instagram.com/stability.ai/),[ LinkedIn](https://www.linkedin.com/company/66318622/), and join our[ Discord Community](https://discord.gg/stablediffusion).

[   Stability AI Joins U.S. Artificial Intelligence Safety Institute Consortium   ](https://stability.ai/news-updates/stability-ai-joins-us-artificial-intelligence-safety-institute-consortium)
