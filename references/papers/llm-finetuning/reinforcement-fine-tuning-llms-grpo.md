title: Reinforcement Fine-Tuning LLMs With GRPO
description: Improve LLM reasoning with reinforcement fine-tuning and reward functions.

# Reinforcement Fine-Tuning LLMs With GRPO

Join **Reinforcement Fine-Tuning LLMs with GRPO**, built in collaboration with Predibase, and taught by Travis Addair, its Co-Founder and CTO, and Arnav Garg, its Senior Engineer and Machine Learning Lead.

Reinforcement Fine-Tuning (RFT) is a technique for adapting LLMs to complex reasoning tasks like mathematics and coding. RFT leverages reinforcement learning (RL) to help models develop their strategies for completing a task, rather than relying on pre-existing examples as in traditional supervised fine-tuning. One RL algorithm, called Group Relative Policy Optimization (GRPO), is beneficial for tasks with verifiable outcomes and can work well even when you have fewer than 100 training examples. Using RFT to adapt small, open-source models can lead to competitive performance on reasoning tasks, giving you more options for your LLM-powered applications.

In this course, you’ll take a technical deep dive into RFT with GRPO. You’ll learn how to build reward functions that you can use in the GRPO training process to guide an LLM toward better performance on multi-step reasoning tasks.

In detail, you’ll learn:

- When reinforcement fine-tuning is a better fit than supervised fine-tuning, especially for tasks involving multi-step reasoning or limited labeled data.
- How GRPO uses programmable reward functions as a more scalable alternative to the human feedback required for other reinforcement learning algorithms, such as RLHF and DPO.
- How to frame the Wordle game as a reinforcement fine-tuning problem and see how an LLM can learn to plan, analyze feedback, and improve its strategy over time.
- How to design reward functions that power the reinforcement fine-tuning process.
- Techniques for evaluating more subjective tasks, such as rating the quality of a text summary, using an LLM as a judge.
- Why reward hacking happens and how to avoid it by adding penalty functions to discourage undesirable behaviors.
- The four key components of the loss calculation in the GRPO algorithm: token probability distribution ratios, advantages, clipping, and KL-divergence.
- How to launch reinforcement fine-tuning jobs using Predibase’s hosted training services.

By the end of this course, you’ll be able to fine-tune LLMs using reinforcement learning to improve reasoning without relying on large labeled datasets or subjective human feedback.
