title: Post-Training Course by Nathan Lambert
description: Free course lectures on RLHF, reward models, preference tuning, RLVR, and modern LLM post-training.

# Course

 A full course accompanying the book with added resources and other lectures I've given. 

The slide decks are usually built with Claude Opus, [Colloquium](https://github.com/natolambert/colloquium), and substantial human revisions.

### Welcome to the Course

Introduction and overview of what you'll learn

- [**Prerequisites**](#prerequisites) — what to know before starting.
- [**Primary Material**](#lectures) — the core lecture series that follows the book chapter by chapter, with recordings, slides, PDFs, and source.
- [**Extra Resources**](#extra-resources) — recommended books, external RL courses, and Nathan's own talks paired with the chapters they go with.
- [**Other Guest Lectures and Talks**](#other-lectures) — invited talks and standalone presentations.

## Prerequisites

This course is roughly aimed at early AI PhD or master's students, but it is designed to be accessible to anyone willing to put in the work.

You do **not** need prior reinforcement learning or language modeling background to start. A motivated learner who studies hard — leaning on today's top LLMs as a tutor to unpack unfamiliar math, code, and jargon — can follow the entire course. I encourage you to go down rabbit holes, skip or reorder videos, and chase what excites you.

If you prefer the traditional coursework path, the usual background is the basics of language modeling / NLP plus basic machine learning (e.g. an intro to AI course and an intro to ML course).

### Lecture 0: The ML Foundations of LLM Post-Training

A refresher on the ML prerequisites of post-training — language modeling, KL, cross-entropy, & other math — to acclimate to the series

[Additional learning material](#extra-resources)

## Primary Material

### Lecture 1: Overview

Chapters 1-3 · Foundations of RLHF and post-training

### Lecture 2: IFT, Reward Models, & Rejection Sampling

Chapters 4, 5, 9 · Start of the core optimization methods section

### Lecture 3: RL Motivation & Math

Chapter 6, Part 1 · Policy gradients math, intuitions, and theory

### Lecture 4: RL Implementation & Practice

Chapter 6, Part 2 · Code, loss aggregation, async training, and practical engineering

### Lecture 5: The Rise of Reasoning Models

Chapter 7 · RLVR, inference-time scaling, and the 2025 reasoning model wave

### Lecture 6: Direct Preference Optimization

Chapter 8 · Deriving DPO step by step, plus variants and practice

### Conversation 1: Frontier post-training recipes in 2026 (w/ Finbarr Timbers)

### Lecture 7: Synthetic Data and Modern Post-training Methods

Chapter 12 · On-policy distillation, AI feedback, Constitutional AI, and rubrics

### Lecture 9: Over-Optimization and RLHF's Bad Reputation

Chapter 14 & Appendix B · Goodhart's law, reward hacking, sycophancy, and style

### Lecture 10: Regularization in RL, Why RL Generalizes, and Why SFT Forgets

Chapter 15 · The KL penalty, implicit regularization, and why RL forgets less

### Conversation 2: From academic research to frontier practice (w/ Scott Geng)

We discuss what it takes to land a well-grounded academic result into a near-frontier model. Scott and I worked together on DPO for Olmo 3, and how building models is much more than the idea, it's making it fit in a more complex puzzle.

### Lecture 11: Tool Use, Function Calling and The Road to Agents

Chapter 13 · Why models need tools, MCP and harnesses, and tool-use RL

### Lecture 12: The Evolution of Frontier Model Evaluation

Chapter 16 · The eras of evaluation, agentic evals, and trusting the numbers

### Lecture 13: An Introduction to Character Training

Chapter 17 · Constitutions, model specs, model personalities, and open questions

### Bonus 1: What's With the Shoggoths?

Bonus talk · The meme that explained post-training.

## Extra Resources

Outside material that I've personally used for going deeper on reinforcement learning and language models. The books in particular are wonderful complements.

### Books & Courses

### More talks from Nathan

Various presentations over my last few years working in post-training, grouped by corresponding chapter.

## Other Guest Lectures and Talks

### 2026

### An Introduction to Reinforcement Learning from Human Feedback and Post-training

SALA 2026 · Quito, Ecuador · March 2026

#### Citation

If you found this useful for your research, please cite it!

For the web and arXiv version:

```
@misc{lambert2025reinforcementlearninghumanfeedback,
  title = {Reinforcement Learning from Human Feedback},
  author = {Nathan Lambert},
  year = {2025},
  eprint = {2504.12501},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2504.12501}
}
```

For the Manning edition:

```
@book{lambert2026reinforcement,
  author = {Nathan Lambert},
  title = {Reinforcement Learning from Human Feedback: Alignment and post-training of {LLMs}},
  year = {2026},
  publisher = {Manning Publications},
  isbn = {9781633434301},
  url = {https://www.manning.com/books/reinforcement-learning-from-human-feedback}
}
```
