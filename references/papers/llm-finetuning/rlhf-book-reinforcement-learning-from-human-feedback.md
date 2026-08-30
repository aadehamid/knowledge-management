# Reinforcement Learning from Human Feedback

A short introduction to RLHF and post-training focused on language models.

Nathan Lambert

## Abstract

Reinforcement learning from human feedback (RLHF) has become a crucial tool to build the latest machine learning systems at scale. The field grew around the core methods of RLHF into today’s broader suite of post-training techniques. In this book, we give a comprehensive introduction to the core methods for post-training models for people with some level of quantitative background, organized around the canonical RLHF recipe. The book starts with what RLHF does and why it was created, with seminal technical milestones in its young history and a primer on reinforcement learning context needed to understand the book. The core of the book details every optimization stage in using RLHF, from starting with instruction tuning to training a reward model and finally all of rejection sampling, reinforcement learning, on-policy distillation, and direct alignment algorithms. The book also discusses broader topics, such as the origins of RLHF – both in recent literature and in a convergence of disparate fields of science in economics, philosophy, and optimal control. The book concludes with advanced topics – understudied or emerging research questions in synthetic data, tool-use, character training, and evaluation – and open questions for the field. The book is released with a variety of companion resources, including a [codebase](https://rlhfbook.com/code), a [library](https://rlhfbook.com/library) to compare model completions from within post-training stages, and an educational [course](https://rlhfbook.com/course), to be a one-stop shop for learning all foundational concepts for post-training language models.

### Web Version vs. Physical Book (Errata Fixes)

The book will be re-printed roughly 2 and 6 months after the initial print in July 2026. This section tracks the differences between the web version and the physical book, and will be updated to note which improvements or fixes make it into which print version.

**Content additions and fixes:**

* Added a short subsection on agentic evaluation (Chapter 16) — [#492](https://github.com/natolambert/rlhf-book/pull/492).
* Clarified the history of outcome reward models, fixed some loose language, and reorganized the reward modeling chapter (Chapter 5) — [#516](https://github.com/natolambert/rlhf-book/pull/516).
* Cleaned up RL notation, especially the trajectory sampling distribution and time indexing (Chapter 6) — [#466](https://github.com/natolambert/rlhf-book/pull/466).
* Expanded the on-policy distillation section, particularly more on self-distillation and OPSD (Chapter 12) — [#439](https://github.com/natolambert/rlhf-book/pull/439).

**Organizational improvements:**

* Renamed Chapter 12 to "Synthetic Data & Distillation" — [#469](https://github.com/natolambert/rlhf-book/pull/469).
* Reordered the regularization chapter for better flow (Chapter 15) — [#486](https://github.com/natolambert/rlhf-book/pull/486).

**Typos and minor fixes:**

* Added an extra line to the Bradley-Terry reward model loss derivation (Chapter 5) — [#465](https://github.com/natolambert/rlhf-book/pull/465).
* Fixed the description of the reference model in the policy ratio (Chapter 6) — [#488](https://github.com/natolambert/rlhf-book/pull/488).
* Cleaned up assorted typos and minor wording issues — [#495](https://github.com/natolambert/rlhf-book/pull/495).

### RLHF Book Ecosystem

### Welcome to the Course

Book overview & course introduction

[Watch](https://www.youtube.com/watch?v=jQPiH-KB4B0&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

### Run the code

A codebase for the algorithms in this book

[Code](https://github.com/natolambert/rlhf-book/tree/main/code)

### Example RLHF'd model completions

Compare model completions at post-training stages

[View](https://rlhfbook.com/library)

### Join the Community

Discuss the book on Discord

[Join](https://discord.gg/yz5AwK4gBR)

### Community Translations

Unofficial translations maintained by readers, independent of the official print editions

[简体中文 (Simplified Chinese)](https://github.com/jweihe/RLHF-book-Chinese)

### Acknowledgements

I would like to thank the following people who helped me directly with this project: Costa Huang, Ross Taylor, Hamish Ivison, John Schulman, Valentina Pyatkin, Daniel Han, Shane Gu, Joanne Jang, LJ Miranda, Sharan Maiya, Andrew Carr, Cameron R. Wolfe, and others in my RL sphere (and of course Claude).

Additionally, thank you to the [contributors on GitHub](https://github.com/natolambert/rlhf-book/graphs/contributors) who helped improve this project.

### Changelog

*Last built: 12 August 2026*

**August 2026**: Finished accompanying course, Amazon sales begin.

**July 2026**: The print, ePub, and liveBook editions were published by Manning.

**April 2026**: Final editorial polish for print — ported Manning edition improvements, clarity pass on equations and terminology, typo/grammar fixes across all chapters, product chapter expansions. The book is heading to print, so expect fewer content changes going forward.

**March 2026**: Launch [course page](https://rlhfbook.com/course) with lecture videos; PDF syntax highlighting; product chapter expansions (Ch. 17).

**February 2026**: v2 content: direct alignment chapter, new diagrams, RL cheatsheet, appendices, search bar, Kindle support, editor fixes.

**January 2026**: Major chapter reorganization to match Manning book structure; code examples library; old URLs redirect to new locations.

**December 2025**: Working on v2 of the book based on editors' feedback. Check back for updates!

**November 2025**: [Manning preorder](https://www.manning.com/books/reinforcement-learning-from-human-feedback) available.

**July 2025**: Add tool use chapter (see [PR](https://github.com/natolambert/rlhf-book/pull/122))

**June 2025**: v1.1. Lots of RLVR/reasoning improvements (see [PR](https://github.com/natolambert/rlhf-book/pull/120))

**April 2025**: Finish v0; overoptimization, open questions, etc.; evaluation section; RLHF x Product research, improving website, reasoning section.

**March 2025**: Improving policy gradient section; finish DPO, major cleaning; start DPO chapter, improve intro.

**February 2025**: Improve SEO, add IFT chapter; RM additions, preference data, policy gradient finalization; PPO and GAE; added changelog, revamped introduction.

**January 2025**: Policy gradients (REINFORCE, PPO, GRPO); overoptimization content; discussion content merged from the blog; navigation and code-listing improvements.

**December 2024**: Preferences chapter; continued cleaning and additions.

**October 2024**: Regularization, preferences, and reward modeling chapters; figures and formatting.

**August 2024**: First chapters drafted (rejection sampling, bibliography); automated site builds set up.

**May 2024**: rlhfbook.com domain purchased; project started.

### Citation

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

[![GitHub](/assets/github.svg)](https://github.com/natolambert/rlhf-book)
[![arXiv](/assets/arxiv.svg)](https://arxiv.org/abs/2504.12501)
[![Manning](/assets/manning.svg)](https://www.manning.com/books/reinforcement-learning-from-human-feedback)
[![Amazon](/assets/amazon.svg)](https://amzn.to/4cwCDJQ)
[![Discord](/assets/discord.svg)](https://discord.gg/yz5AwK4gBR)

© 2024-2026 Nathan Lambert · [rlhfbook.com](https://rlhfbook.com)