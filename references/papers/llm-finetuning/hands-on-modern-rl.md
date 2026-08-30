title: GitHub - walkinglabs/hands-on-modern-rl: 🚀 An open-source, hands-on curriculum bridging the gap from basic RL concepts to LLM alignment, RLVR, and advanced Agentic systems.
description: 🚀 An open-source, hands-on curriculum bridging the gap from basic RL concepts to LLM alignment, RLVR, and advanced Agentic systems. - walkinglabs/hands-on-modern-rl

# GitHub - walkinglabs/hands-on-modern-rl: 🚀 An open-source, hands-on curriculum bridging the gap from basic RL concepts to LLM alignment, RLVR, and advanced Agentic systems.

> **📣 Announcement**
> We sincerely thank everyone for your support of this tutorial! A new version is coming soon. Many sections are still being organized and refined, so we appreciate your patience. Suggestions and feedback are always welcome!

> **Note:** This course was created with AI assistance and has not yet been fully reviewed. It may contain factual mistakes or code that does not run as expected. Issues and pull requests are very welcome.

- **\[2026-08-19\]** 🎮 **Online Classic RL Environments and Scripts**: Over the past two weeks, we have added and refined a collection of online reinforcement learning environments, training scripts, and companion notebooks. Learners can now run classic reinforcement learning experiments online, inspect training logs and evaluation results, and study the algorithms more conveniently. We also fixed many previously reported bugs in the course content, links, and experiment code.
- **\[2026-05-15\]** 📖 **Full English Translation & PDF Release**: Complete English translation of all chapters is now available. PDF builds for both Chinese and English editions are released automatically via CI.
- **\[2026-05-13\]** 🚀 **Major Upgrade: LLM and Traditional RL Hands-on Labs**: Added reproducible training examples for **Agentic RL** (Deep Research / rLLM) and **Traditional RL** (Actor-Critic continuous control). Includes complete code and fine-tuning analysis for building an Agentic training system from scratch, along with new VLM RL (GeoQA geometry reasoning) hands-on experiments!
- **\[2026-05-02\]** Initial browsable open-source release for testing and feedback.

WalkingLab is collaborating with ModelScope to provide online training environments for classic reinforcement learning experiments. A ModelScope Studio brings the experiment interface, runtime, and training entry point together on one page, so learners can start training in a browser and observe the agent without first configuring a local environment.

Every Studio has a companion notebook under [`code/online-experiments`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/README.md). The notebook imports the same training runtime as the Studio, exposes the experiment parameters, prints the full training log, plots checkpoint evaluations, and displays the learned-policy replay or result artifact.

| Experiment | Resource | Companion notebook | Live Studio |
|----|----|----|----|
| 01 · CartPole PPO | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole) |
| Gymnasium Playground | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment-gymnasium.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium) |
| 02 · ViZDoom | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment02-vizdoom.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment02-vizdoom) |
| 03 · Atari / ALE | xGPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari) |
| 04 · Board Games & Self-Play | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment04-board-selfplay.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay) |
| 05 · Multi-Agent Games | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment05-multiagent-games.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games) |
| 06 · MiniGrid Adventures | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment06-minigrid-adventure.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure) |
| 07 · JAX MinAtar | CPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment07-jax-games.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games) |
| 08 · ManiSkill | xGPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment08-maniskill.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill) |
| 10 · MineStudio / Minecraft | xGPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment10-minestudio.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio) |
| 11 · Unity ML-Agents | xGPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents) |
| 12 · AI2-THOR | xGPU | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment12-ai2thor-embodied.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied) |

CPU entries run on an ordinary notebook instance. Experiments 03, 08, 10, 11, and 12 require a scheduled ModelScope xGPU Notebook and check CUDA before training.

|  [![Course learning map screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-learning-path.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-learning-path.png) <br> **One continuous learning path** <br> ~(Begin with a CartPole trial and progress through value learning, policy optimization, and modern agents.)  |  [![PPO code focus screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-code-focus.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-code-focus.png) <br> **Equations meet code** <br> ~(Key PPO, DPO, and GRPO derivations sit beside their implementations, with every tensor accounted for.)  |
|:--:|:--:|
|  [![CartPole training metrics screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-training-metrics.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-training-metrics.png) <br> **Claims tested by experiments** <br> ~(Real training curves, ablations, and failure signals show when an algorithm works and when it does not.)  |  [![RLHF pipeline screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-rlhf-pipeline.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-rlhf-pipeline.png) <br> **Classic RL to language models** <br> ~(Policy gradients and PPO lead naturally into RLHF, DPO, GRPO, and RLVR.)  |
|  [![Agentic RL experiment page screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-agentic-rl.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-agentic-rl.png) <br> **Agents as sequential decisions** <br> ~(Tool use, browser interaction, and code repair become problems of states, actions, trajectories, and credit assignment.)  |  [![Atari Pong DQN experiment page screenshot](https://github.com/walkinglabs/hands-on-modern-rl/raw/main/docs/public/readme/feature-atari-game.png){width=100%}](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/public/readme/feature-atari-game.png) <br> **Phenomena before abstractions** <br> ~(CartPole, LunarLander, Atari, and LLM experiments pose the problem before introducing the mathematics.)  |

---

> [!NOTE]
> We hope this open course gives more learners the courage to climb toward the frontier of intelligence and solve more of the hard problems on the path to AGI.
> The course is evolving quickly. We recommend focusing on chapters that are not marked as under construction; chapters still in progress may contain mistakes, and corrections or suggestions are welcome.

> **Help Wanted**
> Because compute resources are limited, we are seeking GPU support. If you can help with GPU access, please contact [physicoada@gmail.com](mailto:physicoada@gmail.com).

- [Book Features](#book-features)
- [About This Book](#about-this-book)
- [Structure of the Book](#structure-of-the-book)
- [Experiment Code](#experiment-code)
- [Recommended Learning Path](#recommended-learning-path)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

Reinforcement learning studies a simple but difficult problem: a system acts, observes the consequences, and uses them to improve its next action. When rewards arrive late, observations are incomplete, and each update changes the distribution of future experience, the familiar supervised-learning model of inputs and labels is no longer enough. We need a language for interaction, a way to estimate long-term return, and methods that improve a policy stably from limited data.

**Hands-On Modern RL** follows that problem from beginning to end. CartPole and multi-armed bandits first make states, actions, rewards, and policies observable. Markov decision processes, value functions, and Bellman equations then provide a common language. From there, the book develops DQN, policy gradients, actor-critic methods, PPO, continuous control, and offline reinforcement learning. With those foundations in place, RLHF, DPO, GRPO, and RLVR become extensions of the same sequential-decision framework rather than an isolated collection of acronyms.

The second half expands the environment to tools, browsers, code repositories, vision, and audio. An action may be a passage of text, a function call, or a sequence of interface operations. A reward may come from human preference, a rule-based verifier, or a process reward model. The setting changes, but three questions run through the entire book: **How should the decision process be represented? How should an outcome be credited to earlier actions? How can we tell whether a policy has actually improved?**

Each chapter follows a problem–method–experiment–reflection rhythm. A concrete task first exposes the difficulty. The concepts and equations needed to solve it come next. Runnable code, training curves, and evaluation metrics then test the argument. The chapter closes by examining assumptions, failure modes, and the limits of the method. Mathematics explains observed behavior; experiments check the mathematics.

Implementations retain the visible skeleton of each algorithm. Readers can trace trajectory collection, advantage estimation, loss construction, and metric changes—and see how reward hacking, KL drift, entropy collapse, distribution shift, or evaluation leakage can invalidate an apparently successful run.

This book is for students, researchers, and engineers with basic machine-learning experience who want a systematic understanding of modern reinforcement learning. Readers should be comfortable with Python and basic PyTorch, and should know introductory linear algebra, probability, and calculus. The mathematical appendices rebuild these tools to the depth required by the chapters, so a separate advanced-mathematics sequence is not a prerequisite.

After completing the core chapters and labs, you should be able to:

- formulate a new decision problem using MDPs, value functions, Bellman equations, and credit assignment;
- implement, read, and diagnose DQN, REINFORCE, actor-critic methods, PPO, DPO, and GRPO;
- explain how SFT, reward modeling, preference optimization, RLHF, and RLVR fit together in LLM post-training;
- design trajectories, rewards, training loops, and evaluation protocols for tool-use, code, and multimodal agents;
- identify failure modes behind training curves and test proposed improvements with controlled experiments.

This repository is an active courseware project. Content is being expanded chapter by chapter, with emphasis on correctness, runnable examples, and a stable learning path.

- Course site: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- Source content: [`docs/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs)
- Runnable examples: [`code/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code)
- Local verification: `npm run verify`
- License: [CC BY-NC-SA 4.0](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE)

Issues and pull requests are welcome for typo fixes, conceptual corrections, reproducibility improvements, references, and focused course extensions.

The course is under active development. Planned milestones:

The book contains seven parts and twenty-six chapters. Parts I–III establish the common language and algorithmic foundations of reinforcement learning. Part IV brings those tools into LLM post-training. Parts V and VI study what changes when the action space expands to tools and multimodal environments. Part VII asks how to detect failures, build trustworthy evaluations, and move the research frontier forward. The appendices provide implementation, mathematics, and engineering references.

| Reading | Central question |
|:---|:---|
| [Introduction to Reinforcement Learning](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/preface/intro.md) | What does RL study, and how does the book connect classical methods to modern language models? |
| [A History of Reinforcement Learning](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/preface/brief-history/index.md) | How did control, TD learning, DQN, AlphaGo, RLHF, and reasoning models develop? |
| [Environment Setup](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/preface/env-setup.md) | How do you prepare the environments for documentation, control tasks, and LLM experiments? |

The book first makes an agent's failures and improvements observable, then develops the mathematical objects needed to describe long-term decisions.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 1 | [Starting with CartPole](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter01_cartpole/principles.md) | Use states, actions, rewards, policies, and training curves to see a complete RL loop. |
| 2 | [RL Problems and Definitions](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/bandit.md) | Move from exploration and exploitation to MDPs, returns, trajectories, and partial observability. |
| 3 | [Value Functions and Bellman Equations](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/value-bellman.md) | Express how a present action changes the future through state values, action values, and recursion. |
| 4 | [Classical RL Methods](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/dp-mc-td.md) | Compare dynamic programming, Monte Carlo, and temporal-difference learning. |

When the state space grows, tables no longer suffice. This part introduces function approximation and follows the value-learning and policy-learning routes into PPO and continuous control.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 5 | [Deep Q-Networks](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter07_dqn/from-q-to-dqn.md) | Approximate action values with neural networks and stabilize learning with replay and target networks. |
| 6 | [Policy Gradient Methods](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter08_policy_gradient/policy-gradient.md) | Optimize the policy directly, derive REINFORCE, and reduce variance with baselines. |
| 7 | [Actor-Critic Methods](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter09_actor_critic/advantage-function.md) | Let policy and value estimation learn together, joined by the advantage function. |
| 8 | [TRPO and PPO](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter10_ppo/trust-region-clipping.md) | Limit each policy update and combine GAE with a clipped objective for stable learning. |
| 9 | [Continuous Control and World Models](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter11_continuous_control/intro.md) | Progress from DDPG, TD3, and SAC to model-based RL, MuZero, and Dreamer. |

When interaction is expensive, expert demonstrations are available, or a task spans multiple agents and time scales, the object of learning changes.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 10 | [Offline Reinforcement Learning](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter12_offline_rl/intro.md) | Learn from a fixed dataset while controlling distribution shift and extrapolation error. |
| 11 | [Imitation, Inverse RL, and Meta-RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter13_imitation_meta_rl/bc-dagger.md) | Learn policies or rewards from experts and adapt to new tasks. |
| 12 | [Exploration, Multi-Agent, and Hierarchical RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter14_exploration_marl_hierarchical/intro.md) | Address sparse rewards, coordination, and the hierarchy of long-horizon tasks. |

Language models expand an “action” into a passage of text. Policy optimization, distribution constraints, and credit assignment now lead into preference alignment, verifiable rewards, and inference-time computation.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 13 | [The RLHF Training Pipeline](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter15_rlhf/base-model-to-assistant.md) | Move from SFT, AI feedback, and reward modeling to PPO-style RL fine-tuning and evaluation. |
| 14 | [Preference Alignment and the DPO Family](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter17_dpo/intro.md) | Derive DPO from a KL-constrained objective and compare preference-optimization assumptions. |
| 15 | [GRPO, RLVR, and Verifier Engineering](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter18_grpo/grpo-practice-and-mechanism.md) | Train mathematical, coding, and tool-use capabilities with group-relative advantages and verifiable rewards. |
| 16 | [Reasoning Models and Inference-Time Compute](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter19_reasoning/emergence-and-o1.md) | Explain long-reasoning training, compute-budget control, and chain-of-thought alignment. |
| 17 | [Process Rewards and Inference-Time Search](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter20_prm_search/outcome-vs-process.md) | Move supervision from final answers to intermediate steps and combine it with search. |
| 18 | [Industrial LLM RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter16_llm_rl_industrial/intro.md) | Scale a single-machine algorithm into a coordinated data, inference, training, and evaluation system. |

Once agents call tools across many environment steps, the unit of training becomes a trajectory. Credit assignment, environment construction, and safety boundaries become central.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 19 | [Tool Use, Multi-Turn Interaction, and Multi-Agent RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter22_agentic/overview.md) | Formalize Agentic RL, synthesize tool trajectories, and run DeepCoder and FinQA labs. |
| 20 | [Reinforcement Learning for Coding Agents](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter23_rl_based_swe/swe-bench-and-rlvr.md) | Study software-engineering agents through SWE-bench, code world models, and self-play. |
| 21 | [Deep Research and Browser Agents](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter24_deep_research/browser-rl-harness.md) | Build trainable browser environments and evaluate deep-research systems. |
| 22 | [Computer Use and GUI Agents](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter25_computer_use/training.md) | Train interface agents while handling instruction hierarchy and prompt injection. |

Vision, audio, robot actions, and generative models introduce new state representations, reward sources, and evaluation criteria.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 23 | [Vision-Language Model RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter26_vlm/vlm-challenges.md) | Design visual rewards and reflection, then run VLM-GRPO and GeoQA experiments. |
| 24 | [Audio, Embodied Intelligence, and Visual Generation](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter27_audio_rl/reward-design.md) | Extend RLVR and RLHF to audio, VLA systems, image generation, and video generation. |

A rising training reward only shows that the optimizer met its objective. This final part asks whether the objective was sound, whether the gain was real, and what new risks follow from broader capabilities.

| Ch. | Topic | Through line |
|:--:|:---|:---|
| 25 | [Reward Hacking and RL Evaluation](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter30_alignment_failures/classical-failures.md) | Analyze specification gaming, spurious gains, sleeper behavior, and evaluation leakage. |
| 26 | [Self-Play, Scaling, and Research Frontiers](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter32_selfplay/self-play-outlook/index.md) | Study self-play, RL scaling laws, multi-agent learning, and evolutionary scientific discovery. |

| Appendix | Topic | Contents |
|:--:|:---|:---|
| A | [Training Debugging and Engineering](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/appendix_industrial_training/training-debugging.md) | Training systems, parallelism, monitoring, agent sandboxes, and bad-case analysis. |
| B | [Core Algorithm Implementations](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/appendix_code_cheatsheet/sft-kl.md) | Compact implementations of SFT, PPO, DPO, GRPO, DAPO, sampling, and attention. |
| C | [Learning and Reference Materials](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/appendix_paper_reading/intro.md) | Paper roadmaps, GPU-hour estimates, a metrics glossary, and engineering exercises. |
| D | [Mathematical Foundations of RL](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/appendix_math/linear-algebra-basics.md) | Progressive reviews of linear algebra, probability, calculus, optimization, and information theory. |

The [`code/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code) directory contains runnable examples aligned with course chapters. Each chapter's code is intentionally compact so it can be inspected, run, and modified independently.

| Area | Code Path | Representative Experiments |
|:---|:---|:---|
| Classic control | [`code/chapter01_cartpole/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole) | Train CartPole, inspect rewards and episode length, and compare PPO implementations. |
| Preference fine-tuning | [`code/chapter17_dpo/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter17_dpo) | Train a DPO model and inspect preference accuracy, reward margin, and KL drift. |
| MDP and value learning | [`code/chapter03_mdp/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter03_mdp) | Run bandit strategies, solve GridWorld, and verify Bellman updates numerically. |
| Deep Q-learning | [`code/chapter04_dqn/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter04_dqn) | Implement replay buffers, target networks, and Double DQN variants. |
| Policy gradient | [`code/chapter05_policy_gradient/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient) | Compare REINFORCE, baseline variants, and Actor-Critic updates. |
| PPO | [`code/chapter07_ppo/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo) | Train LunarLander, inspect clipping, visualize GAE, and compare training stability. |
| RLHF | [`code/chapter08_rlhf/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter08_rlhf) | Walk through SFT, reward model training, PPO-style alignment, and veRL/GSM8K adapter scripts. |
| Alignment and RLVR | [`code/chapter09_alignment/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter09_alignment), [`code/chapter09_grpo_rlvr/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter09_grpo_rlvr) | Explore DPO rewards, GRPO group advantages, and rule-based verifiable rewards. |
| VLM and agents | [`code/chapter10_agentic_rl/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter10_agentic_rl), [`code/chapter11_vlm_rl/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter11_vlm_rl) | Build tool-use agent trajectory synthesis and implement multimodal model RL examples. |
| Advanced topics | [`code/chapter12_future_trends/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter12_future_trends) | Study frontier directions including multi-agent RL and model-based RL. |

See [`code/README.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/README.md) for a code index and chapter-specific dependency notes.

For a first systematic reading, follow the chapters in order. Chapters 1–4 establish the language and recursive ideas of RL; Chapters 5–9 develop the algorithmic core of deep RL; Chapters 10–12 expand the data and task settings. Together, these parts provide the foundation for the rest of the book.

Readers focused on LLM post-training can enter Chapters 13–18 after completing Chapters 6–8. Policy gradients, advantage estimation, PPO, and KL constraints directly explain the objectives used by RLHF, DPO, and GRPO. Then select the relevant topics from Chapters 19–24 for Agentic or multimodal RL. Chapters 25–26 are worth reading alongside any experiment because reward and evaluation errors affect every method in the book.

For each chapter: restate the problem it solves, derive the central equation, run at least one experiment, and change one important assumption to explain the resulting metrics. Use the appendices as references when mathematics or engineering details arise; they do not need to be read front to back first.

Published course site:

```
https://walkinglabs.github.io/hands-on-modern-rl/
```

Requirements:

- Node.js >\= 18.0.0
- npm

```
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

Then open the local VitePress URL shown in the terminal, usually:

```
http://localhost:5173
```

Before submitting a pull request that changes documentation structure, theme code, navigation, build scripts, or generated assets, run:

```
npm run verify
```

This checks formatting, lints the VitePress theme, builds the site, and verifies expected build artifacts.

Most code examples use Python and are organized by chapter.

```
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For smaller installs, use chapter-specific requirements files:

```
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

Some chapters may require additional system libraries, GPU support, model downloads, or environment-specific setup. Start with Chapter 01 before running examples that involve LLMs, VLMs, or heavy simulators.

```
hands-on-modern-rl/
|-- docs/                      # VitePress course content
|   |-- .vitepress/            # Site config, navigation, and theme overrides
|   |-- public/                # Static assets copied into the built site
|   |-- preface/               # Course introduction and history
|   |-- chapter*/              # Main course chapters
|   |-- appendix*/             # Supplementary material and references
|   `-- summaries/             # Part-level review and summary notes
|-- code/                      # Runnable examples aligned with chapters
|-- scripts/                   # Maintenance and verification scripts
|-- package.json               # Site scripts and dependencies
|-- AGENTS.md                  # Repository maintenance guide
`-- README.md                  # Main project overview
```

```
npm run dev           # Start the local documentation server
npm run build         # Build the static site
npm run preview       # Preview the built site locally
npm run format        # Format repository files with Prettier
npm run format:check  # Check formatting
npm run lint          # Lint VitePress theme code
npm run verify        # Run format check, lint, build, and artifact verification
```

Contributions should make the course clearer, more accurate, easier to reproduce, or easier to navigate.

Good contributions include:

- Fixing conceptual errors, formulas, diagrams, broken links, or typos.
- Improving explanations without changing the intended learning path.
- Adding small, reproducible experiments that clarify existing chapters.
- Improving scripts, build reliability, navigation, or accessibility.
- Adding high-quality references to papers, official documentation, or widely used open-source implementations.

Please keep pull requests focused. A good PR usually changes one chapter, one experiment, one group of diagrams, or one infrastructure issue at a time.

When adding content:

1. Put course material under [`docs/`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs).
2. Use kebab-case for new directories and files.
3. Prefer directory-based routes with `index.md`.
4. Update [`docs/.vitepress/config.mjs`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/.vitepress/config.mjs) when adding navigable pages.
5. Run `npm run verify` before requesting review if your change touches config, theme, scripts, or generated site output.
6. Use Conventional Commits, such as `docs: clarify ppo clipping` or `fix: repair chapter link`.

For repository-specific maintenance rules, see [`AGENTS.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/AGENTS.md).

Our team has also created other courses. Take a look:

- [**Learn Harness Engineering**](https://github.com/walkinglabs/learn-harness-engineering) — A course on Harness Engineering for AI coding agents. Through 12 lectures and 6 projects, it teaches you to build instructions, state management, verification, and control mechanisms that make model output reliable.
- [**Modern LLM Notebook**](https://github.com/walkinglabs/modern-llm-notebook) — Build modern LLMs from scratch through 23 runnable Jupyter Notebooks in PyTorch, covering Tokenizer, Transformer, training, inference, alignment, and frontier topics.

For suggestions or feedback, scan the QR code to join the discussion group (WeChat):

[![Discussion Group](https://github.com/walkinglabs/.github/raw/main/profile/wechat.png)](https://github.com/walkinglabs/.github/raw/main/profile/wechat.png)

If you use this course in teaching materials, study notes, or derivative non-commercial educational work, please cite the repository:

```
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

We thank [OpenAI](https://openai.com/) for providing development resources and [AMD](https://www.amd.com/) for providing computing resources that support this project. Without their support, this course could not have evolved so quickly.

This course is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE).

You may share and adapt the material for non-commercial purposes, provided that you give appropriate credit and distribute derivative works under the same license.

---
