title: Reinforcement Learning in Production — Vizuara AI Labs
description: A 4-week intensive workshop on Reinforcement Learning in Production. RL actually works — we'll show you.

# Reinforcement Learning\
in Production

RL Actually Works. We'll Show You.

 A 4-week intensive workshop. Train your own personal AI assistant with GRPO. Deploy RL in embodied AI and LLMs in production. Explore the frontier of RL research. 

[Enroll Now →](#enroll) [View Curriculum](#curriculum)

### Fundamentals of RL

MDPs, Bellman equations, value functions, exploration vs exploitation. Building intuition from first principles.

MDPs Bellman Equations Value Iteration Policy Iteration

### Deep Q-Networks (DQN)

From tabular Q-learning to deep Q-networks. Experience replay, target networks, Double DQN, Dueling DQN, and Rainbow.

Q-Learning DQN Double DQN Rainbow

### Policy Gradients & Actor-Critic

REINFORCE, advantage estimation (GAE), A2C/A3C. The policy gradient theorem and variance reduction.

REINFORCE GAE A2C Actor-Critic

### PPO, TRPO & RLHF

Trust regions, clipped objectives, KL penalty. Why PPO is the backbone of RLHF. The full RLHF pipeline — reward modeling, PPO training, and alignment.

PPO TRPO RLHF Reward Modeling Alignment

### GRPO & Its Variations

Group Relative Policy Optimization — the algorithm behind DeepSeek-R1. Online GRPO, Mini-batch GRPO, DAPO, and Dr. GRPO.

GRPO DAPO Dr. GRPO Online GRPO

### DPO, SimPO & Preference Optimization

Direct Preference Optimization and its successors. SimPO's length-normalized formulation, IPO, KTO, ORPO.

DPO SimPO IPO KTO ORPO

### Agentic RL — DeepEyes & Beyond

RL for autonomous agents. DeepEyes for visual reasoning, SWE-RL for code generation, RLEF for multi-turn feedback. The frontier of RL \+ LLMs.

DeepEyes SWE-RL RLEF Agentic RL

### RL Training at Scale

Distributed RL training with veRL and OpenRLHF. Multi-GPU GRPO, Ray integration, vLLM rollout workers, FSDP pipelines.

veRL OpenRLHF Ray Distributed RL

### Environments & Simulation

Building custom RL environments. Gymnasium, MetaDrive for driving, MuJoCo for robotics, Docker-based execution environments.

Gymnasium MetaDrive MuJoCo OpenEnv

### Autonomous Driving with RL

MetaDrive-Arena deep dive. PPO racing agents, multi-agent competition, ELO leaderboards, sim-to-real transfer.

MetaDrive-Arena PPO Racing Multi-Agent ELO

### Agentic RL for Software Engineering

DeepSWE \+ rLLM \+ R2E-Gym stack. RL-powered coding agents that fix real GitHub issues — 59% on SWE-Bench with pure RL.

DeepSWE rLLM R2E-Gym SWE-Bench

### Embodied RL & Humanoid Control

Embodied RL for robotics. Humanoid walking, OpenClaw manipulation, SmolVLA for robot learning, sim-to-real transfer, and reward shaping.

Embodied RL Humanoid OpenClaw SmolVLA Sim2Real

### World Models & Imagination

IRIS world model — act in imagined environments. Latent dynamics, Dreamer architectures, model-based RL for sample efficiency.

IRIS World Models Dreamer Model-Based RL

### Production Deployment & Evaluation

Shipping RL systems. Reward hacking detection, safety constraints, evaluation pipelines, monitoring, RLHF/RLAIF stack.

Production RL Safety Monitoring RLHF Stack

### OpenRLHF

PPO, DPO, GRPO with Ray \+ vLLM for 70B\+ models.

[github.com/OpenRLHF →](https://github.com/OpenRLHF/OpenRLHF)

### OpenEnv

Build and standardize custom RL environments.

[github.com/openenv →](https://github.com/openenv)

### Weights & Biases

Experiment tracking and visualization.

[wandb.ai →](https://wandb.ai)

### MetaDrive Racing Arena

Train PPO agents for competitive 1v1 autonomous racing. Multi-agent environments, ELO leaderboard, sim-to-real transfer.

![Agentic RL for Software Engineering — Issue to Patch to PR](https://rl-production.vizuara.ai/agentic_swe.png)

### Agentic RL for SWE

Build an RL-powered coding agent using DeepSWE \+ rLLM \+ R2E-Gym. Train on 8.1K real GitHub issues. Target: 59% on SWE-Bench Verified.

### OpenClaw: WhatsApp AI with GRPO

Build an open-source WhatsApp AI gateway trained with GRPO on your own conversations. Real-time dashboard, Process Reward Model scoring, asynchronous training on H100 GPUs via RunPod. The model improves while serving responses live.

### SmolVLA Robot Learning

Vision-Language-Action models for robotic control. RL-tuned inference — making small models perform like large ones through smart RL.

![RL2F: Train on OMNI Math, Transfer on Coding](https://rl-production.vizuara.ai/rl2f_chart.png)

### Implementing RL2F: RL with Language Feedback

Implement the RL2F paper from Google DeepMind — a framework that treats in-context learning from feedback as a trainable skill. Build teacher-student didactic interactions, train with multi-turn RL, and reproduce the result where Gemini Flash nearly matches Gemini Pro on HardMath2. Achieve cross-domain generalization to ARC-AGI and Codeforces.

### Teaching Humanoids to Walk

Train a simulated humanoid to walk using RL. Reward shaping, curriculum learning, MuJoCo environments, and locomotion policy transfer.

![IRIS Atari](https://raw.githubusercontent.com/eloialonso/iris/main/assets/iris.gif)

### IRIS World Model

Implement the IRIS world model for imagination-based RL. Latent dynamics, generate training data from imagined trajectories, benchmark on Atari.

### CaP-X RL: The First Coding Agent for Robotics

Reproduce [CaP-X RL](https://arxiv.org/abs/2603.22435) — the first framework to turn frontier LLMs into coding agents that control real robots. Build CaP-Gym (program-synthesis robot environment), benchmark VLMs on CaP-Bench, run CaP-Agent0 on real embodiments, and train CaP-RL with verifiable rewards for sim-to-real transfer with near-zero gap. Outperforms specialized VLA models on perturbed manipulation tasks.

### ML Engineers

You've trained models but never an RL agent. Understand PPO, GRPO, and the RLHF stack powering LLM alignment.

### Graduate Students

You know the theory but haven't shipped production RL. Bridge the gap between papers and real systems.

### Targeting Top AI Labs

Interviewing at OpenAI, DeepMind, Anthropic, NVIDIA? RL systems knowledge is the differentiator.

### Robotics Engineers

You build hardware. Now train the brains. Sim-to-real, humanoid locomotion, dexterous manipulation.

### LLM Practitioners

Understand the RL layer — RLHF, DPO, GRPO — that turns base models into aligned systems.

### Aspiring Researchers

Research roadmaps, paper reading lists, and mentorship to get your first RL paper published.

![DeepSeek Book — RL Chapter by Vizuara AI Labs](https://rl-production.vizuara.ai/deepseek_book.jpeg)

### The RL Chapter in the DeepSeek Book

Dr. Rajat Dandekar authored the reinforcement learning chapter in Manning's DeepSeek book — covering the algorithms, training pipelines, and production techniques that power state-of-the-art reasoning models.

This isn't a team that learned RL from tutorials. Vizuara has the research depth to write the textbook and the engineering experience to ship production systems. When you enroll in this workshop, you're learning from the people publishers trust to explain RL to the world.

### Abhishek Goswami

 Guest speaker sessions are **complimentary** for all students enrolled in Phase 1 or Phase 2. 

![Dr. Rajat Dandekar](https://rl-production.vizuara.ai/rajat.png)

### Dr. Rajat Dandekar

Dr. Dandekar has successfully taught the acclaimed **"Reasoning LLM from Scratch"** course, helping hundreds of students master complex AI concepts through practical, hands-on learning.

With extensive research experience in reinforcement learning and deep learning at top-tier institutions, Dr. Dandekar brings cutting-edge knowledge directly to the classroom. This workshop is born from the conviction that **RL actually works** — and the gap isn't in the algorithms, it's in knowing how to ship them.

### What You Get

#### Personalized Roadmap

A custom research direction tailored to your interests and background in RL.

#### 1:1 Mentorship Sessions

Bi-weekly sessions with Dr. Rajat Dandekar covering research, career, and publication strategy.

#### Paper Reading & Writing

Curated reading lists, code templates, and guidance on writing your first research paper.

#### Publication Support

End-to-end support from idea to submission — venue selection, draft review, and rebuttal strategy.
