title: Post-training of LLMs
description: Adapt LLMs for specific tasks and behaviors using post-training techniques like SFT, DPO, and online RL.

# Post-training of LLMs

Learn to post-train and customize an LLM in this short course, **“Post-training of LLMs**,” taught by Banghua Zhu, Assistant Professor at the University of Washington, and co-founder of NexusFlow.

Before a large language model can follow instructions or answer questions, it undergoes two key stages: pre-training and post-training. In pre-training, it learns to predict the next word or token from large amounts of unlabeled text. In post-training, it learns useful behaviors such as following instructions, tool use, and reasoning.

Post-training transforms a general-purpose token predictor—trained on trillions of unlabeled text tokens—into an assistant that follows instructions and performs specific tasks.

In this course, you’ll learn three common post-training methods—Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Online Reinforcement Learning (RL)—and how to use each one effectively. With **SFT**, you train the model on input-output pairs with ideal output responses. With **DPO**, you provide both a preferred (‘chosen’) and a less preferred (‘rejected’) response, and train the model to favor the preferred output. With **RL**, the model generates an output, receives a reward score based on human or automated feedback, and updates the model to improve performance.

You’ll learn the basic concepts, common use-cases, and principles for curating high-quality data for effective training in each of these methods. Through hands-on labs, you’ll download a pre-trained model from HuggingFace and post-train it using SFT, DPO, and RL to see how each technique shapes model behavior.

In detail, you’ll:

- Understand what post-training is, when to use it, and how it differs from pre-training.
- Build an SFT pipeline to turn a base model into an instruct model.
- Explore how DPO reshapes behavior by minimizing contrastive loss—penalizing poor responses and reinforcing preferred ones.
- Implement a DPO pipeline to change the identity of a chat assistant.
- Learn online RL methods like Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), and how to design reward functions.
- Train a model with GRPO to improve its math capabilities using a verifiable reward.

Post-training is one of the most rapidly developing areas of LLM training.

Whether you’re looking to create a safer assistant, fine-tune a model’s tone, or improve task-specific accuracy, this course gives you hands-on experience with the most important techniques shaping how LLMs are post-trained today.
