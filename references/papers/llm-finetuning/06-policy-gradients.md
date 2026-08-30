[Reinforcement Learning from Human Feedback](https://rlhfbook.com/)

A short introduction to RLHF and post-training focused on language models.

Nathan Lambert

### Chapter Contents

* [Reinforcement Learning](#reinforcement-learning)
  + [The Role of Reinforcement Learning in RLHF](#the-role-of-reinforcement-learning-in-rlhf)
  + [Policy Gradient Algorithms](#policy-gradient-algorithms)
    - [Deriving the Policy Gradient](#deriving-the-policy-gradient)
    - [Vanilla Policy Gradient](#vanilla-policy-gradient)
    - [REINFORCE](#reinforce)
    - [REINFORCE Leave One Out (RLOO)](#reinforce-leave-one-out-rloo)
    - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
    - [Understanding the PPO Objective](#understanding-the-ppo-objective)
    - [Value Functions and PPO](#value-functions-and-ppo)
    - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
    - [Group Sequence Policy Optimization (GSPO)](#group-sequence-policy-optimization-gspo)
    - [Clipped Importance Sampling Policy Optimization (CISPO)](#clipped-importance-sampling-policy-optimization-cispo)
    - [Comparing Algorithms](#comparing-algorithms)
  + [Implementation](#implementation)
    - [Policy-Gradient Basics](#policy-gradient-basics)
    - [Loss Aggregation Tradeoffs](#loss-aggregation-tradeoffs)
    - [Asynchronous RL Systems](#asynchronous-rl-systems)
    - [Truncated Importance Sampling](#truncated-importance-sampling)
    - [Example: PPO](#example-ppo)
    - [Example: GRPO](#example-grpo)
  + [Auxiliary Topics](#auxiliary-topics)
    - [Generalized Advantage Estimation (GAE)](#generalized-advantage-estimation-gae)
    - [Double Regularization](#double-regularization)
    - [Further Reading](#further-reading)
  + [Suggested Experiments](#suggested-experiments)
* [Bibliography](#bibliography)
- [Lecture 3: Understanding Policy Gradient Algorithms for RL on LLMs](https://www.youtube.com/watch?v=K_Sj_-1BUMM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=4)
- [Lecture 4: Implementing RL Algorithms for LLMs](https://www.youtube.com/watch?v=i-AIMpZHgeg&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=5)

# Reinforcement Learning

In the RLHF process, the reinforcement learning algorithm slowly updates the model’s weights with respect to feedback from a reward model. The policy – the model being trained – generates completions to prompts in the training set, then the reward model scores them, and the reinforcement learning optimizer takes gradient steps based on this information (see fig. [1](#fig:rlhf-overview) for an overview). This chapter explains the mathematics and trade-offs across various algorithms used to learn from the signal the reward model gives to on-policy data. These algorithms are run for a period of many epochs, often thousands or millions of batches across a larger set of prompts, with gradient updates in between each of them.

## The Role of Reinforcement Learning in RLHF

The algorithms that popularized RLHF for language models were policy-gradient reinforcement learning algorithms. These algorithms, such as Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and REINFORCE, use recently generated samples to update their model (rather than storing scores in a replay buffer like algorithms, e.g. Deep Q-Networks, DQN, used in popular projects such as AlphaGo). In this section we will cover the fundamentals of the policy gradient algorithms and how they are used in the modern RLHF framework.

At a machine learning level, this section is the subject with the highest complexity in the RLHF process. However, as with most modern AI models, the largest determining factor in its success is the data provided as inputs to the process.

![Figure 1: Overview of the RLHF training loop. A prompt from the dataset is passed to the tuned policy, which generates a completion. The reward model scores this completion, while the frozen initial model (typically the instruction-tuned model before RL) computes log probabilities on the same text to calculate a KL penalty that prevents excessive drift. The combined reward signal then drives a reinforcement learning update to the policy parameters.](images/rlhf-overview.png)

Figure 1: Overview of the RLHF training loop. A prompt from the dataset is passed to the tuned policy, which generates a completion. The reward model scores this completion, while the frozen initial model (typically the instruction-tuned model before RL) computes log probabilities on the same text to calculate a KL penalty that prevents excessive drift. The combined reward signal then drives a reinforcement learning update to the policy parameters.

When RLHF came onto the scene with ChatGPT, it was largely known that they used a variant of PPO, and many initial efforts were built upon that. Over time, multiple research projects showed the promise of REINFORCE-style algorithms [[1]](#ref-ahmadian2024back) [[2]](#ref-wang2024helpsteer2p), touted for their simplicity over PPO without a separate value model (saves memory and therefore the number of GPUs required) and with simpler advantage estimation (no Generalized Advantage Estimation, GAE, which is a method to compute advantages used for variance reduction in policy gradient algorithms). More algorithms have emerged, including Group Relative Policy Optimization, which is particularly popular with reasoning tasks, but in general many of these algorithms can be tuned to fit a specific task. In this chapter, we cover the core policy gradient setup and the three algorithms mentioned above due to their central role in the establishment of a canonical RLHF literature.

At its simplest, the RL stage of RLHF requires two models: a policy (the model being trained) and a reward model that scores its outputs (as covered in the previous chapter). A copy of the policy before RL serves as the reference model for computing a KL penalty (this model is frozen, i.e. it is not updated with gradients from the automatic differentiation engine). The most complex algorithm covered here, PPO, adds a fourth model – a learned value function used to estimate how good each token in the action was, also a large language model updated during training. The algorithms in this chapter differ mainly in how they estimate a quantity called *advantages* – a measure of how good the current action (completion) from the model is relative to average – and how they constrain policy updates so the optimization is numerically stable. A visual overview of this RLHF process (without the value model) is shown in fig. [1](#fig:rlhf-overview).

For definitions of symbols, see the problem setup chapter.

*This chapter uses \((s, a)\) notation from the reinforcement learning literature, where \(s\) denotes states and \(a\) denotes actions. In the language model context, you will often see \((x, y)\) instead, where \(x\) is the prompt and \(y\) is the completion. The \((s, a)\) framing is more general—these algorithms were designed for sequential decision problems where actions are taken at each timestep. However, many RLHF implementations treat the entire completion as a single action, making the \((x, y)\) notation equally valid.*

***RL Cheatsheet:** A one-page reference of all core RL loss functions from this chapter is available at [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet).*

## Policy Gradient Algorithms

At its core, this chapter is dedicated to understanding the following shape of equation. This equation is computing the gradient, \(\Delta \theta\), to the language model we are training, \(\pi\_\theta\):

\[\Delta \theta \propto \Psi\_t \, \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\qquad{(1)}\]

Here, the equation is composed of two key components: 1. \(\nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\) — which direction in parameter space makes action \(a\_t\) more likely. 2. \(\Psi\_t\) — how good was it? A scalar scoring the outcome.

When you put this together, yes, by multiplying the quantities, you get the policy gradient update. Some things are simple, such as that \(\Psi\_t > 0\) updates parameters to make \(a\_t\) more likely, \(\Psi\_t < 0\) updates them to make it less likely. The policy gradient is computing which parameters contribute to an action and if we should make it more or less likely to occur in the future. The rest of this chapter goes very deep on the different ways to do this, and what the specific tricks are to make it work for LLMs.

Now, let us formalize this a bit further. Reinforcement learning algorithms are designed to maximize the future, discounted reward across a trajectory of states, \(s \in \mathcal{S}\), and actions, \(a \in \mathcal{A}\) (for more notation, see Appendix A, Definitions). The objective of the agent, often called the *return*, is the sum of discounted rewards starting at a given time \(t\) (where \(\gamma\in [0,1]\) is a factor that prioritizes near-term rewards):

\[G\_t = r\_t + \gamma r\_{t+1} + \cdots = \sum\_{k=0}^\infty \gamma^k r\_{t+k}.\qquad{(2)}\]

The return definition can also be written recursively as: \[G\_{t} = r\_t + \gamma G\_{t+1}.\qquad{(3)}\]

This return is the basis for learning a value function \(V(s)\) that is the estimated future return given a current state:

\[V(s) = \mathbb{E}\left[G\_t \mid S\_t = s \right].\qquad{(4)}\]

All policy gradient algorithms optimize a policy \(\pi\_\theta(a\mid s)\) to maximize expected return; this objective can be expressed using the induced value function \(V^{\pi\_\theta}(s)\).

Let \(d\_0(s)\) be the initial-state distribution. The episodic objective we maximize can be written as: \[
J(\theta)
\;=\;
\sum\_{s} d\_0(s) V^{\pi\_\theta}(s),
\qquad{(5)}\]

In a finite MDP this is a sum over possible starting states, but in practice we never compute it exactly. Instead, we estimate it from data by sampling rollouts from the current policy. In RLHF this typically means sampling prompts \(x\_i\) from a dataset and generating completions \(y\_i \sim \pi\_\theta(\cdot\mid x\_i)\). Let \(R(x\_i, y\_i)\) denote the scalar sequence-level reward assigned to that prompt-completion pair; if \(\tau\_i\) is the corresponding episode, this is the trajectory reward \(R(\tau\_i)\). We then take an empirical average such as:

\[
\hat{J}(\theta) = \frac{1}{B}\sum\_{i=1}^{B} R(x\_i, y\_i),
\qquad{(6)}\]

or, in an MDP view with per-step rewards,

\[
\hat{J}(\theta) = \frac{1}{B}\sum\_{i=1}^{B} \sum\_{t=0}^{T\_i} \gamma^t r\_{i,t}.
\qquad{(7)}\]

In practice, RLHF for language models sets \(\gamma = 1\) (no discounting) because the unit of optimization is the collective completion, not individual tokens – this choice is discussed further in the MDP vs. Bandit section later in this chapter.

The core of policy gradient algorithms is computing the gradient with respect to the finite-time expected return over the current policy. With this expected return, \(J\), the parameter update can be computed as follows, where \(\alpha\) is the learning rate:

\[\theta \leftarrow \theta + \alpha \nabla\_\theta J(\theta)\qquad{(8)}\]

The core implementation detail is how to compute said gradient.

### Deriving the Policy Gradient

Let \(p\_\theta(\tau)\) denote the trajectory distribution induced by the initial-state distribution \(d\_0\), the policy \(\pi\_\theta\), and the environment transition dynamics, as expanded in eq. [11](#eq:trajectory_probability) below. Another way to pose the RL objective we want to maximize is as follows: \[
J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ R(\tau) \right],
\qquad{(9)}\]

where \(\tau = (s\_0, a\_0, s\_1, a\_1, \ldots)\) is a trajectory and \(R(\tau) = \sum\_{t=0}^\infty r\_t\) is the total reward of the trajectory. Alternatively, we can write the expectation as an integral over all possible trajectories: \[
J(\theta) = \int\_\tau p\_\theta (\tau) R(\tau) d\tau
\qquad{(10)}\]

Notice that we can express the trajectory probability as follows, where \(\pi\_\theta(a\_t|s\_t) p(s\_{t+1}|s\_t, a\_t)\) combines the policy probability with the environment transition probability from one state-action pair to the next state: \[
p\_\theta (\tau) = d\_0(s\_0) \prod\_{t=0}^\infty \pi\_\theta(a\_t|s\_t) p(s\_{t+1}|s\_t, a\_t),
\qquad{(11)}\]

If we take the gradient of the objective (eq. [9](#eq:policy_objective_expectation)) with respect to the policy parameters \(\theta\): \[
\nabla\_\theta J(\theta) = \int\_\tau \nabla\_\theta p\_\theta (\tau) R(\tau) d\tau
\qquad{(12)}\]

Notice that we can use the [log-derivative trick](https://andrewcharlesjones.github.io/journal/log-derivative.html) in order to rewrite the gradient of the integral as an expectation: \[
\begin{aligned}
\nabla\_\theta \log p\_\theta(\tau) &= \frac{\nabla\_\theta p\_\theta(\tau)}{p\_\theta(\tau)} &\text{(from chain rule)} \\
\implies \nabla\_\theta p\_\theta(\tau) &= p\_\theta(\tau) \nabla\_\theta \log p\_\theta(\tau) &\text{(rearranging)}
\end{aligned}
\qquad{(13)}\]

Using this log-derivative trick: \[
\begin{aligned}
\nabla\_\theta J(\theta) &= \int\_\tau \nabla\_\theta p\_\theta (\tau) R(\tau) d\tau \\
&= \int\_\tau p\_\theta (\tau) R(\tau) \nabla\_\theta \log p\_\theta (\tau) d\tau \\
&= \mathbb{E}\_{\tau \sim p\_\theta} \left[ R(\tau) \nabla\_\theta \log p\_\theta (\tau) \right]
\end{aligned}
\qquad{(14)}\]

Where the final step uses the definition of an expectation under the trajectory distribution \(p\_\theta(\tau)\): for any function \(f\), \(\mathbb{E}\_{\tau \sim p\_\theta}[f(\tau)] = \int\_\tau f(\tau)\,p\_\theta(\tau)\,d\tau\) (or a sum in the discrete case). Writing it as an expectation is useful because we can approximate it with Monte Carlo rollouts, e.g., \(\frac{1}{B}\sum\_{i=1}^{B} f(\tau\_i)\) for trajectories \(\tau\_i \sim p\_\theta\) induced by the current policy.

Back to the derivation, expanding the log probability of the trajectory:

\[
\log p\_\theta (\tau) = \log d\_0(s\_0) + \sum\_{t=0}^\infty \log \pi\_\theta(a\_t|s\_t) + \sum\_{t=0}^\infty \log p(s\_{t+1}|s\_t, a\_t)
\qquad{(15)}\]

Now, if we take the gradient of the above, we get:

* \(\nabla\_\theta \log d\_0(s\_0) = 0\) (initial state distribution doesn’t depend on \(\theta\))
* \(\nabla\_\theta \log p(s\_{t+1}|s\_t, a\_t) = 0\) (environment transition dynamics don’t depend on \(\theta\))
* only \(\nabla\_\theta \log \pi\_\theta(a\_t|s\_t)\) survives

Therefore, the gradient of the log probability of the trajectory simplifies to: \[
\nabla\_\theta \log p\_\theta (\tau) = \sum\_{t=0}^\infty \nabla\_\theta \log \pi\_\theta(a\_t|s\_t)
\qquad{(16)}\]

Reaching this equation is a crucial point in the implementation. Here, we have gone far enough to see that the gradient of the trajectory distribution reduces to a sum of gradients from language model policy probabilities (which are just the probabilities of tokens given by the model we’re training). In practice, this results in a common form of the policy gradient equations. They end up looking like a sum of log-probabilities in the loss, and then we compute the gradients via autodiff. A short snippet you’ll see again and again roughly follows:

```
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

You’ll see this throughout the chapter. Now, back to the formal policy gradient mathematics.

Substituting this back in eq. [14](#eq:policy_gradient_expectation), we get: \[
\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ \sum\_{t=0}^\infty R(\tau) \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \right]
\qquad{(17)}\]

Quite often, people use a more general formulation of the policy gradient: \[
g = \nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ \sum\_{t=0}^\infty \Psi\_t \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \right]
\qquad{(18)}\]

Where \(\Psi\_t\) can be the following (where the rewards can also often be discounted by \(\gamma\)), a taxonomy adopted from Schulman et al. 2015 [[3]](#ref-schulman2015high):

1. \(R(\tau) = \sum\_{t=0}^{\infty} r\_t\): total reward of the trajectory.
2. \(\sum\_{t'=t}^{\infty} r\_{t'}\): reward following action \(a\_t\), also described as the return from time \(t\), \(G\_t\).
3. \(\sum\_{t'=t}^{\infty} r\_{t'} - b(s\_t)\): baselined version of previous formula.
4. \(Q^{\pi}(s\_t, a\_t)\): state-action value function.
5. \(A^{\pi}(s\_t, a\_t)\): advantage function, which yields the lowest possible theoretical variance if it can be computed accurately.
6. \(r\_t + \gamma V^{\pi}(s\_{t+1}) - V^{\pi}(s\_t)\): Temporal Difference (TD) residual.

The *baseline* is a value used to reduce variance of policy updates (more on this below).

For language models, some of these concepts do not make as much sense. For example, for a deterministic policy \(\pi\) the state value is \(V^{\pi}(s\_t) = Q^{\pi}(s\_t, \pi(s\_t))\) (and for the optimal value function one has \(V^\*(s\_t)=\max\_{a\_t} Q^\*(s\_t,a\_t)\)). For a stochastic policy, the analogous identity is \(V^{\pi}(s\_t) = \mathbb{E}\_{a\_t \sim \pi(\cdot\mid s\_t)}\!\left[Q^{\pi}(s\_t,a\_t)\right]\). The Bellman equation relates Q to V: in general \(Q^\pi(s\_t,a\_t) = \mathbb{E}\!\left[r\_t + \gamma V^\pi(s\_{t+1}) \mid s\_t, a\_t\right]\), but for language models where state transitions are deterministic, this simplifies to \(Q(s\_t,a\_t) = r\_t + \gamma V(s\_{t+1})\). The advantage function measures how much better action \(a\_t\) is compared to the average:

\[A(s\_t,a\_t) = Q(s\_t,a\_t) - V(s\_t) = r\_t + \gamma V(s\_{t+1}) - V(s\_t)\qquad{(19)}\]

This final form is exactly the temporal difference (TD) residual (item 6 above) – a fundamental quantity in RL that measures the gap between the value function’s prediction and what actually occurred, driving value function updates toward more accurate estimates. In practice, a learned value function \(\hat{V}\) is used to estimate the advantage via this TD error.

### Vanilla Policy Gradient

The vanilla policy gradient implementation optimizes the above expression for \(J(\theta)\) by differentiating with respect to the policy parameters. A simple version, with respect to the time-\(t\) return, is:

\[\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ \sum\_{t=0}^T G\_t \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \right]\qquad{(20)}\]

A common problem with vanilla policy gradient algorithms is the high variance in gradient updates, which can be mitigated in multiple ways. The high variance comes from the gradient updates being computed by estimating the return \(G\) from an often small set of rollouts in the environment that tend to be susceptible to noise (e.g. the stochastic nature of generating from language models with temperature \(>0\)). The variance across return estimates is higher in domains with sparse rewards, as more of the samples are 0 or 1, rather than closely clustered. In order to alleviate this, various techniques are used to normalize the value estimation, called *baselines*. Baselines accomplish this in multiple ways, effectively normalizing by the value of the state relative to the downstream action (e.g. in the case of Advantage, which is the difference between the Q value and the value). The simplest baselines are averages over the batch of rewards or a moving average. Even these action-independent baselines can reduce variance without changing the expected gradient, since \(\mathbb{E}\_{a \sim \pi(a|s)}\!\left[b(s) \nabla\_\theta \log \pi\_\theta(a|s)\right] = 0\) for any state-dependent \(b(s)\), improving the learning signal substantially.

Many of the policy gradient algorithms discussed in this chapter build on the advantage formulation of policy gradient:

\[\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ \sum\_{t=0}^T A^{\pi\_\theta}(s\_t, a\_t) \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \right]\qquad{(21)}\]

### REINFORCE

The algorithm REINFORCE is likely a backronym, but the components of the algorithm it represents are quite relevant for modern reinforcement learning algorithms. Defined in the seminal paper *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [[4]](#ref-williams1992simple):

> The name is an acronym for “REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility.”

The three components of this are how to do the *reward increment*, a.k.a. the policy gradient step. It has three pieces to the update rule:

1. Nonnegative factor: This is the learning rate (step size) that must be a positive number, e.g. \(\alpha\) below.
2. Offset Reinforcement: This is a baseline \(b\) or other normalizing factor of the reward to improve stability.
3. Characteristic Eligibility: This attributes the scalar reward signal to the parameters that produced the action. Williams denotes this eligibility term as \(e\) (not the exponential function). In modern policy-gradient notation, it corresponds to \(\nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\).

Thus, the form looks quite familiar:

\[ \Delta\_\theta = \alpha(r - b)e \qquad{(22)}\]

With more modern notation and the generalized return \(G\), the REINFORCE operator appears as:

\[
\nabla\_{\theta}\,J(\theta)
\;=\;
\mathbb{E}\_{\tau \sim p\_{\theta}}\!\left[
\sum\_{t=0}^{T}
(G\_t - b(s\_t))\,\nabla\_{\theta} \log \pi\_{\theta}(a\_t \mid s\_t)
\right],
\qquad{(23)}\]

Here, the value \(G\_t - b(s\_t)\) is the *advantage* of the policy at the current state, so we can reformulate the policy gradient in a form that we continue later with the advantage, \(A\):

\[
\nabla\_{\theta}\,J(\theta)
\;=\;
\mathbb{E}\_{\tau \sim p\_{\theta}}\!\left[
\sum\_{t=0}^{T}
A\_t\,\nabla\_{\theta} \log \pi\_{\theta}(a\_t \mid s\_t)
\right],
\qquad{(24)}\]

REINFORCE is a specific implementation of vanilla policy gradient that uses a Monte Carlo estimator of the gradient.

![Figure 2: Basic REINFORCE architecture for language models. The shaped reward combines the reward model score with a KL penalty from the reference model. We build on this structure throughout the chapter.](images/reinforce_tikz.png)

Figure 2: Basic REINFORCE architecture for language models. The shaped reward combines the reward model score with a KL penalty from the reference model. We build on this structure throughout the chapter.

### REINFORCE Leave One Out (RLOO)

The core implementation detail of REINFORCE Leave One Out versus standard REINFORCE is that it takes the average reward of the *other* samples in the batch to compute the baseline – rather than averaging over all rewards in the batch [[5]](#ref-huang2024putting), [[1]](#ref-ahmadian2024back), [[6]](#ref-kool2019buy). By excluding the current sample’s reward from its own baseline, the RLOO baseline is independent of the action being evaluated, which keeps the gradient estimator exactly unbiased.

Crucially, this only works when generating multiple trajectories (completions) per state (prompt), which is common practice in multiple domains of fine-tuning language models with RL.

Specifically, for the REINFORCE Leave-One-Out (RLOO) baseline, given \(K\) sampled trajectories (actions taken conditioned on a prompt) \(a\_1, \dots, a\_K\), to a given prompt \(s\) we define the baseline explicitly as the following *per-prompt*:

\[
b(s, a\_k) = \frac{1}{K-1}\sum\_{i=1, i\neq k}^{K} R(s, a\_i),
\qquad{(25)}\]

resulting in the advantage:

\[
A(s, a\_k) = R(s, a\_k) - b(s, a\_k).
\qquad{(26)}\]

Equivalently, this can be expressed as:

\[
A(s, a\_k) = \frac{K}{K - 1}\left(R(s, a\_k) - \frac{1}{K}\sum\_{i=1}^{K} R(s, a\_i)\right).
\qquad{(27)}\]

This is a simple, low-variance *per-prompt* advantage estimate that is closely related to the group-relative advantage used in Group Relative Policy Optimization, GRPO (discussed shortly, after Proximal Policy Optimization, PPO). In practice, GRPO-style training mainly differs in how it applies the KL regularizer (as an explicit loss term vs. folded into the reward) and whether it uses PPO-style ratio clipping. To be specific, the canonical GRPO implementation applies the KL penalty at the loss level, whereas the derivation for RLOO or traditional policy-gradients applies the KL penalty to the reward itself. With the transition from RLHF to reasoning and reinforcement learning with verifiable rewards (RLVR), the prevalence of KL penalties has decreased overall, with many reasoning adaptations of RLHF code turning them off entirely. Still, the advantage from RLOO could be combined with the clipping of PPO, showing how similar many of these algorithms are.

RLOO and other algorithms that do not use a value network – an additional model copy (a critic) that predicts a scalar value \(V(s\_t)\) per token – assign the same sequence-level advantage (or reward) to every token when computing the loss. Algorithms that use a learned value network, such as PPO, assign a different value to every token individually, discounting from the final reward achieved at the EOS token. With a KL distance penalty, RLOO aggregates the per-token KL over the completion and folds that scalar into the sequence reward, so the resulting advantage is broadcast to all tokens. PPO subtracts a per-token KL from the per-token reward before computing \(A\_t\), giving token-level credit assignment. GRPO typically retains a sequence-level advantage but adds a separate per-token term to the loss, rather than subtracting it from the reward. These details and trade-offs are discussed later in the chapter.

![Figure 3: REINFORCE Leave-One-Out (RLOO) architecture. Multiple completions per prompt provide a leave-one-out baseline for advantage estimation without learning a value function.](images/rloo_tikz.png)

Figure 3: REINFORCE Leave-One-Out (RLOO) architecture. Multiple completions per prompt provide a leave-one-out baseline for advantage estimation without learning a value function.

### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) [[7]](#ref-schulman2017proximal) is one of the foundational algorithms behind Deep RL’s successes (such as OpenAI Five, which mastered Dota 2 [[8]](#ref-berner2019dota) and large amounts of research). The objective that PPO maximizes, with respect to the advantages and the policy probabilities, is as follows:

\[J(\theta) = \min\left(\frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).\qquad{(28)}\]

Here, \(\pi\_\theta(a|s)\) is the current policy being optimized and \(\pi\_{\theta\_{\text{old}}}(a|s)\) is the policy that was used to collect the training data (i.e., the policy from the previous iteration). The ratio between these two policies emerges from *importance sampling*, which allows us to reuse data collected under an old policy to estimate gradients for a new policy.

Recall from the advantage formulation of the policy gradient (eq. [21](#eq:advantage_policy_gradient)) that we have: \[\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim p\_\theta} \left[ \sum\_{t=0}^T A^{\pi\_\theta}(s\_t, a\_t) \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \right].\qquad{(29)}\]

This expectation is taken over trajectories sampled from the trajectory distribution induced by \(\pi\_\theta\), but in practice we want to take multiple gradient steps on a batch of data that was collected from a fixed policy \(\pi\_{\theta\_{\text{old}}}\). To correct for this distribution mismatch, we multiply by the importance weight \(\frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)}\), which reweights samples to account for how much more or less likely they are under the current policy versus the data-collection policy. Without constraints, optimizing this importance-weighted objective can lead to destructively large policy updates when the ratio diverges far from 1. PPO addresses this by clipping the ratio to the range \([1-\varepsilon, 1+\varepsilon]\), ensuring that the policy cannot change too drastically in a single update.

Note that, as we move to PPO and its peer algorithms, we often work with the *objective* rather than an explicit gradient. This is because the PPO objective does *not* have an easily interpretable analytical gradient once the \(\min\) and clipping operations are included (the gradient has ~4 terms corresponding to the regions in fig. [5](#fig:ppo-obj), depending on how it is written); writing the objective is simply the clearer way to convey these algorithms.

For completeness, PPO is typically written as an *expected* clipped surrogate objective over timesteps:

\[
J(\theta)
=
\mathbb{E}\_{t}\left[
\min\left(\rho\_t(\theta)A\_t,\ \text{clip}(\rho\_t(\theta),1-\varepsilon,1+\varepsilon)A\_t\right)
\right],
\qquad
\rho\_t(\theta)=\frac{\pi\_\theta(a\_t\mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t\mid s\_t)}.
\qquad{(30)}\]

The objective is often converted into a loss function by simply adding a negative sign, which makes the optimizer seek to make it as negative as possible.

For language models, the objective (or loss) is computed per token, which intuitively can be grounded in how one would compute the probability of the entire sequence of autoregressive predictions – by a product of probabilities. From there, the common implementation is with *log-probabilities* that make the computation simpler to perform in modern language modeling frameworks. In practice, one computes the difference of token log-probabilities and exponentiates it to recover the policy ratio \(\rho\_t\).

\[ J(\theta) = \frac{1}{|a|} \sum\_{t=0}^{|a|} \min\left(\frac{\pi\_\theta(a\_{t}|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_{t}|s\_t)}A\_{t}, \text{clip} \left( \frac{\pi\_\theta(a\_{t}|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_{t}|s\_t)}, 1-\varepsilon, 1+\varepsilon \right) A\_{t} \right). \qquad{(31)}\]

This is the per-token version of PPO, which also applies to other policy-gradient methods, but is explored further later in the implementation section of this chapter. Here, the term for averaging by the number of tokens in the action, \(\frac{1}{|a|}\), comes from common implementation practices, but is not in a formal derivation of the loss (shown in [[9]](#ref-liu2025understanding)).

![Figure 4: PPO framework. A learned value function enables Generalized Advantage Estimation (GAE) for per-token advantages, used with a clipped surrogate objective.](images/ppo_tikz.png)

Figure 4: PPO framework. A learned value function enables Generalized Advantage Estimation (GAE) for per-token advantages, used with a clipped surrogate objective.

Here we will explain the different cases this loss function triggers given various advantages and policy ratios. At an implementation level, the inner computations for PPO involve two main terms: 1) a standard policy gradient with a learned advantage and 2) a clipped policy gradient based on a maximum step size.

To understand how different situations emerge, we can define the policy ratio as:

\[\rho(\theta) = \frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)}\qquad{(32)}\]

The policy ratio is a centerpiece of PPO and related algorithms. It emerges from computing the gradient of a policy and controls the parameter updates in a very intuitive way. For any batch of data, the policy ratio starts at 1 for the first gradient step for that batch, since \(\pi\_{\theta}\) is the same as \(\pi\_{\theta\_{\text{old}}}\) at this point. Then, in the next gradient step, the policy ratio will be above one if that gradient step increased the likelihood of certain tokens with an associated positive advantage, or less than one for the other case. A common practice is to take 1-4 gradient steps per batch with policy gradient algorithms before updating \(\pi\_{\theta\_{\text{old}}}\).

### Understanding the PPO Objective

Overall, the PPO objective can be visualized by two lines of a plot of objective versus policy ratio, which is shown in fig. [5](#fig:ppo-obj). The PPO objective is maximized by changing the probability of the sampled actions. Numerically, the objective controls for both positive and negative advantage cases by clever use of the minimum operation, making it so the update is at most pushed by an epsilon distance away from a policy ratio of 1.

Within the trust region, PPO operates the same as other policy gradient algorithms. This is by design! The trust region is a concept used to cap the maximum step size of PPO and its peer algorithms for stability of updates. The core of the PPO algorithm, the clip and min/max functions, define this region. The objective becomes flat outside of it.

The idea of a “trust region” comes from the numerical optimization literature [[10]](#ref-nocedal2006numerical), but was popularized within Deep RL from the algorithm Trust Region Policy Optimization (TRPO), which is accepted as the predecessor to PPO [[11]](#ref-schulman2015trust). The trust region is the area where the full policy-gradient steps are applied, as the updates are not “clipped” by the max/min operations of the PPO objective.

![Figure 5: Visualization of the PPO objective J(\theta) as a function of the policy ratio \rho(\theta), for both positive and negative advantage. Within each panel, the three ratio regions are annotated with their unclipped term, clipped term, resulting objective, and gradient.](images/ppo-clip-viz.png)

Figure 5: Visualization of the PPO objective \(J(\theta)\) as a function of the policy ratio \(\rho(\theta)\), for both positive and negative advantage. Within each panel, the three ratio regions are annotated with their unclipped term, clipped term, resulting objective, and gradient.

The policy ratio and advantage together can occur in a few different configurations, which fig. [5](#fig:ppo-obj) enumerates by the sign of the advantage \(A\_t\) and by which of the three regions the policy ratio \(\rho(\theta)\) falls into. Two facts determine the outcome in every region: the sign of the advantage sets whether we want to make the action more or less likely, and the \(\min\) operation selects either the unclipped term \(\rho(\theta) A\_t\) or its clipped counterpart.

The clipping only zeroes out the gradient in the two regions where the policy has *already* moved the sampled action in the desired direction, past the edge of the trust region:

* **Positive advantage and \(\rho(\theta) > 1+\varepsilon\)**: the action is already substantially more likely under \(\pi\_\theta\) than under \(\pi\_{\theta\_{\text{old}}}\). The objective saturates at \((1+\varepsilon)A\_t\), its gradient is zero, and no update is made — we avoid over-reinforcing an action that is already more expressed.
* **Negative advantage and \(\rho(\theta) < 1-\varepsilon\)**: the action is already substantially less likely under \(\pi\_\theta\). The objective saturates at \((1-\varepsilon)A\_t\), its gradient is again zero, and no update is made — we avoid over-suppressing an action that is already discouraged.

Everywhere else the unclipped term \(\rho(\theta) A\_t\) is active and PPO takes a standard policy-gradient step: increasing the action’s probability when \(A\_t > 0\) and decreasing it when \(A\_t < 0\). We can read off fig. [5](#fig:ppo-obj) in terms of what each region asks of the updated policy \(\pi\_\theta\):

* the sloped, unclipped region under a positive advantage (green) **increases** the probability of the sampled action;
* the sloped, unclipped region under a negative advantage (red) **decreases** it;
* the flat, clipped region (grey) leaves the policy **unchanged**, since its gradient is zero.

The same regions, written out term by term:

#### Positive Advantage (\(A\_t > 0\))

This means that the action taken was beneficial according to the value function, and we want to increase the likelihood of taking that action in the future. Now, let’s look at different cases for the policy ratio \(\rho(\theta)\):

1. \(\rho(\theta) < 1 - \varepsilon\):

   * **Interpretation**: Action is less likely with the new policy than the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \((1 - \varepsilon) A\_t\)
   * **Objective**: \(\rho(\theta) A\_t\)
   * **Gradient**: \(\nabla\_\theta \rho(\theta) A\_t \neq 0\)
   * **What happens**: Normal policy-gradient update - increase likelihood of action
2. \(1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon\):

   * **Interpretation**: Action is almost equally likely with the new policy as the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \(\rho(\theta) A\_t\)
   * **Objective**: \(\rho(\theta) A\_t\)
   * **Gradient**: \(\nabla\_\theta \rho(\theta) A\_t \neq 0\)
   * **What happens**: Normal policy-gradient update - increase likelihood of action
3. \(1 + \varepsilon < \rho(\theta)\):

   * **Interpretation**: Action is more likely with the new policy than the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \((1 + \varepsilon) A\_t\)
   * **Objective**: \((1 + \varepsilon) A\_t\)
   * **Gradient**: \(\nabla\_\theta (1 + \varepsilon) A\_t = 0\)
   * **What happens**: NO UPDATE - action is already more likely under the new policy

To summarize, when the advantage is positive (\(A\_t>0\)), we want to boost the probability of the action. Therefore:

* We perform gradient steps only in the case when \(\pi\_{\text{new}}(a) \leq (1+\varepsilon) \pi\_{\text{old}}(a)\). Intuitively, we want to boost the probability of the action, since the advantage was positive, but not boost it so much that we have made it substantially more likely.
* Crucially, when \(\pi\_{\text{new}}(a) > (1+\varepsilon) \pi\_{\text{old}}(a)\), then we don’t perform any update, and the gradient of the clipped objective is \(0\). Intuitively, the action is already more expressed with the new policy, so we don’t want to over-reinforce it.

#### Negative Advantage (\(A\_t < 0\))

This means that the action taken was detrimental according to the value function, and we want to decrease the likelihood of taking that action in the future. Now, let’s look at different cases for the policy ratio \(\rho(\theta)\):

1. \(\rho(\theta) < 1 - \varepsilon\):

   * **Interpretation**: Action is less likely with the new policy than the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \((1 - \varepsilon) A\_t\)
   * **Objective**: \((1 - \varepsilon) A\_t\)
   * **Gradient**: \(\nabla\_\theta (1 - \varepsilon) A\_t = 0\)
   * **What happens**: NO UPDATE - action is already less likely under the new policy
2. \(1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon\):

   * **Interpretation**: Action is almost equally likely with the new policy as the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \(\rho(\theta) A\_t\)
   * **Objective**: \(\rho(\theta) A\_t\)
   * **Gradient**: \(\nabla\_\theta \rho(\theta) A\_t \neq 0\)
   * **What happens**: Normal policy-gradient update - decrease likelihood of action
3. \(1 + \varepsilon < \rho(\theta)\):

   * **Interpretation**: Action is more likely with the new policy than the old policy
   * **Unclipped Term**: \(\rho(\theta) A\_t\)
   * **Clipped Term**: \((1 + \varepsilon) A\_t\)
   * **Objective**: \(\rho(\theta) A\_t\)
   * **Gradient**: \(\nabla\_\theta \rho(\theta) A\_t \neq 0\)
   * **What happens**: Normal policy-gradient update - decrease likelihood of action

To summarize, when the advantage is negative (\(A\_t < 0\)), we want to decrease the probability of the action. Therefore:

* We perform gradient steps only in the case when \(\pi\_{\text{new}}(a) \geq (1-\varepsilon) \pi\_{\text{old}}(a)\). Intuitively, we want to decrease the probability of the action, since the advantage was negative, and we do so proportional to the advantage.
* Crucially, when \(\pi\_{\text{new}}(a) < (1-\varepsilon) \pi\_{\text{old}}(a)\), then we don’t perform any update, and the gradient of the clipped objective is \(0\). Intuitively, the action is already less likely under the new policy, so we don’t want to over-suppress it.

It is crucial to remember that PPO within the trust region is roughly the same as standard forms of policy gradient.

### Value Functions and PPO

The value function within PPO is an additional copy of the model that is used to predict the value per token. The value of a token (or state) in traditional RL is predicting the future return from that moment, often with discounting. This value in PPO is used as a learned baseline, representing an evolution of the simple Monte Carlo version used with REINFORCE (which doesn’t need the learned value network). This highlights how PPO is an evolution of REINFORCE and vanilla policy-gradient in multiple forms, across the optimization form, baseline, etc. In practice, with PPO and other algorithms used for language models, this is predicting the return of each token after the deduction of KL penalties (the per-token loss includes the KL from the reward traditionally, as discussed).

There are a few different methods (or targets) used to learn the value functions. Generalized Advantage Estimation (GAE) is considered the state-of-the-art and canonical implementation in modern systems, but it carries more complexity by computing the value prediction error over multiple steps – see the later section on GAE in this chapter. A value function can also be learned with Monte Carlo estimates from the rollouts used to update the policy. PPO has two losses – one to learn the value function and another to use that value function to update the policy.

![Figure 6: Value function training uses on-policy rollouts to compute targets. The model predicts V_t at each token, which is trained via MSE against the target return \hat{V}_t. The advantage A_t = \hat{V}_t - V_t then weights the policy gradient update.](images/value_fn_training.png)

Figure 6: Value function training uses on-policy rollouts to compute targets. The model predicts \(V\_t\) at each token, which is trained via MSE against the target return \(\hat{V}\_t\). The advantage \(A\_t = \hat{V}\_t - V\_t\) then weights the policy gradient update.

A simple example implementation of a value network loss is shown below.

```
# Basic PPO critic targets & loss (no GAE)
#
# B: Batch Size
# L: Completion Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards; EOS row includes outcome
#   done_mask: (B, L) 1.0 at terminal token (EOS or truncation if penalized), else 0.0
#   completion_mask: (B, L) 1.0 on response tokens to supervise (ignore the prompt)
#   values: (B, L) current critic predictions V_theta(s_t)
#       because a value network is a running update
#   old_values: (B, L) critic predictions at rollout time V_{theta_old}(s_t)
#   gamma: discount factor, float (often 1.0 for LM RLHF)
#   epsilon_v: float value clip range (e.g., 0.2), similar to PPO Loss Update itself, optional
#
# Returns:
#   value_loss: scalar; advantages: (B, L) detached (for policy loss)

B, L = rewards.shape

# 1) Monte Carlo returns per token (reset at terminals)
# Apply discounting, if enabled
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t (post-KL)

# 2) PPO-style value clipping (optional)
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) Mask to response tokens and aggregate
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) Advantages for policy loss (no GAE): A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# The value loss is applied later, often with the PG loss, e.g.
# total_loss = policy_loss + vf_coef * value_loss
```

### Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is introduced in DeepSeekMath [[12]](#ref-shao2024deepseekmath), and used in other DeepSeek works, e.g. DeepSeek-V3 [[13]](#ref-deepseekai2025deepseekv3technicalreport) and DeepSeek-R1 [[14]](#ref-guo2025deepseek). GRPO can be viewed as a PPO-inspired algorithm with a very similar surrogate loss, but it avoids learning a value function with another copy of the original policy language model (or another checkpoint for initialization). This brings two posited benefits:

1. Avoiding the challenge of learning a value function from an LM backbone, where research hasn’t established best practices.
2. Saves memory by not needing to keep the extra set of model weights in memory (going from needing the current policy, the reference policy, and a value function, to just the first two copies).

GRPO does this by simplifying the value estimation and assigning the same value to every token in the episode (i.e. in the completion to a prompt, each token gets assigned the same value rather than discounted rewards in a standard value function) by estimating the advantage or baseline. The estimate is done by collecting multiple completions (\(a\_i\)) and rewards (\(r\_i\)), i.e. a Monte Carlo estimate, from the same initial state / prompt (\(s\)).

To state this formally, the GRPO objective is very similar to the PPO objective above. For GRPO, the objective (or loss) is accumulated over a group of completions \(\{a\_1, a\_2, ..., a\_G\}\) to a given prompt \(s\). Here, we show the GRPO objective:

\[J(\theta) = \frac{1}{G}\sum\_{i=1}^G \left(\min\left(\frac{\pi\_\theta(a\_i|s)}{\pi\_{\theta\_{\text{old}}}(a\_i|s)}A\_i, \text{clip} \left( \frac{\pi\_\theta(a\_i|s)}{\pi\_{\theta\_{\text{old}}}(a\_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A\_i \right) - \beta \mathcal{D}\_{\text{KL}}(\pi\_\theta||\pi\_{\text{ref}})\right).\qquad{(33)}\]

Note that relative to PPO, the standard implementation of GRPO includes the KL distance in the loss. As above, we can expand this into a per-token computation:

\[\begin{aligned}
J(\theta) = \frac{1}{G}\sum\_{i=1}^G \frac{1}{|a\_i|} \sum\_{t=1}^{|a\_i|} \Bigg( &\min\!\left(\frac{\pi\_\theta(a\_{i,t}|s\_{i})}{\pi\_{\theta\_{\text{old}}}(a\_{i,t}|s\_{i})}A\_{i,t},\; \text{clip} \left( \frac{\pi\_\theta(a\_{i,t}|s\_{i})}{\pi\_{\theta\_{\text{old}}}(a\_{i,t}|s\_{i})}, 1-\varepsilon, 1+\varepsilon \right) A\_{i,t} \right) \\
&- \beta \mathcal{D}\_{\text{KL}}\!\left(\pi\_\theta(\cdot|s\_{i})\|\pi\_{\text{ref}}(\cdot|s\_{i})\right) \Bigg)
\end{aligned}\qquad{(34)}\]

With the advantage computation for the completion index \(i\):

\[A\_i = \frac{r\_i - \text{mean}({r\_1, r\_2, \cdots, r\_G})}{\text{std}({r\_1, r\_2, \cdots, r\_G})}.\qquad{(35)}\]

![Figure 7: GRPO architecture. Advantages are normalized relative to the group mean and standard deviation. The KL penalty is applied directly in the loss rather than shaping the reward.](images/grpo_tikz.png)

Figure 7: GRPO architecture. Advantages are normalized relative to the group mean and standard deviation. The KL penalty is applied directly in the loss rather than shaping the reward.

Intuitively, the GRPO update is comparing multiple answers to a single question within a batch. The model learns to become more like the answers marked as correct and less like the others. This is a very simple way to compute the advantage, which is the measure of how much better a specific action is than the average at a given state. Relative to PPO, REINFORCE, and broadly RLHF performed with a reward model rating (relative to output reward), GRPO is often run with a far higher number of samples per prompt because the advantage is entirely about the relative value of a completion to its peers from that prompt. Here, the current policy generates multiple responses to a given prompt, and the group-wise GRPO advantage estimate is given valuable context. PPO and vanilla policy-gradient algorithms were designed to accurately estimate the reward of every completion (in fact, more completions can do little to improve the value estimate in some cases). GRPO and its variants are particularly well-suited to modern language model tools, where having multiple completions to a given prompt is very natural (especially when compared to, e.g., multiple actions from a set environment state in a robotic task).

The advantage computation for GRPO has trade-offs in its biases. The normalization by standard deviation rewards questions in a batch that have a low variation in answer correctness. For questions with either nearly all correct or all incorrect answers, the standard deviation will be lower and the advantage will be higher. Liu et al. 2025 [[9]](#ref-liu2025understanding) proposes removing the standard deviation term given this bias, but this comes at the cost of down-weighting questions that were all incorrect with a few correct answers, which could be seen as valuable learning signal for the model. Those high-variance prompts can be exactly the hardest cases, where only a few sampled completions find the correct answer and provide a strong training signal.

eq. [35](#eq:GRPO_ADV) is the implementation of GRPO when working with outcome supervision (either a standard reward model or a single verifiable reward) and a different implementation is needed with process supervision. In this case, GRPO computes the advantage as the sum of the normalized rewards for the following reasoning steps.

Finally, GRPO’s advantage estimation can also be applied without the PPO clipping to more vanilla versions of policy gradient (e.g. REINFORCE), but it is not the canonical form. As an example of how these algorithms are intertwined, we can show that the advantage estimation in a variant of GRPO, Dr. GRPO (GRPO Done Right) [[9]](#ref-liu2025understanding), is equivalent to the RLOO estimation (which uses the average reward of other samples as its baseline) up to a constant scaling factor (which normally does not matter due to implementation details to normalize the advantage). Dr. GRPO removes the standard deviation normalization term from eq. [35](#eq:GRPO_ADV) – note that this also scales the advantage *up*, which is equivalent to increasing the GRPO learning rate on samples with a variance in answer scores. This addresses a bias towards questions with low reward variance – i.e. almost all the answers are right or wrong – but comes at a potential cost if it is important to learn from problems where just one sample gets the answer right. The Dr. GRPO advantage for completion \(i\) within a group of size \(G\) is defined as:

\[ \tilde{A}\_i = r\_i - \text{mean}({r\_1, r\_2, \cdots, r\_G}) = r\_i - \frac{1}{G}\sum\_{j=1}^G r\_j \qquad{(36)}\]

Here, in the same notation, we can recall the RLOO advantage estimation as:

\[ A\_i^\text{RLOO} = r\_i - \frac{1}{G-1}\sum\_{j=1, i\neq j}^G r\_j \qquad{(37)}\]

Thus, if we multiply the Dr. GRPO advantage definition by \(\frac{G}{G-1}\) we can see a scaled equivalence:

\[
\begin{aligned}
\frac{G}{G-1} \tilde{A}\_i &= \frac{G}{G-1} \left( r\_i - \frac{1}{G}\sum\_{j=1}^G r\_j \right) \\
&= \frac{G}{G-1} r\_i - \frac{1}{G-1} \sum\_{j=1}^G r\_j \\
&= \frac{G}{G-1} r\_i - \frac{1}{G-1} \sum\_{j=1, j\neq i}^G r\_j - \frac{1}{G-1} r\_i \\
&= r\_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum\_{j=1, j\neq i}^G r\_j \\
&= r\_i - \frac{1}{G-1} \sum\_{j=1, j\neq i}^G r\_j \\
&= A\_i^{\text{RLOO}}
\end{aligned}
\qquad{(38)}\]

### Group Sequence Policy Optimization (GSPO)

When taking multiple gradient steps on a batch of data collected from a previous policy, importance sampling is required to correct for the distribution mismatch between the data-collection policy and the current policy being optimized. The standard importance sampling identity allows us to estimate expectations under one distribution using samples from another:

\[
\mathbb{E}\_{p}[f(x)] = \mathbb{E}\_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
\qquad{(39)}\]

where \(p\) is the target distribution, \(q\) is the sampling distribution, and \(\frac{p(x)}{q(x)}\) is the importance weight. In policy gradient methods, \(p = \pi\_\theta\) is the current policy we want to optimize and \(q = \pi\_{\theta\_{\text{old}}}\) is the policy that generated the training data. This allows us to reweight samples collected under \(\pi\_{\theta\_{\text{old}}}\) to estimate gradients for \(\pi\_\theta\), enabling multiple gradient steps per batch of rollouts.

This distribution mismatch arises in two common scenarios: (1) taking multiple gradient steps on a single batch, where \(\pi\_\theta\) drifts from \(\pi\_{\theta\_{\text{old}}}\) after each update, and (2) in asynchronous training systems where the inference backend (e.g., vLLM) and training backend (e.g., FSDP) may have different model weights due to synchronization delays (see the Asynchronicity section later in this chapter, which emerged particularly with the focus on RL for verifiable rewards, but is also used in RLHF setups).

PPO and GRPO apply importance sampling at the token level and stabilize learning by clipping the *surrogate objective*. However, this approach has a subtle failure mode: when a token’s importance ratio moves outside the clipping range \([1-\varepsilon, 1+\varepsilon]\), that token receives zero gradient. For rare but important tokens—such as key reasoning steps that the model initially assigns low probability—this “token dropping” can prevent the model from learning to produce them more reliably.

Group Sequence Policy Optimization (GSPO) [[15]](#ref-zheng2025gspo) extends GRPO by computing importance ratios at the sequence level rather than the token level. The practical motivation for this algorithm – and its peer, CISPO, which modifies how importance sampling is computed for policy gradient algorithms, as we will discuss later – is that the per-token importance sampling ratio is often numerically unstable. The conceptual motivation is that when rewards are assigned at the sequence level (as in most RLHF and RLVR setups), the importance sampling correction should match that granularity.

Token-level ratios can behave erratically for long sequences and/or large, sparse models (e.g. modern mixture-of-experts (MoE) models): a single token with a large ratio can dominate the policy update, or many tokens may get clipped independently within a response, fragmenting the learning signal across a single response. GSPO addresses this by computing a single importance weight per response.

Recall that the probability of a full response factorizes autoregressively:

\[
\pi\_\theta(a \mid s) = \prod\_{t=1}^{|a|} \pi\_\theta(a\_t \mid s, a\_{<t}).
\qquad{(40)}\]

Note that for simplicity, we often shorten the conditional policy, \(\pi\_\theta(a\_t \mid s, a\_{<t})\), as \(\pi\_\theta(a\_t \mid s)\), which implicitly contains the previous actions (tokens) in a completion. GSPO defines a length-normalized sequence-level importance ratio using the geometric mean (to avoid numerical issues with long sequences):

\[
\rho\_i(\theta) = \left( \frac{\pi\_\theta(a\_i \mid s)}{\pi\_{\theta\_{\text{old}}}(a\_i \mid s)} \right)^{\frac{1}{|a\_i|}} = \exp\left( \frac{1}{|a\_i|} \sum\_{t=1}^{|a\_i|} \log \frac{\pi\_\theta(a\_{i,t} \mid s, a\_{i,<t})}{\pi\_{\theta\_{\text{old}}}(a\_{i,t} \mid s, a\_{i,<t})} \right).
\qquad{(41)}\]

The GSPO objective mirrors GRPO but uses this sequence-level ratio:

\[
J\_{\text{GSPO}}(\theta) = \mathbb{E}\_{s \sim \mathcal{D},\, \{a\_i\}\_{i=1}^G \sim \pi\_{\theta\_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum\_{i=1}^G \min\left( \rho\_i(\theta) A\_i,\, \text{clip}(\rho\_i(\theta), 1-\varepsilon, 1+\varepsilon) A\_i \right) \right].
\qquad{(42)}\]

Because the ratio is length-normalized, the clipping range \(\varepsilon\) operates on a per-token average scale, making the effective constraint comparable across responses of different lengths. In implementation, the sequence-level weight \(\rho\_i\) is applied uniformly to all tokens in response \(a\_i\), which simplifies gradient computation while maintaining the sequence-level IS correction.

The advantage computation remains the same as GRPO (eq. [35](#eq:GRPO_ADV)), using the group-relative mean and standard deviation normalization, which can be modified as done in other derivative studies of GRPO. GSPO can be summarized as “GRPO with sequence-level importance ratios”—the IS correction granularity is matched to the reward granularity.

### Clipped Importance Sampling Policy Optimization (CISPO)

Clipped Importance Sampling Policy Optimization (CISPO) [[16]](#ref-minimax2025minimax_m1) takes a different approach: rather than clipping the surrogate objective, CISPO clips the importance weights themselves while preserving gradients for all tokens. The objective uses a stop-gradient on the clipped importance weight, returning to a REINFORCE-style formulation instead of the PPO-style, two-sided clipping:

\[
J\_{\text{CISPO}}(\theta) = \mathbb{E}\_{s \sim \mathcal{D},\, \{a\_i\}\_{i=1}^K \sim \pi\_{\theta\_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{\sum\_{i=1}^K |a\_i|} \sum\_{i=1}^K \sum\_{t=1}^{|a\_i|} \text{sg}\left( \hat{\rho}\_{i,t}(\theta) \right) A\_{i,t} \log \pi\_\theta(a\_{i,t} \mid s, a\_{i,<t}) \right],
\qquad{(43)}\]

where \(\text{sg}(\cdot)\) denotes stop-gradient (the weight is used but not differentiated through), and the clipped importance ratio is:

\[
\hat{\rho}\_{i,t}(\theta) = \text{clip}\left( \rho\_{i,t}(\theta),\, 1 - \varepsilon\_{\text{low}},\, 1 + \varepsilon\_{\text{high}} \right), \quad \rho\_{i,t}(\theta) = \frac{\pi\_\theta(a\_{i,t} \mid s, a\_{i,<t})}{\pi\_{\theta\_{\text{old}}}(a\_{i,t} \mid s, a\_{i,<t})}.
\qquad{(44)}\]

The key difference from PPO/GRPO is subtle but important: clipping the weight (not the objective) means every token still receives a gradient signal proportional to its advantage—the weight just bounds how much that signal is amplified or suppressed by the importance ratio. This is a bias-variance tradeoff: clipping weights introduces bias but controls variance and, critically, avoids dropping token gradients entirely.

Both CISPO and GSPO were developed by organizations pushing the limits of applying RL on large-scale MoE models, which are known for their numerical issues. The papers highlight how the per-token importance sampling ratios are unstable and can add substantial variance to the gradients, mitigating learning. This can make these algorithms particularly impactful on large-scale models, but less studied and beneficial within smaller, academic experiments.

CISPO also allows asymmetric clipping bounds (\(\varepsilon\_{\text{low}} \neq \varepsilon\_{\text{high}}\)), similar to DAPO’s “clip-higher” modification discussed later in this chapter, which can encourage exploration by allowing larger updates for tokens the model wants to upweight. Related work includes Tapered Off-Policy REINFORCE (TOPR) [[17]](#ref-leroux2025topr), which also clips IS weights directly (like CISPO) rather than clipping within the objective (like PPO/GRPO), but operates at the sequence level (like GSPO) and uses asymmetric clipping based on reward sign—applying no IS correction for positive rewards while clipping ratios to \([0, 1]\) for negative rewards—enabling stable off-policy learning.

### Comparing Algorithms

Each algorithm in this chapter shares the same core gradient shape (eq. [1](#eq:policy_gradient_intuition)), but differs in how it estimates the advantage and controls the optimization:

* **REINFORCE**: The simplest policy gradient implementation, using Monte-Carlo estimates of reward and a state-based baseline to reduce variance.
* **RLOO**: REINFORCE with multiple samples per prompt, with each sample’s baseline being the average reward of the others (leave-one-out) to reduce gradient variance.
* **PPO**: Adds a learned value function and a clipped policy ratio to get more accurate and stable gradient updates.
* **GRPO**: A simplified variant of PPO that groups multiple completions per prompt and normalizes rewards within the group to compute advantages, removing the need for a value function.
* **CISPO**: A REINFORCE-style algorithm that clips importance-sampling weights (not the objective as in PPO/GRPO) with a stop-gradient for stability, so every token receives a gradient signal.
* **GSPO**: Like GRPO but normalizes the policy ratio by completion length, preventing length bias.
* **DPO**: Not an RL algorithm, but a method to solve the same preference optimization problem by bypassing the separate reward model entirely, optimizing directly from preference pairs (see Chapter 8).

All of the policy gradient algorithms above are on-policy in derivation, though most are applied slightly off-policy in practice. DPO and the other direct alignment algorithms in Chapter 8 are off-policy by default. All can be paired with a learned reward model or verifiable rewards. Only PPO requires a learned value function. REINFORCE and RLOO have no importance-sampling ratio — the remaining algorithms each introduce one to enable multiple gradient steps per batch of rollouts, differing in granularity and clipping strategy as summarized below.

Table 1: Comparing policy gradient algorithms.

| Method | IS Granularity | Clipping Style | Advantage |
| --- | --- | --- | --- |
| **REINFORCE** | None | None | Monte Carlo baseline |
| **RLOO** | None | None | Leave-one-out |
| **PPO** | Token | Objective (bilateral) | Learned value fn |
| **GRPO** | Token | Objective (bilateral) | Group-relative |
| **GSPO** | Sequence | Objective (bilateral) | Group-relative |
| **CISPO** | Token | Weights (stop-grad) | Group-relative |

The core loss \(\mathcal{L}(\theta)\) for each method is:

\[\begin{aligned}
\textbf{REINFORCE:}\quad & -\frac{1}{T}\sum\_{t=1}^{T}\log \pi\_\theta(a\_t\mid s\_t)\,\big(G\_t - b(s\_t)\big) \\[6pt]
\textbf{RLOO:}\quad & -\frac{1}{K}\sum\_{i=1}^{K}\sum\_t \log \pi\_\theta(a\_{i,t}\mid s\_{i,t})\left(R\_i-\frac{1}{K-1}\sum\_{j\neq i}R\_j\right) \\[6pt]
\textbf{CISPO:}\quad & -\sum\_{i,t} \mathrm{sg}(\hat{\rho}\_{i,t})\, A\_{i,t} \log \pi\_\theta(a\_{i,t}\mid s\_{i,t}) \\
& \quad \hat{\rho}\_{i,t} = \mathrm{clip}(\rho\_{i,t},\, 1-\varepsilon,\, 1+\varepsilon) \\[6pt]
\textbf{PPO:}\quad & -\frac{1}{T}\sum\_{t=1}^{T}\min\!\big(\rho\_t A\_t,\ \mathrm{clip}(\rho\_t,1-\varepsilon,1+\varepsilon)\, A\_t\big) \\
& \quad \rho\_t = \frac{\pi\_\theta(a\_t\mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t\mid s\_t)} \\[6pt]
\textbf{GRPO:}\quad & -\frac{1}{G}\sum\_{i=1}^{G}\min\!\big(\rho\_i A\_i,\ \mathrm{clip}(\rho\_i,1-\varepsilon,1+\varepsilon)\, A\_i\big) \\
& \quad \rho\_i = \frac{\pi\_\theta(a\_i\mid s)}{\pi\_{\theta\_{\text{old}}}(a\_i\mid s)},\quad A\_i = \frac{r\_i-\mathrm{mean}(r\_{1:G})}{\mathrm{std}(r\_{1:G})} \\[6pt]
\textbf{GSPO:}\quad & -\frac{1}{G}\sum\_{i=1}^{G}\min\!\big(\rho\_i A\_i,\ \mathrm{clip}(\rho\_i,1-\varepsilon,1+\varepsilon)\, A\_i\big) \\
& \quad \rho\_i = \left(\frac{\pi\_\theta(a\_i\mid s)}{\pi\_{\theta\_{\text{old}}}(a\_i\mid s)}\right)^{1/|a\_i|} \\[6pt]
\textbf{DPO:}\quad & -\mathbb{E}\_{(x,y^{w},y^{l})}\!\left[\log \sigma\!\big(\beta[\Delta\log \pi\_\theta(x)-\Delta\log \pi\_{\mathrm{ref}}(x)]\big)\right]
\end{aligned}\]

## Implementation

Compared to the original Deep RL literature where many of these algorithms were developed, implementing RL for optimizing language models or other large AI models requires many small implementation details. In this section, we highlight some key factors that differentiate the implementations of popular algorithms.

There are many other small details that go into this training. For example, when doing RLHF with language models a crucial step is generating text that will then be rated by the reward model. Under normal circumstances, the model should generate an end-of-sequence (EOS) token indicating it finished generating, but a common practice is to put a hard cap on generation length to efficiently utilize infrastructure. A failure mode of RLHF is that the model is regularly truncated in its answers, driving the ratings from the reward model out-of-distribution and to unpredictable scores. The solution to this is to *only* run reward model scoring on the `eos_token`, and to otherwise assign a penalty to the model for generating too long.

The popular open-source tools for RLHF have a large variance in implementation details across the algorithms (see table 10 in [[18]](#ref-ivison2024unpacking)). Some decisions not covered here include:

* **Value network initialization**: The internal learned value network used by PPO and other similar algorithms can be started from a different model of the same architecture or randomly selected weights. This can have a large impact on performance. The standard established in InstructGPT [[19]](#ref-ouyang2022training) (and re-used in Tülu 3 for its work on RLVR [[20]](#ref-lambert2024t)) is to initialize the value network from the reward model used during RLHF. Others have used the previous checkpoint to RLHF training (normally an SFT model) with a value head appended randomly initialized, or fully re-initialized language models (less common as it will take longer for RLHF to converge, but possible).
* **Reward normalization, reward whitening, and/or advantage whitening**: Normalization bounds all the values from the RM (or environment) to be between 0 and 1, which can help with learning stability. [Whitening](https://en.wikipedia.org/wiki/Whitening_transformation) goes further by transforming rewards or advantage estimates to have zero mean and unit variance, providing an even stronger boost to stability.
* **Different KL estimators**: With complex language models, precisely computing the KL divergence between models can be complex, so multiple approximations are used to substitute for an exact calculation [[21]](#ref-schulman2016klapprox).
* **KL controllers**: Original implementations of PPO and related algorithms had dynamic controllers that targeted specific KLs and changed the penalty based on recent measurements. Most modern RLHF implementations use static KL penalties, but this can also vary.

For more details on implementation details for RLHF, see [[22]](#ref-huang2024n). For further information on the algorithms, see [[23]](#ref-weng2018PG).

### Policy-Gradient Basics

A simple implementation of policy gradient, using advantages to estimate the gradient to prepare for advanced algorithms such as PPO and GRPO follows:

```
pg_loss = -advantages * ratio
```

Ratio here is the (per-token) probability ratio (often computed from a log-probability difference) of the new policy model probabilities relative to the old policy that generated the batch.

In order to understand this equation, it is good to understand different cases that can fall within a batch of updates. Remember that we want the loss to *decrease* as the model gets better at the task.

Case 1: Positive advantage, so the action was better than the expected value of the state. We want to reinforce this. In this case, the model will make this more likely with the negative sign. To do so, it’ll increase the logratio. A positive logratio, or sum of log probabilities of the tokens, means that the model is more likely to generate those tokens.

Case 2: Negative advantage, so the action was worse than the expected value of the state. This follows very similarly. Here, the loss will be positive if the new model was more likely, so the model will try to make it so the policy parameters make this completion less likely.

Case 3: Zero advantage, so no update is needed. The loss is zero, don’t change the policy model.

### Loss Aggregation Tradeoffs

The question when implementing any policy gradient algorithm with language models is: How do you aggregate per-token losses into a final scalar loss? Given per-token losses \(\ell\_{i,t}\) for sample \(i\) at token \(t\), with completion lengths \(|a\_i|\) and batch size \(B\), there are three main strategies:

**Strategy 1: Per-sequence normalization** (standard GRPO; also used in some PPO implementations)

\[L = \frac{1}{B} \sum\_{i=1}^{B} \frac{1}{|a\_i|} \sum\_{t=1}^{|a\_i|} \ell\_{i,t}\qquad{(45)}\]

Each sequence contributes equally to the batch loss, regardless of length. In code:

```
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**Strategy 2: Per-token normalization** (DAPO [[24]](#ref-yu2025dapo))

\[L = \frac{\sum\_{i=1}^{B} \sum\_{t=1}^{|a\_i|} \ell\_{i,t}}{\sum\_{i=1}^{B} |a\_i|}\qquad{(46)}\]

Each token contributes equally; longer sequences have proportionally more influence on the gradient. In code:

```
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**Strategy 3: Fixed-length normalization** (Dr. GRPO [[9]](#ref-liu2025understanding))

\[L = \frac{1}{B} \sum\_{i=1}^{B} \frac{1}{L\_{\max}} \sum\_{t=1}^{|a\_i|} \ell\_{i,t}\qquad{(47)}\]

Normalizes by max sequence length \(L\_{\max}\), equalizing the per-token scale across sequences while still letting longer sequences contribute more total gradient because they contain more active tokens. In code:

```
# Strategy 3: Fixed-length normalization
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

Where \(L\_{\max}\) is typically a global constant during the entire training procedure, which specifies the maximum number of generation tokens.

Note that `completion_mask` in the code above is a matrix of 1s and 0s, where the prompt tokens are masked out (0s) because we don’t want the model to learn from predicting prompt tokens.

#### Why Does This Matter?

Intuitively, per-sequence normalization (Strategy 1) seems best since we care about *outcomes*, not individual tokens. However, this introduces subtle biases based on sequence length, which can cause the model to overthink or down-weight strategies that naturally need to use more tokens, depending on the direction of the bias. Consider two sequences of different lengths with per-token losses:

```
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

With **Strategy 1** (per-sequence): The batch loss is \((2.8 + 1.9)/2 = 2.35\), and crucially, each token in the short sequence receives a larger gradient than tokens in the long sequence.

With **Strategy 2** (per-token): The batch loss is \((14 + 19)/15 = 2.2\), and all tokens receive equal gradient magnitude.

With **Strategy 3** (fixed-length with \(L\_{\max}=10\)): The short sequence contributes \(1.4\) and the long sequence contributes \(1.9\), balancing per-token gradients while still weighting by sequence.

For a more complete example showing how these strategies affect gradients, see the script below.

```
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)

advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

The output shows that with Strategy 1 (`masked_mean`), the short sequence has larger per-token gradients (0.25) than the long sequence (0.14). Strategies 2 and 3 equalize the per-token gradients across sequences. Note that these results can vary substantially if gradient accumulation is used, where the gradients are summed across multiple minibatches before taking a backward step—in this case, the balance between shorter and longer sequences can flip.

In practice, the best strategy depends on the specific training setup. Often in RLHF the method with the best numerical stability or the least variance in loss is preferred.

#### Related: MDP vs. Bandit Framing

The choice of loss aggregation connects to a deeper distinction in how we frame the RL problem. The **MDP (token-level)** view treats each token \(a\_t\) as an action with state \(s\_t\) being the running prefix. In practice, this is the framing used when we compute token-level advantages with a learned value function \(V(s\_t)\) (e.g., GAE [[3]](#ref-schulman2015high)) and apply KL penalties per token. PPO with a learned value network is the canonical example [[7]](#ref-schulman2017proximal).

In contrast, the **bandit (sequence-level)** view treats the whole completion as a single action with one scalar reward \(R\). In code, this means computing a sequence-level advantage \(A\_{\text{seq}}\) and broadcasting it to all tokens. RLOO and GRPO-style advantages are often used in this bandit-style setting [[6]](#ref-kool2019buy) [[1]](#ref-ahmadian2024back) [[12]](#ref-shao2024deepseekmath). Direct alignment methods like DPO and A-LoL also define sequence-level objectives, although they are not policy-gradient estimators [[25]](#ref-baheti2023leftover).

Note that many GRPO implementations use a bandit-style advantage *and* add a separate per-token KL term in the loss, while many PPO/RLOO implementations fold KL into the reward before computing advantages; both conventions exist in practice.

An example comparison highlighting the two approaches is below:

```
# === Bandit-style (sequence-level) ===
# One scalar reward per sequence; advantage broadcast to all tokens
reward = torch.tensor([3.0, 1.0])       # (B,) e.g., reward model scores
baseline = reward.mean()                 # simple baseline (RLOO uses leave-one-out)
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- same advantage for all tokens
#         [-1., -1., -1., -1.]])

# === MDP-style (token-level) ===
# Per-token rewards + learned V(s_t); each token gets its own advantage
# (could also use per-token KL shaping, format rewards, or other token-level signals)
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- varies by position
#         [-0.3, -0.5, -0.8, -1.4]])
```

This framing distinction also explains why the discount factor \(\gamma\) is set to 1.0 in virtually all RLHF implementations. In standard RL, discounting (\(\gamma < 1\)) is essential: it balances the optimization between short-term and long-term reward across a multi-step episode, which is crucial for the agent to learn effective behavior over time. But in the RLHF setting, even when using the token-level MDP view, the inductive bias of the optimization is the quality of the collective completion – the reward signal scores the entire response, not individual tokens. Discounting earlier tokens would arbitrarily down-weight their contribution with no principled justification. As agentic RL settings mature – where models take real multi-step actions such as tool calls, code execution, and web browsing – discounting may become relevant again, since these involve genuinely distinct sequential decisions whose long-term consequences differ.

### Asynchronous RL Systems

The default implementation for policy-gradient algorithms is what is called **on-policy** execution, where the actions (generations) taken by the agent (language model) are scored before updating the model. The theoretical derivations of policy-gradient rely on all actions being exactly on-policy where the model is always up to date with the results from the latest trials/roll-outs. In practice, maintaining exact on-policy execution substantially slows training [[26]](#ref-noukhovitch2024asynchronous)—and perfect synchronization is technically impossible regardless. Therefore, all of the recent empirical results with language models tend to be slightly outside of the theoretical proofs. What happens in practice is designing the algorithms and systems for what actually works.

![Figure 8: A comparison of the generation-update phases for synchronous or asynchronous RL training following Noukhovitch et al. 2024.](images/async_v_synch_rl.png)

Figure 8: A comparison of the generation-update phases for synchronous or asynchronous RL training following Noukhovitch et al. 2024.

The common solution used is to constantly run inference and training on separate GPU nodes with software designed to efficiently run both, as shown in the bottom of fig. [8](#fig:async). Common practice in popular open-source RL tools for language models is to use a distributed process management library such as Ray to hand information off between the policy-gradient learning loop and the inference loop using an efficient inference engine, e.g., vLLM. In these setups, the GPUs dedicated to taking the RL steps are called the “learners” and the GPUs dedicated to sampling from the language model are called the “actors”. The primary challenges faced when making training more asynchronous are keeping training stable and maintaining learning signal.

![Figure 9: An example distributed RL system, where two queues are managed to pass data to the learner and actor GPUs, which can both be synchronized with a distributed computing library such as Ray. Olmo Team 2025, license CC-BY.](images/distributed-rl.png)

Figure 9: An example distributed RL system, where two queues are managed to pass data to the learner and actor GPUs, which can both be synchronized with a distributed computing library such as Ray. Olmo Team 2025, license CC-BY.

These systems are designed and implemented with the presumption that nearly on-policy data is good enough for stable learning. Here, the generation and update phases can easily be synced to avoid idle compute on either piece of the training system, which would be passing model weights from the learners to the actors in fig. [9](#fig:async_system). With reasoning models, the extremely long inference characteristics of problems requiring 10K to 100K+ tokens per answer makes the generation of roll-outs a far stronger bottleneck. A common problem when training reasoning models on more synchronous RL infrastructure is that an answer to one prompt in the batch can take substantially more time to generate (either through more tokens or more tool calls), resulting in the majority of the allocated compute being idle until it completes. A second solution to this length mismatch issue, called sequence-level packing, is to stack shorter samples within a batch with clever masking to enable continued roll-outs from the model and better distribute length normalization across samples within a batch. The full complexity of distributed RL infrastructure is out of scope for this book, as it can cause many other subtle issues that slow down training or cause instability.

Following the emergence of these reasoning models, further interest has been taken to make the training and inference loops fully off-policy, where training batches for the policy gradient updates are filled with the most recently completed roll-outs across multiple instances generating answers [[27]](#ref-wu2025llamarl) [[28]](#ref-fu2025areal). Fully asynchronous training would also enable scaling RL training runs across multiple datacenters more easily due to the option of increasing the time between weight syncs between the learner node (taking policy gradient steps) and the actor (trying to solve problems) [[29]](#ref-primeintellectteam2025intellect2reasoningmodeltrained).

Related methods are exploring fully off-policy policy gradient algorithms [[17]](#ref-leroux2025topr).

### Truncated Importance Sampling

Truncated importance sampling (TIS) is a crucial tool used to stabilize training in modern, asynchronous RL frameworks with language models. Importance sampling is a correction that reweights samples drawn from one distribution to estimate expectations under another (as introduced in eq. [39](#eq:IS_identity)). Truncated importance sampling [[30]](#ref-ionides2008truncated) caps these weights with \(\min(\rho, C)\) for some constant \(C\), trading a small bias for bounded variance in the policy gradient.

This is an importance-sampling correction applied to the policy gradient, but unlike the bilateral clipping in PPO and CISPO (which constrains the ratio near 1), TIS uses a one-sided upper cap: the ratio can fall freely below 1, but is capped at \(C\) to prevent extreme upweighting. In all of PPO, GRPO, CISPO (and related algorithms), the ratio \(\rho\_t^{\text{policy}} = \pi\_\theta(a\_t \mid s) / \pi\_{\theta\_{\text{old}}}(a\_t \mid s)\) corrects for policy drift across multiple gradient steps within one RL batch. As we shift to real-world RL frameworks, centered around the idea of asynchronicity in the previous subsection, there can be even larger sources of numerical differences (that also require the numerical correction of importance sampling). Even when the sampler and learner share identical parameters \(\theta\), their effective token distributions can differ because the inference engine (e.g., vLLM) and training framework (e.g., FSDP) use different kernels, precision, and parallelism strategies [[31]](#ref-yao2025offpolicy). It is therefore useful to distinguish the same policy evaluated on two systems, \(\pi\_\theta^{\text{sampler}}\) and \(\pi\_\theta^{\text{learner}}\), and define the corresponding ratio and its truncated form:

\[
\rho\_t^{\text{learner}} = \frac{\pi\_\theta^{\text{learner}}(a\_t \mid s, a\_{<t})}{\pi\_\theta^{\text{sampler}}(a\_t \mid s, a\_{<t})}, \qquad \tilde{\rho}\_t^{\text{learner}} = \min(\rho\_t^{\text{learner}},\; C).
\qquad{(48)}\]

These two corrections are complementary, but they appear in policy-gradient implementations for different reasons — one compensates for policy drift within the training of an RL batch, the other for implementation-induced divergence — and can be applied simultaneously. How they combine depends on the algorithm:

#### REINFORCE with TIS (Single Gradient Step)

There is no policy drift (\(\pi\_\theta = \pi\_{\theta\_\text{old}}\)), so the only mismatch is between the learner and sampler. Here \(\pi\_{\theta\_\text{old}} = \pi\_\text{gen}\), and TIS directly corrects the learner–sampler gap:

\[
\nabla\_\theta J \approx \mathbb{E}\_{a \sim \pi\_\theta^{\text{sampler}}} \left[ \tilde{\rho}\_t^{\text{learner}} \cdot A\_t \cdot \nabla\_\theta \log \pi\_\theta^{\text{learner}}(a\_t \mid s, a\_{<t}) \right].
\qquad{(49)}\]

#### PPO/GRPO with TIS (Multiple Gradient Steps)

Now both ratios are active. In careful implementations, the “old logprobs” in the policy ratio are recomputed on the learner (the GSPO paper discusses this), so the policy ratio \(\rho\_t^{\text{policy}} = \pi\_\theta^{\text{learner}} / \pi\_{\theta\_\text{old}}^{\text{learner}}\) captures pure policy drift, while \(\tilde{\rho}\_t^{\text{learner}} = \min(\pi\_{\theta\_\text{old}}^{\text{learner}} / \pi\_{\theta\_\text{old}}^{\text{sampler}},\; C)\) separately corrects the backend mismatch at the generation checkpoint:

\[
J\_{\text{PPO+TIS}}(\theta) = \mathbb{E}\left[ \min\!\left( \rho\_t^{\text{policy}}\, A\_t,\; \text{clip}\!\left(\rho\_t^{\text{policy}}, 1-\varepsilon, 1+\varepsilon\right) A\_t \right) \cdot \tilde{\rho}\_t^{\text{learner}} \right].
\qquad{(50)}\]

Here \(\pi\_{\theta\_\text{old}} \neq \pi\_\text{gen}\): the old logprobs come from the learner, not the sampler. If a framework skips this recomputation and uses the sampler logprobs directly as \(\pi\_{\theta\_\text{old}}\), the policy ratio already captures the backend mismatch and no separate TIS correction is needed — but the clip then operates on a noisier ratio that starts away from 1.0 even before any gradient steps. This is the “your framework secretly brings you off-policy RL” observation from Yao et al. [[31]](#ref-yao2025offpolicy).

In practice, LLM RL systems apply TIS as a per-token correction weight on the policy-gradient loss:

```
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

The \([-10, 10]\) clamp is only for numerical stability before exponentiation; the actual truncated-importance-sampling step is the one-sided cap at \(C\). In practice, the bookkeeping around these logprobs — storing sampler logprobs from generation, recomputing learner logprobs at the old checkpoint, and tracking current logprobs during gradient steps — is a substantial part of the scaffolding in distributed RL frameworks. Unlike GSPO, this correction is token-level because it addresses token-level numerical mismatch rather than sequence-level reward granularity. TIS for the learner–sampler ratio has been adopted across major open-source RL frameworks (VeRL, TRL, OpenRLHF, SkyRL, OAT, and Open Instruct, which uses \(C = 2\)), and becomes increasingly important for long reasoning traces (Chapter 7), where small per-token differences compound over thousands of generated tokens.

### Example: PPO

There are many, many implementations of PPO available. The core *loss* computation is shown below. Crucial to stable performance is also the *value* computation, where multiple options exist (including multiple options for the *value model* loss).

Note that the reference policy (or old logprobs) here are from the time the generations were sampled and not necessarily the reference policy. The reference policy is only used for the KL distance constraint/penalty.

```
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute returns via backward pass (gamma typically 1.0 for LM RLHF)
# Mask rewards to avoid padding tokens (which may have KL penalties) leaking into returns
returns = torch.zeros_like(rewards)
running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(rewards.shape[1])):
    # Zero out padding: only accumulate rewards/returns for valid completion tokens
    running = (rewards[:, t] + self.gamma * running) * completion_mask[:, t]
    returns[:, t] = running

# Compute advantages: A_t = G_t - V(s_t)
advantages = returns - values.detach()  # Shape: (B*G, L)
# Note: We detach the value network here to not update the parameters of
# the value function when computing the policy-gradient loss

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Value function loss: predict returns
vf_loss = 0.5 * ((returns - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()

    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

The core piece to understand with PPO is how the policy gradient loss is updated. Focus on these three lines:

```
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```

`pg_losses1` is the vanilla advantage-weighted policy gradient loss. `pg_losses2` applies the same formula but with the probability ratio clamped to the range \([1-\varepsilon, 1+\varepsilon]\), limiting how much the policy can change in a single update.

The key insight is taking `torch.max` of the two losses. Because we’re minimizing a *negative* loss (recall the negative sign in front of advantages), taking the maximum selects the more pessimistic gradient—the one that produces a smaller policy update. When the advantage is positive (good action), clipping prevents the policy from increasing that action’s probability too aggressively. When the advantage is negative (bad action), clipping prevents over-correction in the other direction.

By clamping the log-probability ratio, PPO bounds how far the policy can drift from the version that generated the training data, stabilizing learning without requiring an explicit trust region computation.

The code above also shows PPO learning a value function alongside the policy, which adds implementation complexity, but the clipped objective is the core mechanism.

#### PPO/GRPO Simplification with One Gradient Step per Sample (No Clipping)

PPO (and GRPO) implementations can be handled much more elegantly if the hyperparameter “number of gradient steps per sample” is equal to 1. Many typical values for this are from 2-4 or higher. In the main PPO or GRPO equations, see eq. [28](#eq:PPO_EQN), the “reference” policy is the previous parameters – those used to generate the completions or actions. Thus, if only one gradient step is taken, \(\pi\_\theta = \pi\_{\theta\_{\text{old}}}\), and the update rule reduces to the following (the notation \([]\_\nabla\) indicates a stop gradient):

\[J(\theta) = \frac{1}{G}\sum\_{i=1}^G \left(\frac{\pi\_\theta(a\_i|s)}{\left[\pi\_{\theta}(a\_i|s)\right]\_\nabla}A\_i - \beta \mathcal{D}\_{\text{KL}}(\pi\_\theta||\pi\_{\text{ref}})\right). \qquad{(51)}\]

This leads to PPO or GRPO implementations where the second policy gradient and clipping logic can be omitted, making the optimizer far closer to standard policy gradient.

### Example: GRPO

The DeepSeekMath paper describes some implementation details of GRPO that differ from PPO [[12]](#ref-shao2024deepseekmath), especially if comparing to a standard application of PPO from Deep RL rather than language models. For example, the KL penalty within the RLHF optimization (recall the KL penalty is also used when training reasoning models on verifiable rewards without a reward model) is applied directly in the loss update rather than to the reward function. Where the standard KL penalty application for RLHF is applied as \(r=r\_\theta - \beta \mathcal{D}\_{\text{KL}}\), the GRPO implementation is along the lines of:

\[ L = L\_{\text{policy gradient}} + \beta \* \mathcal{D}\_{\text{KL}} \qquad{(52)}\]

However, there are multiple ways to implement this. Traditionally, the KL distance is computed with respect to each token in the completion to a prompt \(s\). For reasoning training, multiple completions are sampled from one prompt, and there are multiple prompts in one batch, so the KL distance will have a shape of [B, L, N], where B is the batch size, L is the sequence length, and N is the number of completions per prompt.

Putting it together, using the first loss accumulation, the pseudocode can be written as below.

```
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute group-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
```

For more details on how to interpret this code, see the PPO section above. The core differences from the PPO example are:

* **Advantage computation**: GRPO normalizes rewards relative to the group (mean and std across generations for the same prompt) rather than using a learned value function as baseline.
* **No value network**: GRPO removes the value model entirely, eliminating `vf_loss` and the associated complexity.
* **KL penalty placement**: GRPO adds the KL penalty directly to the loss rather than subtracting it from the reward (this is the standard implementation, but more versions exist on how the KL is applied).

#### RLOO vs. GRPO

The advantage updates for RLOO follow GRPO very closely, highlighting the conceptual similarity of the algorithm when taken separately from the PPO style clipping and KL penalty details. Specifically, for RLOO, the advantage is computed relative to a baseline that is extremely similar to that of GRPO – the completion reward relative to the others for that same question. Concisely, the RLOO advantage estimate follows as (expanded from [TRL](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433)’s implementation):

```
# rloo_k --> number of completions per prompt
# rlhf_reward --> Initially a flat tensor of total rewards for all completions. Length B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) #
# Now, Shape: (k, N), each column j contains the k rewards for prompt j.

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> Leave-one-out baseline rewards. Shape: (k, N)
#  baseline[i, j] is the avg reward of samples i' != i for prompt j.

advantages = rlhf_reward - baseline
# advantages --> Same Shape: (k, N)

advantages = advantages.flatten() # Same shape as original tensor
```

The rest of the implementation details for RLOO follow the other trade-offs of implementing policy-gradient.

## Auxiliary Topics

In order to master the application of policy-gradient algorithms, there are countless other considerations. Here we consider some of the long-tail of complexities in successfully deploying a policy-gradient RL algorithm.

### Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation (GAE) is an alternate method to compute the advantage for policy gradient algorithms [[3]](#ref-schulman2015high) that better balances the bias-variance tradeoff. Traditional single-step advantage estimates can introduce too much bias, while using complete trajectories can suffer from high variance. GAE computes an exponentially-weighted average of multi-step advantage estimates, where the \(\lambda\) hyperparameter controls the bias-variance tradeoff—ranging from single-step TD (\(\lambda=0\)) to full trajectory returns (\(\lambda=1\)); \(\lambda=0.95\) is a common default for LLM fine-tuning.

Advantage estimates can take many forms, but we can define an \(n\)-step advantage estimator (similar to the TD residual at the beginning of the chapter) as follows:

\[
\hat{A}\_t^{(n)} = \begin{cases}
r\_t + \gamma V(s\_{t+1}) - V(s\_t), & n = 1 \\
r\_t + \gamma r\_{t+1} + \gamma^2 V(s\_{t+2}) - V(s\_t), & n = 2 \\
\vdots \\
r\_t + \gamma r\_{t+1} + \gamma^2 r\_{t+2} + \cdots - V(s\_t), & n = \infty
\end{cases}
\qquad{(53)}\]

Here a shorter \(n\) will have lower variance but higher bias as we are attributing more learning power to each trajectory – it can overfit. GAE attempts to generalize this formulation into a weighted multi-step average instead of a specific \(n\). To start, we must define the temporal difference (TD) residual of predicted value.

\[
\delta\_t^V = r\_t + \gamma V(s\_{t+1}) - V(s\_t)
\qquad{(54)}\]

To utilize this, we introduce another variable \(\lambda\) as the GAE mixing parameter. This folds into an exponential decay of future advantages we wish to estimate:

\[
\begin{array}{l}
\hat{A}\_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}\_t^{(1)} + \lambda\hat{A}\_t^{(2)} + \lambda^2\hat{A}\_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta\_t^V + \lambda(\delta\_t^V + \gamma\delta\_{t+1}^V) + \lambda^2(\delta\_t^V + \gamma\delta\_{t+1}^V + \gamma^2\delta\_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta\_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta\_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)\left(\delta\_t^V\frac{1}{1-\lambda} + \gamma\delta\_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots\right) \\
= \sum\_{l=0}^{\infty}(\gamma\lambda)^l\delta\_{t+l}^V
\end{array}
\qquad{(55)}\]

Intuitively, this can be used to average multi-step estimates of Advantage in an elegant fashion. An example implementation is shown below:

```
# GAE (token-level) for LM RLHF
#
# B: Batch Size
# L: Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards
#   values:  (B, L) current V_theta(s_t)
#   done_mask: (B, L) 1.0 at terminal token (EOS or penalized trunc), else 0.0
#   gamma: float (often 1.0),
#   lam (short for lambda): float in [0,1]
#   (Padding beyond terminal should have rewards=0, values=0)
B, L = rewards.shape
advantages = torch.zeros_like(rewards)
next_v = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards[:, t] + gamma * not_done * next_v - values[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values[:, t]

targets = advantages + values      # y_t for value regression
advantages = advantages.detach()   # for policy loss
```

The backward loop accumulates temporal-difference (TD) errors (\(\delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t)\)), which measure how much better or worse the actual outcome was compared to the value function’s prediction, with exponential decay \((\gamma\lambda)^l\). At terminal tokens, `not_done=0` prevents bootstrapping from future states and resets the GAE accumulator, so each episode’s advantages are computed independently (since the loop runs backward, the terminal token cleanly stops the exponentially-weighted accumulation at episode boundaries—this makes the implementation packing-friendly, correctly handling multiple sequences concatenated into one). The final `targets` serve as regression targets for the separate value function learned outside this GAE loop, while the detached `advantages` weight the policy gradient—detached so that policy updates don’t backpropagate through the value network. In RLHF for language models, \(\gamma=1.0\) is common because episodes are short token sequences where undiscounted credit assignment is preferred (and often all of the tokens in one).

*For further reading, see [[32]](#ref-seita2017gae).*

### Double Regularization

We’ve seen in this chapter two types of regularization. One is built into algorithms like PPO with step-size constraints, and the other is a KL divergence based distance penalty relative to the start of the optimization.

Many popular policy gradient algorithms from Deep Reinforcement Learning, including PPO and its predecessors, originated due to the need to control the learning process of the agent. In RLHF, as discussed extensively in Chapter 15 on Regularization and in Chapter 3 on Training Overview, there is a built-in regularization term via the distance penalty relative to the original policy one is fine-tuning. In this view, a large part of the difference between algorithms like PPO (which have internal step-size regularization) and REINFORCE (which is simpler, and to which PPO reduces under certain hyperparameters) is far less meaningful for fine-tuning language models than training agents from scratch.

In PPO, the objective that handles capping the step-size of the update is known as the [surrogate objective](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective). To monitor how much the PPO regularization is impacting updates in RLHF, one can look at the clip fraction variable in many popular implementations, which is the percentage of samples in the batch whose probability ratio falls outside the clipping interval. This is a useful proxy for how often PPO’s regularizer may be active, but not every such sample has zero gradient: the surrogate becomes flat only when the clipped branch is selected, such as positive-advantage samples with ratios above \(1+\varepsilon\) or negative-advantage samples with ratios below \(1-\varepsilon\).

In practice with language models, algorithms like PPO and GRPO are often run with only one gradient step per batch, which means that the PPO-native regularization is never applied (as clipping can only occur within a batch when the policy changes substantially) and the KL distance penalties predominate. However, this is not universal. For example, DAPO uses 16 gradient steps per batch [[24]](#ref-yu2025dapo), and Tülu 3 uses 4 PPO update iterations per batch for 8B and 70B models but reduces to 1 for 405B to maintain training stability [[20]](#ref-lambert2024t).

### Further Reading

As RLHF has cemented itself at the center of modern post-training, other policy-gradient RL algorithms and RL algorithms generally have been proposed to improve the training process, but they have not had a central role in governing best practices. Examples for further reading include:

* **Pairwise Proximal Policy Optimization (P3O; Wu et al., 2023)** [[33]](#ref-wu2023pairwise) uses pairwise data directly in a PPO-style policy update without learning an intermediate reward model.
* **Soft Adaptive Policy Optimization (SAPO)** [[34]](#ref-gao2025sapo) replaces hard PPO/GRPO-style clipping with smooth, temperature-controlled gating, aiming for a continuous trust region that preserves near-on-policy learning signal while down-weighting off-policy tokens.
* Off-policy policy-gradient algorithms could enable further asynchronous training, such as **Contrastive Policy Gradient (CoPG)** [[35]](#ref-flet2024contrastive) (a generalization of the direct alignment algorithm IPO and vanilla policy gradient), which was used by Cohere for their Command A model [[36]](#ref-cohere2025command).
* Other implementations of REINFORCE algorithms have been designed for language models, such as **ReMax** [[37]](#ref-li2023remax), which implements a baseline normalization designed specifically to accommodate the sources of uncertainty from reward model inference.
* Some foundation models, such as Apple Intelligence Foundation Models [[38]](#ref-gunter2024apple) or Kimi k1.5 reasoning model [[39]](#ref-team2025kimi), have used variants of **Mirror Descent Policy Optimization (MDPO)** [[40]](#ref-tomar2020mirror). Research is still developing further on the fundamentals here [[41]](#ref-zhang2025improving), but Mirror Descent is an optimization method rather than directly a policy gradient algorithm. What is important here is that it is substituted in very similarly to existing RL infrastructure.
* **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** proposes 4 modifications to GRPO to better suit reasoning language models, where long traces are needed and new, underutilized tokens need to be increased in probability [[24]](#ref-yu2025dapo). The changes are: 1, have two different clip hyperparameters, \(\varepsilon\_\text{low}\) and \(\varepsilon\_\text{high}\), so clipping on the positive side of the logratio can take bigger steps for better exploration; 2, dynamic sampling, which removes all samples with reward = 0 or reward = 1 for all samples in the batch (no learning signal); 3, use the per-token loss as discussed above in Implementation: GRPO; and 4, a soft penalty on samples that are too long to avoid trying to learn from truncated answers.
* **Value-based Augmented Proximal Policy Optimization (VAPO)** [[42]](#ref-yuan2025vapo) combines optimizations from DAPO (including clip-higher, token-level policy-gradient, and different length normalization) with insights from Value-Calibrated PPO [[43]](#ref-yuan2025s) to pretrain the value function and length-adaptive GAE to show the promise of value-based methods relative to GRPO.

## Suggested Experiments

The companion implementation in `code/policy_gradients/` is designed for small, observable RL runs. The default configs train `Qwen/Qwen3-1.7B` on the `spell_backward` procedural task from `reasoning-gym`, which is a good first exercise because failures and partial progress are easy to inspect.

1. **Run the word reversal task with GRPO.**

   ```
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   Track `avg_correctness`, `avg_format`, and `avg_binary`. The useful first question is whether each prompt group contains contrast: if all sampled completions are right or all are wrong, a group-relative update has little learning signal.
2. **Compare group-relative and single-sample estimators.** Run the matched starting configs:

   ```
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   Compare how quickly the correctness signal improves and how noisy the loss is. RLOO and GRPO should make the role of within-prompt baselines much more concrete than the equations alone.
3. **Sweep the contrast knobs.** Copy `policy_gradients/configs/grpo.yaml` and vary `num_rollouts`, `temperature`, `data.size`, and `format_weight`. Small `num_rollouts` reduces group contrast; very low temperature can collapse samples; very high temperature can generate too many malformed answers. This is the simplest way to see why RLVR recipes often spend so much effort on sampling settings before touching the optimizer.
4. **Move from toy rewards toward math.** For GSM8K-style experiments, start with the `code/reward_models/train_orm.py` and `code/rejection_sampling/` examples before adding a new online RL environment. A good contribution would be a small `reasoning-gym` or GSM8K policy-gradient config that runs on a sub-1B Qwen model and reports the same group-contrast diagnostics.

# Bibliography

[1]

A. Ahmadian *et al.*, “Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms,” in *Annual meeting of the association for computational linguistics (ACL)*, 2024.

[2]

Z. Wang *et al.*, “HelpSteer2-preference: Complementing ratings with preferences,” in *International conference on learning representations (ICLR)*, 2025.

[3]

J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” in *Proceedings of the international conference on learning representations (ICLR)*, 2016.

[4]

R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” *Machine learning*, vol. 8, pp. 229–256, 1992.

[5]

S. C. Huang, A. Ahmadian, and C. F. AI, “Putting RL back in RLHF.” <https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo>, 2024.

[6]

W. Kool, H. van Hoof, and M. Welling, “Buy 4 reinforce samples, get a baseline for free!” 2019.

[7]

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” *arXiv preprint arXiv:1707.06347*, 2017.

[8]

C. Berner *et al.*, “Dota 2 with large scale deep reinforcement learning,” *arXiv preprint arXiv:1912.06680*, 2019.

[9]

Z. Liu *et al.*, “Understanding R1-zero-like training: A critical perspective,” *arXiv preprint arXiv:2503.20783*, Mar. 2025, Available: <https://arxiv.org/abs/2503.20783>

[10]

J. Nocedal and S. J. Wright, *Numerical optimization*. Springer, 2006.

[11]

J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in *International conference on machine learning*, PMLR, 2015, pp. 1889–1897.

[12]

Z. Shao *et al.*, “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” *arXiv preprint arXiv:2402.03300*, 2024.

[13]

DeepSeek-AI *et al.*, “DeepSeek-V3 technical report.” 2025. Available: <https://arxiv.org/abs/2412.19437>

[14]

D. Guo *et al.*, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” *arXiv preprint arXiv:2501.12948*, 2025.

[15]

C. Zheng *et al.*, “Group sequence policy optimization.” 2025. doi: [10.48550/arXiv.2507.18071](https://doi.org/10.48550/arXiv.2507.18071).

[16]

MiniMax, “MiniMax-M1: Scaling test-time compute efficiently with lightning attention.” 2025. doi: [10.48550/arXiv.2506.13585](https://doi.org/10.48550/arXiv.2506.13585).

[17]

N. Le Roux *et al.*, “Tapered off-policy REINFORCE: Stable and efficient reinforcement learning for LLMs.” 2025. doi: [10.48550/arXiv.2503.14286](https://doi.org/10.48550/arXiv.2503.14286).

[18]

H. Ivison *et al.*, “Unpacking DPO and PPO: Disentangling best practices for learning from preference feedback,” in *Advances in neural information processing systems (NeurIPS)*, 2024.

[19]

L. Ouyang *et al.*, “Training language models to follow instructions with human feedback,” *Advances in neural information processing systems*, vol. 35, pp. 27730–27744, 2022.

[20]

N. Lambert *et al.*, “Tulu 3: Pushing frontiers in open language model post-training,” *arXiv preprint arXiv:2411.15124*, 2024.

[21]

J. Schulman, “Approximating KL-divergence.” <http://joschu.net/blog/kl-approx.html>, 2016.

[22]

S. Huang, M. Noukhovitch, A. Hosseini, K. Rasul, W. Wang, and L. Tunstall, “The n+ implementation details of RLHF with PPO: A case study on TL;DR summarization,” in *First conference on language modeling*, 2024. Available: <https://openreview.net/forum?id=kHO2ZTa8e3>

[23]

L. Weng, “Policy gradient algorithms,” *lilianweng.github.io*, 2018, Available: <https://lilianweng.github.io/posts/2018-04-08-policy-gradient/>

[24]

Q. Yu *et al.*, “DAPO: An open-source LLM reinforcement learning system at scale.” 2025.

[25]

A. Baheti, X. Lu, F. Brahman, R. L. Bras, M. Sap, and M. Riedl, “Leftover lunch: Advantage-based offline reinforcement learning for language models,” in *International conference on learning representations (ICLR)*, 2024.

[26]

M. Noukhovitch, S. Huang, S. Xhonneux, A. Hosseini, R. Agarwal, and A. Courville, “Asynchronous RLHF: Faster and more efficient off-policy RL for language models,” in *International conference on learning representations (ICLR)*, 2025.

[27]

B. Wu *et al.*, “LlamaRL: A distributed asynchronous reinforcement learning framework for efficient large-scale LLM trainin,” *arXiv preprint arXiv:2505.24034*, 2025.

[28]

W. Fu *et al.*, “AReaL: A large-scale asynchronous reinforcement learning system for language reasoning,” *arXiv preprint arXiv:2505.24298*, 2025.

[29]

P. I. Team *et al.*, “INTELLECT-2: A reasoning model trained through globally decentralized reinforcement learning.” 2025. Available: <https://arxiv.org/abs/2505.07291>

[30]

E. L. Ionides, “Truncated importance sampling,” *Journal of Computational and Graphical Statistics*, vol. 17, no. 2, pp. 295–311, 2008.

[31]

F. Yao, L. Liu, D. Zhang, C. Dong, J. Shang, and J. Gao, “Your efficient RL framework secretly brings you off-policy RL training.” 2025. Available: <https://fengyao.notion.site/off-policy-rl>

[32]

D. Seita, “Notes on the generalized advantage estimation paper.” 2017. Available: <https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/>

[33]

T. Wu, B. Zhu, R. Zhang, Z. Wen, K. Ramchandran, and J. Jiao, “Pairwise proximal policy optimization: Harnessing relative feedback for llm alignment,” *arXiv preprint arXiv:2310.00212*, 2023.

[34]

C. Gao *et al.*, “Soft adaptive policy optimization,” *arXiv preprint arXiv:2511.20347*, Nov. 2025, Available: <https://arxiv.org/abs/2511.20347>

[35]

Y. Flet-Berliac *et al.*, “Contrastive policy gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion,” in *Conference on empirical methods in natural language processing (EMNLP)*, 2024.

[36]

T. Cohere *et al.*, “Command a: An enterprise-ready large language model,” *arXiv preprint arXiv:2504.00698*, 2025.

[37]

Z. Li *et al.*, “Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models,” in *Forty-first international conference on machine learning*, 2024.

[38]

T. Gunter *et al.*, “Apple intelligence foundation language models,” *arXiv preprint arXiv:2407.21075*, 2024.

[39]

K. Team *et al.*, “Kimi k1. 5: Scaling reinforcement learning with llms,” *arXiv preprint arXiv:2501.12599*, 2025.

[40]

M. Tomar, L. Shani, Y. Efroni, and M. Ghavamzadeh, “Mirror descent policy optimization,” in *International conference on learning representations (ICLR)*, 2022.

[41]

Y. Zhang *et al.*, “Improving LLM general preference alignment via optimistic online mirror descent,” *arXiv preprint arXiv:2502.16852*, 2025.

[42]

Y. Yuan *et al.*, “VAPO: Efficient and reliable reinforcement learning for advanced reasoning tasks,” *arXiv preprint arXiv:2504.05118*, 2025.

[43]

Y. Yuan, Y. Yue, R. Zhu, T. Fan, and L. Yan, “What’s behind PPO’s collapse in long-CoT? Value optimization holds the secret,” *arXiv preprint arXiv:2503.01491*, 2025.

[← Previous: Reward Modeling](05-reward-models)
[Next: Reasoning and Inference-Time Scaling →](07-reasoning)

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

[![GitHub](/assets/github.svg)](https://github.com/natolambert/rlhf-book)
[![arXiv](/assets/arxiv.svg)](https://arxiv.org/abs/2504.12501)
[![Manning](/assets/manning.svg)](https://www.manning.com/books/reinforcement-learning-from-human-feedback)
[![Amazon](/assets/amazon.svg)](https://amzn.to/4cwCDJQ)
[![Discord](/assets/discord.svg)](https://discord.gg/yz5AwK4gBR)

© 2024-2026 Nathan Lambert · [rlhfbook.com](https://rlhfbook.com)