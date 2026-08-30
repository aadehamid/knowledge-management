[A. Weers](/)

[Home](/)
[Blog](/blog/)
[Publications](/publications/)

☼
☾

#### Contents

* [State of RL for reasoning LLMs](#state-of-rl-for-reasoning-llms)
  + [Brief RL Introduction](#brief-rl-introduction)
  + [REINFORCE](#reinforce)
  + [PPO](#ppo)
  + [GRPO](#grpo)
  + [RLOO](#rloo)
  + [Dr. GRPO](#dr-grpo)
  + [DAPO](#dapo)
  + [CISPO](#cispo)
  + [MaxRL](#maxrl)
  + [DPPO](#dppo)
  + [ScaleRL](#scalerl)
  + [Summary](#summary)
  + [Open Problems](#open-problems)

# State of RL for reasoning LLMs

March 15, 2026 · 26 min read
· [Markdown version](/blog/2026/rl-for-llms/index.md)

# State of RL for reasoning LLMs[¶](#state-of-rl-for-reasoning-llms "Permanent link")

Reinforcement learning has been one of the most consequential additions to the LLM post-training stack.
It was the key ingredient that transformed GPT-3 into InstructGPT [[1]](#ref-ouyang2022training), and it has since become central to the current wave of reasoning improvements [[2]](#ref-jaech2024openai)[[3]](#ref-deepseekai2025deepseekr1incentivizingreasoningcapability).

The first generation of RL for LLMs was dominated by PPO [[4]](#ref-schulman2017proximal), a method developed for more conventional RL settings such as Atari games and robotics, but adapted very successfully to RLHF.

The second generation, driven by the goal of improving reasoning capabilities, brought another round of algorithmic refinement.
A large number of variants have appeared in a short period of time, with most differing from their predecessors in only small but consequential ways.

This post provides a compact overview of the major developments in **RL for reasoning LLMs** (2024-2026). It starts from the foundations (REINFORCE and PPO), then covers GRPO and subsequent methods that refined and improved upon it.

## Brief RL Introduction[¶](#brief-rl-introduction "Permanent link")

In the standard RL setting, an **agent** observes a **state** $s\_t$, chooses an **action** $a\_t$ according to a **policy** $\pi(a\_t \mid s\_t)$, transitions to a new state $s\_{t+1}$ according to the environment dynamics $p(s\_{t+1} \mid s\_t, a\_t)$, and receives a **reward** $r\_t$.

A concrete example is a robot navigating a room: the state is its current position and sensor reading, the actions are movement commands, the transition dynamics are governed by physics (wheels might slip), and the reward reflects progress toward the goal.

This loop repeats over $T$ **time steps**. The agent aims to maximize the expected discounted return

$$
J = \mathbb{E}\left[\sum\_{t=0}^{T} \gamma^t r\_t\right]
$$

where the **discount factor** $0 \leq \gamma \leq 1$ controls how strongly future rewards are discounted.

The policy is usually parametrized by $\theta$.
A central object in many RL algorithms is the **value function**

$$
V^\pi(s) = \mathbb{E}\_\pi\left[\sum\_{l=0}^{T-t} \gamma^l r\_{t+l} \mid s\_t = s\right],
$$

which measures how good it is to be in state $s$ under policy $\pi$. From this, one can derive **advantages**, which estimate whether a particular action was better or worse than expected.

For LLMs, the setup is often simplified substantially.
We have a parametrized **model** $\pi\_\theta$ sampling **responses** $y \sim \pi\_\theta(\cdot|x)$ given a **prompt** $x \sim \mathcal{D}$ from our dataset, which we grade with a scalar **reward** $r(x, y)$.
The objective becomes:

$$
J(\theta) = \mathbb{E}\_{x \sim \mathcal{D},\, y \sim \pi\_\theta(\cdot|x)}\left[r(x, y)\right]
$$

One can still model this environment with states being *(prompt + previously generated tokens)* and actions being the next token.
In practice, however, it is usually not possible to assign meaningful rewards to individual tokens, but only provide one reward for the complete response given the prompt.
The reward would be zero for all tokens except the last, making the setup unnecessarily complicated.

## REINFORCE[¶](#reinforce "Permanent link")

We start with REINFORCE [[5]](#ref-williams1992simple), since it is both conceptually simple and the foundation of all policy gradient methods.

![Image](/assets/img/grpo/all_reinforce.jpg)

In its simplest form, the REINFORCE objective is:

$$
J(\theta) = \mathbb{E}\_{y \sim \pi\_\theta(\cdot \mid x)}\left[r(x, y)\right]
$$

The gradient of this objective has a simple and interpretable form:

$$
\nabla\_\theta J(\theta) = \mathbb{E}\_{y \sim \pi\_\theta(\cdot \mid x)}\left[\nabla\_\theta \log \pi\_\theta(y \mid x) \cdot r(x, y)\right]
$$

For comparison, the gradient of supervised fine-tuning is

$$
\nabla\_\theta L\_{\text{SFT}}(\theta) = -\nabla\_\theta \log \pi\_\theta(y^\* \mid x)
$$

(Note, that the SFT loss is minimized, while the RL objective is maximized).

This comparison reveals that REINFORCE is essentially a *weighted* form of SFT.
Instead of reinforcing provided, off-policy answers $y^\*$, we reinforce or punish sampled, on-policy answers $y$ weighted according to their rewards.

The main disadvantage of REINFORCE is variance. Even when the reward is relatively structured (e.g., a large test suite where each test contributes partial reward), gradient estimates can vary substantially across samples.

To reduce variance, REINFORCE subtracts a **baseline** $b(x)$ that does not depend on the sampled action (response).
This leaves the expected gradient unchanged since

$$
\mathbb{E}\_{y \sim \pi\_\theta(\cdot \mid x)}\left[\nabla\_\theta \log \pi\_\theta(y \mid x)\, b(x)\right] = 0,
$$

while often reducing variance substantially. The gradient then becomes

$$
\nabla\_\theta J(\theta) = \mathbb{E}\_{y \sim \pi\_\theta(\cdot \mid x)}\left[\nabla\_\theta \log \pi\_\theta(y \mid x)\, \bigl(r(x,y)-b(x)\bigr)\right].
$$

The quantity $r(x,y)-b(x)$ is the simplest form of an advantage estimate.

## PPO[¶](#ppo "Permanent link")

PPO (Proximal Policy Optimization) [[4]](#ref-schulman2017proximal) became the dominant general-purpose policy gradient algorithm and, for several years, the default choice for RLHF.

The PPO objective is often presented in a form that appears complex:

$$
J^{\text{PPO}}(\theta) = \mathbb{E}\_t\left[\min\left(\rho\_t(\theta)\hat{A}\_t,\; \operatorname{clip}(\rho\_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\_t\right)\right],
$$

where

$$
\rho\_t(\theta) = \frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t \mid s\_t)}
$$

is the **importance sampling** (IS) ratio between the current policy $\pi\_\theta$ and the policy that generated the rollout $\pi\_{\theta\_\text{old}}$.

One might ask: "Shouldn't this always be 1 for on-policy RL?" The answer is yes, but only for the *first* optimizer step after generating rollouts.

This ratio is needed because rollouts are expensive.
In practice, one usually reuses a batch of generated data for multiple minibatch updates or multiple epochs.
After the first optimizer step, the training policy is no longer exactly the same as the generation policy, so PPO becomes slightly off-policy.
The ratio corrects for this mismatch, and the clipping limits how far optimization can move away from the generation policy.
This is PPO's approximation to a trust region[[6]](#ref-schulman2015trust).

Note that clipping affects not only the objective's value but, more importantly, its dependence on $\theta$.
Since we optimize $\theta$ to maximize $J$, clipped cases produce zero gradients because the learned policy is no longer part of the equation.
Updates for these cases are skipped since we have moved outside the trust region.

The clipping handles four scenarios:

|  | Not clipped | Clipped |
| --- | --- | --- |
| **Positive advantage** (good answer, reinforce) | $\rho\_t(\theta)\hat{A}\_t$: the answer is good and has not been over-updated | $(1+\epsilon)\hat{A}\_t$: the answer is already sufficiently more likely, so the gradient is stopped |
| **Negative advantage** (bad answer, discourage) | $\rho\_t(\theta)\hat{A}\_t$: the answer is bad and has not been over-updated | $(1-\epsilon)\hat{A}\_t$: the answer is already sufficiently less likely, so the gradient is stopped |

We can also express this clipping as a mask:

$$
M(\hat{A}\_t, \rho\_t, \epsilon) = \begin{cases}
0 & \text{if } (\hat{A}\_t > 0 \land \rho\_t > 1 + \epsilon) \lor (\hat{A}\_t < 0 \land \rho\_t < 1 - \epsilon) \\
1 & \text{otherwise}
\end{cases}
$$

With this formulation, the objective simplifies to:

$$
J^{\text{PPO}}(\theta) = \mathbb{E}\_t\left[M(\hat{A}\_t,\rho\_t(\theta),\epsilon)\,\rho\_t(\theta)\,\hat{A}\_t\right].
$$

So, PPO is essentially an importance-weighted policy gradient with a trust region mask.

For advantage estimation, PPO uses the **Generalized Advantage Estimator** (GAE):

$$
\hat{A}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}
$$

Computing $\delta$ requires a learned value function.
In the LLM setting, this usually requires an additional value model, often of similar size to the policy model.
That is costly in memory and increases the training complexity.
We will not examine GAE in detail in this post, since removing this component is GRPO's main practical contribution.
See this [detailed post](https://huggingface.co/blog/NormalUhr/rlhf-pipeline) for an in-depth explanation of PPO and all its components.

Finally, the PPO objective is often combined with KL regularization:

$$
J^{\text{PPO-KL}}(\theta) = \mathbb{E}\_t\left[M(\hat{A}\_t,\rho\_t,\epsilon)\rho\_t(\theta)\hat{A}\_t\right] - \beta\, D\_{\text{KL}}(\pi\_\theta \,\|\, \pi\_{\text{ref}}).
$$

Here $\pi\_{\text{ref}}$ is usually the model before RL training.
In RLHF this term is especially important, since it preserves general capabilities and helps to control the distribution shift relative to the reward model (that was trained on the reference policy $\pi\_{\text{ref}}$).
In reasoning RL, the KL penalty is often set much smaller or omitted entirely[[3]](#ref-deepseekai2025deepseekr1incentivizingreasoningcapability)[[7]](#ref-yu2025dapo).

In its full form, PPO requires four large components in memory: the trainable policy, the rollout policy, the reference policy, and the value model.

## GRPO[¶](#grpo "Permanent link")

GRPO (Group Relative Policy Optimization), introduced in DeepSeekMath and later popularized by DeepSeek-R1 [[8]](#ref-shao2024deepseekmath)[[3]](#ref-deepseekai2025deepseekr1incentivizingreasoningcapability), removes PPO's value model and replaces it with a group-relative baseline.

The key insight is that we can get a good baseline for each response by comparing it to other responses for the same prompt. For each prompt $x \sim \mathcal{D}$, GRPO samples a group of $G$ responses $\{y\_1, \ldots, y\_G\}$, computes rewards $r\_i = r(x, y\_i)$, and **normalizes rewards within the group** to obtain advantages:

$$
\hat{A}\_i = \frac{r\_i - \mu\_G}{\sigma\_G}, \qquad
\mu\_G = \frac{1}{G}\sum\_{j=1}^{G} r\_j, \qquad
\sigma\_G = \sqrt{\frac{1}{G}\sum\_{j=1}^{G}(r\_j-\mu\_G)^2}.
$$

Intuitively, the baseline for a rollout is no longer a learned value function, but the performance of the other rollouts for the same prompt. This works particularly well when rewards are sparse but multiple samples per prompt are available.

The GRPO objective keeps PPO-style clipped importance sampling and in its original formulation includes a KL term:

$$
J^{\text{GRPO}}(\theta) = \mathbb{E}\_{x \sim \mathcal{D}}\left[\frac{1}{G}\sum\_{i=1}^{G} \min\left(\rho\_i(\theta) \hat{A}\_i,\, \text{clip}(\rho\_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}\_i\right) - \beta \cdot D\_{\text{KL}}(\pi\_\theta(\cdot|x) \| \pi\_{\text{ref}}(\cdot|x))\right]
$$

where $\rho\_i(\theta)=\frac{\pi\_\theta(y\_i \mid x)}{\pi\_{\theta\_{\text{old}}}(y\_i \mid x)}$.

Group normalization has two useful effects.
Subtracting the mean makes the learning signal prompt-relative: a reward of $0.8$ should be interpreted differently if all samples for that prompt lie in $[0.8, 1.0]$ than if they lie in $[0.2, 0.8]$.
Dividing by the standard deviation makes reward scale less sensitive, which is useful when combining tasks with different reward ranges.

The more important reason for GRPO's success, however, is simpler: it **removed the critic**.
That cuts memory use significantly and made large-scale RL for reasoning models much easier to run.

## RLOO[¶](#rloo "Permanent link")

RLOO (REINFORCE Leave-One-Out) [[9]](#ref-ahmadian2024back) arrived at a similar conclusion from a different direction: PPO might be more complex than required for the LLM fine-tuning setting [[1]](#fn-1)[1] LLMs are already well-trained when RL is applied, unlike traditional RL agents that typically start from random initialization.
Though the action space (vocabulary) is much larger, the probability mass is concentrated on a small number of plausible tokens. [↩](#fnref-1).

For each prompt, RLOO samples $K$ responses $\{y\_1, \ldots, y\_K\}$. The advantage for response $y\_i$ is its reward minus the mean reward of the other $K-1$ responses:

$$
\hat{A}\_i = r\_i - \frac{1}{K-1}\sum\_{j \neq i} r\_j
$$

This baseline is unbiased and requires no learned value model.
Unlike GRPO, RLOO does not divide by the group's standard deviation.

More importantly, RLOO drops PPO-style clipping and returns to a pure REINFORCE-style update.

The RLOO objective is:

$$
J^{\text{RLOO}}(\theta) = \mathbb{E}\_{x \sim \mathcal{D}}\left[\frac{1}{K}\sum\_{i=1}^{K} \nabla\_\theta \log \pi\_\theta(y\_i|x) \cdot \hat{A}\_i\right]
$$

The authors argue that such clipping is active in less than 5% of cases in their experiments and may not be necessary in this setting.
As we will see, subsequent work reaches different conclusions.

## Dr. GRPO[¶](#dr-grpo "Permanent link")

DeepSeek reported in their DeepSeek-Math and R1 papers that response length increases substantially as RL training progresses. They attributed this to improving reasoning and reflection capabilities (the well-known "Aha" moment).
While this may be one driver, the authors of Dr. GRPO (short for "GRPO Done Right") [[10]](#ref-liu2025understanding) identify another, more significant cause: the standard sample-level loss normalization introduces a bias favoring short correct responses and long incorrect ones.

In the common GRPO implementation, token losses are first averaged within each sequence and then across sequences.
That means a fixed sequence-level reward is spread over all tokens in the sequence.
Long responses therefore receive weaker per-token reinforcement if they are correct, and weaker per-token penalty if they are incorrect.
This can create an incentive to be overly verbose.

The fix is straightforward: instead of dividing first by sequence length and then by batch size, Dr. GRPO **divides by a fixed constant** (maximum tokens).
This effectively removes the incentive for incorrect answers to be unnecessarily long.

Dr. GRPO also removes another normalization that introduces an unwanted bias.
When rewards per prompt are normalized by their standard deviation, prompts where all answers have similar rewards (e.g., all but one are correct, low variance in rewards), even small reward differences can turn into large normalized advantages.
As a result, prompts on which the model is already mostly correct can receive disproportionately large updates.

The Dr. GRPO advantage simplifies to:

$$
\hat{A}\_i = r\_i - \mu\_G
$$

**without the division by standard deviation** and the loss is aggregated at the token level with a fixed normalization, rather than first averaging by sequence length.

The practical message is not that GRPO was fundamentally broken, but that some of its seemingly innocuous normalizations were not neutral.
In long-form reasoning, they change which prompts and which tokens receive gradient signal.

## DAPO[¶](#dapo "Permanent link")

DAPO (Decoupled Advantage Policy Optimization) [[7]](#ref-yu2025dapo) is another in-depth analysis of multiple components of GRPO and proposes four improvements:

First, DAPO replaces sample-level averaging with an aggregation on **token-level** (similar to Dr. GRPO, but DAPO divides by number of actual tokens, while Dr. GRPO uses a constant).

The second improvement targets the clipping mechanism:
PPO's symmetric ratio clipping is particularly (overly) restrictive for low probability tokens.
E.g., if a token has probability $0.01$, then with $\epsilon = 0.2$ its probability can only rise to $0.012$ before being clipped, barely changing its likelihood of being sampled.
This can suppress learning of rare but useful reasoning continuations.
DAPO therefore decouples the clip bounds and uses a larger upper bound $\epsilon\_{\text{high}} = 0.28$, keeping $\epsilon\_{\text{low}} = 0.2$ (**asymmetric clipping**).

With token-level aggregation and asymmetric clipping, the DAPO objective becomes:

$$
J^{\text{DAPO}}(\theta) =
\mathbb{E}\left[
\frac{1}{\sum\_{i=1}^{G}|y\_i|}
\sum\_{i=1}^{G}\sum\_{t=1}^{|y\_i|}
\min\left(
\rho\_{i,t}(\theta)\hat{A}\_i,\;
\operatorname{clip}(\rho\_{i,t}(\theta),1-\epsilon\_{\text{low}},1+\epsilon\_{\text{high}})\hat{A}\_i
\right)
\right].
$$

The other two improvements do not modify the objective equation, but improve the step efficiency.

The third change is **overlong reward shaping**.
In many setups, truncated responses receive the same reward as completely wrong responses.
That is noisy: a response may contain mostly correct reasoning and still be cut off by the length limit.
DAPO adds a soft penalty zone before the hard cutoff:

$$
R\_\text{length}(y) =
\begin{cases}
0, & |y| \le L\_\text{max} - L\_\text{cache} \\
\frac{(L\_\text{max} - L\_\text{cache}) - |y|}{L\_\text{cache}}, & L\_\text{max} - L\_\text{cache} < |y| \le L\_\text{max} \\
-1, & L\_\text{max} < |y|.
\end{cases}
$$

This creates a more straightforward learning signal, because slightly overlong responses are penalized only mildly, while excessive responses receive stronger negative feedback.
The model can therefore learn that response length is the problem, instead of conflating truncation with complete task failure.

The fourth change is **dynamic sampling**.
If all sampled responses for a prompt are correct, or all are incorrect, then group-relative advantages are all zero and the prompt contributes no gradient.
In such cases, DAPO keeps sampling until each prompt has mixed outcomes, ensuring that every prompt in the optimization batch provides a learning signal.
This improves step efficiency, although it can increase wall-clock time because hard batches may require more generation.

## CISPO[¶](#cispo "Permanent link")

CISPO (Clipped Importance Sampling Policy Optimization), introduced in the MiniMax-M1 report [[11]](#ref-chen2025minimax), targets a specific weakness of PPO-style clipping: when a token falls outside the clip range, PPO blocks its gradient entirely.

This behaviour is conservative, but can also be overcautious.
Tokens that undergo large probability shifts are often precisely the ones that matter most for learning reasoning behavior (the reports mentions for example "However", "Recheck", "Wait", and "Aha" to have low probability in the base model, but can serve as forks in reasoning traces).
If those tokens are masked whenever the ratio becomes too large, the learning is slowed down by discarding some informative gradients.

CISPO therefore decouples clipping from gradient flow.
Instead of clipping the objective in a way that induces a hard mask, it clips only the importance-sampling **weight** and applies a stop-gradient operation to that weight:

$$
J^{\text{CISPO}}(\theta) =
\mathbb{E}\left[
\operatorname{sg}\!\left(\hat{\rho}\_t(\theta)\right)\,
\hat{A}\_t\,
\log \pi\_\theta(a\_t \mid s\_t)
\right],
\qquad
\hat{\rho}\_t(\theta)=\operatorname{clip}\bigl(\rho\_t(\theta), 1-\epsilon\_{l}, 1+\epsilon\_{h}\bigr),
$$

where $\operatorname{sg}(\cdot)$ denotes stop-gradient.

Interestingly they report that only the upper clipping $\epsilon\_h$ is required and tuned, the lower one $\epsilon\_l$ is set to a high enough value to be effectively not active.

This formulation preserves the variance-reduction benefits of IS weight clipping while allowing **gradients to flow for all tokens**.
The result is more stable training that does not suppress learning on high-information tokens, and in the MiniMax experiments a 2x speed-up in step-efficiency compared to DAPO.

CISPO can be viewed as a soft alternative to PPO-style masking: keep the trust region intuition, but clip the weight rather than deleting the complete update.

## MaxRL[¶](#maxrl "Permanent link")

MaxRL (Maximum Likelihood Reinforcement Learning) [[12]](#ref-tajwar2026maxrl) starts from a different perspective:
standard RL objectives optimize for expected reward (pass@1), which is often observed (pass@1 improves at the cost of pass@$k$), yet not necessarily the most suitable objective.
In contrast, maximum-likelihood training (as used during pre-training and SFT) would maximize $\log p\_\theta(x)$ instead.

This matters, since they show that

$$
\log p\_\theta(x) = -\sum\_{k=1}^{\infty}\frac{(1-p\_\theta(x))^k}{k},
$$

so the maximum-likelihood gradient is an infinite harmonic mixture of pass@$k$ gradients, not just pass@1.
And standard RL keeps just the first-order term of that expansion.

Therefore, MaxRL defines a compute-indexed family of truncated objectives:

$$
J\_{\text{MaxRL}}^{(T)}(x) = -\sum\_{k=1}^{T}\frac{(1-p\_\theta(x))^k}{k},
$$

where $T=1$ recovers standard RL and $T\to \infty$ recovers maximum likelihood.

The on-policy estimator whose expected gradients match this objective is remarkably simple:
Given $N$ rollouts for a prompt, let $K$ be the number of successful rollouts.
Then MaxRL averages the score functions of the successful trajectories only:

$$
\hat{g}\_N(x) =
\begin{cases}
\displaystyle \frac{1}{K}\sum\_{i=1}^{N} r\_i \nabla\_\theta \log \pi\_\theta(y\_i \mid x), & K \ge 1, \\[0.8em]
0, & K = 0.
\end{cases}
$$

This estimator is unbiased for a truncated MaxRL objective with $T=N$.
The key difference from REINFORCE is that in this case an increase in rollouts reduces the estimator variance, and simultaneously also makes the **optimized objective itself a better approximation** to maximum likelihood.

One can also rewrite the estimator in a REINFORCE-like form with a zero-mean control variate, which makes the weighting more explicit.
If $\hat r = K/N$ is the success rate for that prompt, then the effective advantage becomes proportional to

$$
\hat{A}\_i^{\text{MaxRL}} \propto \frac{r\_i - \hat r}{\hat r}.
$$

This shows why MaxRL concentrates learning signal on hard prompts.
When $\hat r$ is small but non-zero, successful rollouts on that prompt are weighted strongly.
By contrast, easy prompts with $\hat r \approx 1$ receive relatively little extra emphasis.

Empirically, MaxRL improves pass@$k$, preserves output diversity better than GRPO, and yields substantial gains in test-time scaling efficiency.

Conceptually, it is also interesting because it reframes RL for verifiable tasks as approximate maximum-likelihood training under non-differentiable sampling.

## DPPO[¶](#dppo "Permanent link")

DPPO (Divergence PPO) [[13]](#ref-qi2026rethinking) revisits the trust region question more directly than DAPO or CISPO.

The core critique is that PPO clips based on the probability ratio of the sampled token.
This might be a poor proxy for actual policy divergence, especially for rare tokens.
Their probability could change by an order of magnitude and still only have a very small effect on the full-distribution.

This problem is further amplified by training / inference framework mismatches: even with identical parameters, the probability ratio can be highly volatile for low-probability tokens between different frameworks, while divergence measures such as total variation are much more stable.

DPPO therefore replaces ratio-based masking with a **trust region defined in terms of estimated policy divergence** (TV or KL).
The computation of exact full divergence over the vocabulary is expensive, but a binary approximation (comparing just the sampled token's probability under both policies) or top-K approximation both work well empirically.

The DPPO update becomes

$$
J^{\text{DPPO}}(\theta) =
\mathbb{E}\left[
M\_{\text{div}}\!\left(\widehat{D}(\pi\_\theta,\pi\_{\theta\_{\text{old}}}), \tau\right)\,
\rho(\theta)\,
\hat{A}
\right],
$$

where $M\_{\text{div}}$ masks updates whose estimated divergence exceeds a threshold $\tau$.

An interesting insight from their experiments: just a small fraction (less than 0.5%) of updates are the cause of instability, when negative samples push the policy too far.
Blocking those is enough to stabilize training (in their experiments).

DPPO therefore raises (and proposes one answer for) the question on how we define trust regions in the LLM regime.

## ScaleRL[¶](#scalerl "Permanent link")

ScaleRL [[14]](#ref-khatri2025art) is less about inventing a new objective than about determining which design choices continue to matter once compute is scaled seriously.
The paper reports more than 400,000 GPU-hours of ablations and, more importantly, evaluates methods by fitting sigmoidal performance vs compute curves rather than comparing a single training checkpoint.

That framing is useful because it separates two quantities that are often conflated: how quickly a method improves at a given compute budget, and where it eventually saturates.
A method can look strong at low compute and still plateau early.
Another can rise more slowly but reach a better asymptote.

Their main findings are:

**Asynchronous RL.** ScaleRL prefers a pipelined asynchronous setup over the common generate-then-update loop.
In this setup rollouts are generated continuously and weight updates are pushed immediately.
This mainly improves compute efficiency by reducing idle time, while keeping the final performance competitive or better.

**Loss type.** Among the off-policy loss functions they compare, CISPO and GSPO outperform DAPO in asymptotic performance, with CISPO selected as the default because it combines strong results with relative robustness.

**FP32 logits.** Small numerical mismatches between generation kernels and training kernels can materially distort importance-sampling ratios.
As proposed in the MiniMax report[[11]](#ref-chen2025minimax), computing the LM head in FP32 sharply reduces this issue and substantially improves asymptotic performance in their ablations.

**Loss aggregation.** For loss aggregation they show the same bias as outlined by Dr. GRPO[[10]](#ref-liu2025understanding) and DAPO [[7]](#ref-yu2025dapo) that a sample averaging is suboptimal.
Instead they see the best performance for prompt-level averaging.

**Zero variance filtering.** If all answers for a prompt are correct or all are incorrect, there is no learning signal. Instead of sampling more (as in DAPO[@dapo
], which may be optimal for number of steps), they exclude those prompts from optimization, accelerating training.

**No positive resampling.** If a prompt results in more than 90% correct answers, it is excluded from future epochs. This slightly slows training but reaches higher asymptotic performance.

ScaleRL is valuable both for its large-scale empirical validation and for clarifying the shape of improvement curves, including both early learning speed and asymptotic performance.

## Summary[¶](#summary "Permanent link")

The table below summarizes the main differences between methods:

| Method | Baseline/Advantage | Clipping | Masking | Loss aggregation | Improvements |
| --- | --- | --- | --- | --- | --- |
| **REINFORCE** | EMA or batch mean reward | None | None | Sample average | establishes policy gradients |
| **PPO** | GAE with critic | symmetric IS | $M\_\text{sym}(\hat{A}\_t, \rho\_t, 0.2)$ | Sample average | stable, more sample efficient |
| **GRPO** | $(r-\mu\_G)/\sigma\_G$ | symmetric IS | $M\_\text{sym}(\hat{A}\_t, \rho\_t, 0.2)$ | Length normalized$^\dagger$ | less memory-intense |
| **RLOO** | Leave-one-out mean | None | None | Sample average | variance reduction without critic |
| **Dr. GRPO** | $r - \mu\_G$ | symmetric IS | $M\_\text{sym}(\hat{A}\_t, \rho\_t, 0.2)$ | Token average$^\ddagger$ | remove length-bias and std weighting |
| **DAPO** | $(r-\mu\_G)/\sigma\_G$ | asymmetric IS | $M\_\text{asym}(\hat{A}\_t, \rho\_t, 0.2, 0.28)$ | Token average | give small probabilities more room to increase |
| **CISPO** | $(r-\mu\_G)/\sigma\_G$ within group | upper-bound IS | None | Token average | don't mask gradients, just clip |
| **DPPO** | $(r-\mu)/\sigma$ within group | symmetric DV | $M\_\text{div}(\hat{A}\_t, \upsilon\_t, 0.15)$ | Sample average | use DV trust regions to adapt to LLM domain |
| **MaxRL** | $(r\_i - \hat{r})/(N\cdot \hat{r})$ with $\hat{r}=K/N$ | None | None | Sample average | interpolates between RL and MLE, better pass@k |
| **ScaleRL** | $(r-\mu\_B)/\sigma\_B$ | upper-bound IS | None | Prompt average | Large scale validation and scaling laws |

$^\dagger$ implementations might differ, e.g., [Huggingface TRL](https://huggingface.co/docs/trl/grpo_trainer#computing-the-loss)

$^\ddagger$ with constant denominator

where

$$
\begin{align\*}
\rho\_t(\theta) &= \frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)} \\
\upsilon\_t(\theta) &= \pi\_\theta(a|s) - \pi\_{\theta\_{\text{old}}}(a|s) \\
M\_\text{sym}(\hat{A}\_t, \rho\_t, \epsilon) &= \begin{cases}
0 & \text{if } (\hat{A}\_t > 0 \land \rho\_t > 1 + \epsilon) \lor (\hat{A}\_t < 0 \land \rho\_t < 1 - \epsilon) \\
1 & \text{otherwise}
\end{cases} \\
M\_\text{asym}(\hat{A}\_t, \rho\_t, \epsilon\_l, \epsilon\_h) &= \begin{cases}
0 & \text{if } (\hat{A}\_t > 0 \land \rho\_t > 1 + \epsilon\_h) \lor (\hat{A}\_t < 0 \land \rho\_t < 1 - \epsilon\_l) \\
1 & \text{otherwise}
\end{cases} \\
M\_\text{div}(\hat{A}\_t, \upsilon\_t, \delta) &= \begin{cases}
0 & \text{if } (\hat{A}\_t > 0 \land \upsilon\_t > \delta) \lor (\hat{A}\_t < 0 \land \upsilon\_t < \delta) \\
1 & \text{otherwise}
\end{cases}
\end{align\*}
$$

Across these methods, a few patterns appear repeatedly:

**The critic appears unnecessary for LLM training.**
Every method since PPO has found that simpler baselines (group means, leave-one-out, greedy rollouts) match or exceed learned value functions while saving approximately 50% memory.
The LLM fine-tuning setting, where models start from strong pretrained checkpoints rather than random initialization, seems to make PPO's variance-reduction machinery largely redundant.
That does not mean we will never see value models again. They just don't justify the memory cost at the moment as variance reduction.

**Standard deviation normalization tends to hurt.**
Both Dr. GRPO and MaxRL show that dividing advantages by $\sigma$ adds too much weight on nearly solved problems.
The ScaleRL ablation confirms that DAPO (with standard deviation normalization) reaches a significantly lower asymptotic performance compared to CISPO and GSPO[[15]](#ref-zheng2025group) (not covered in this article, possibly in an extension).

**Loss aggregation is not a minor detail**
Dr. GRPO and DAPO show that sequence-level rewards combined with sample-level averaging can distort per-token learning signals.
The reduction of loss is a crucial part of the method, and a wrong choice might introduce a subtle bias.

**Trust regions are a good point for optimization**.
PPO's definition of trust regions ($\epsilon = 0.2$) seems to be remarkably well-chosen, as it works well across models and tasks.
However, recently many methods target trust regions and show improved performance:
DAPO relaxes it asymmetrically, CISPO clips weights instead of masking gradients, and DPPO argues that the sampled-token ratio is the wrong quantity to constrain in the first place.
The field has not yet converged on a good definition for trust regions, and there might not be a single, task- and model-agnostic one, but some further research here might lead to continued improvements.

**A provisional recipe is emerging.**
The strongest current large-scale evidence points toward critic-free training, token-aware or prompt-aware loss aggregation, softer or more principled trust region handling, and increasingly explicit attention to curriculum and compute allocation [[11]](#ref-chen2025minimax)[[13]](#ref-qi2026rethinking)[[14]](#ref-khatri2025art).
While that is progress, it can quickly change with the introduction of a new method or detail.

## Open Problems[¶](#open-problems "Permanent link")

Despite rapid progress, several fundamental challenges remain.
References in this section are incomplete, please reach out if you think I missed a one.

**Credit assignment.**[[16]](#ref-zhang2025lessons)[[17]](#ref-she2025r)[[18]](#ref-sharma2026prism)
Current outcome-based methods assign essentially the same reward to all tokens in a response.
That works surprisingly well, they are easy to implement, but it is clearly inefficient.
The token that caused a reasoning failure receives the same signal as boilerplate tokens around it.
Process reward models, step-level verifiers, search-based methods, and branch-sensitive training objectives all try to address this, but none has yet become the standard solution.

**Sample efficiency.**[[19]](#ref-mao2026dynamics)
Famously, the information gain in RL is just a single bit (correct/incorrect).
Most current recipes rely on multiple rollouts per prompt, often 8 to 64, to construct useful relative baselines.
That is expensive even with automatic verifiers, and much worse when verification is costly or partially manual.
Better reuse of unsuccessful samples, better offline-to-online mixing, or better prompt selection policies could reduce this cost substantially.

**Very hard problems.**[[20]](#ref-setlur2026reuse)[[21]](#ref-qu2026pope)
If a model never produces a correct rollout for a prompt, then all of the methods here provide no gradient.
Curriculum learning helps in practice, but it is only a workaround.
Stronger methods for extracting signal from partially correct trajectories, or for combining search with RL, remain an important research direction (connected to credit assignment).

**Extension beyond math and code.**[[22]](#ref-zhao2025learning)[[23]](#ref-lu2026golden)
Nearly all recent progress comes from domains with cheap and unambiguous verification (math and code).
Extending these methods to settings with noisy rewards, delayed rewards, subjective evaluation, or multi-turn interaction is still difficult.

**Empirical reliability.**[[24]](#ref-hu2025breaking)[[25]](#ref-yue2025does)
Perhaps the most underappreciated open problem is that much of the evidence in this area is still empirical, relatively narrow, and expensive to reproduce.
Many papers test one model family, one verifier setup, one dataset mix, and one compute budget.
As ScaleRL makes clear, an intervention can change early learning speed, asymptotic performance, or both, and these are not interchangeable.
We therefore still know less than it sometimes appears.
Some methods may be robust algorithmic improvements, while others may work mainly for a particular model, reward design, or training regime.
Both can be useful, but we need to know their limitations.

These open problems suggest a broader conclusion.
RL for LLMs is no longer bottlenecked by the absence of workable algorithms.
We now have several.
The harder problems are about efficiency, robustness, generality, and understanding which empirical improvements actually survive scale and transfer.

Comments, corrections, and relevant references are very welcome.
Just reach out on [X](https://x.com/a_weers) or write me at [[email protected]](/cdn-cgi/l/email-protection#d8bab4b7bf98b9afbdbdaaabf6bcbd)

---

**Further reading**

* [Denser rewards, nearly free: contrastive potential shaping](https://aweers.de/blog/2026/potential/): Proposal and exploration of improved credit assignment method for RL using contrastive likelihood evaluations for reward shaping.
* [DPO is SFT](https://aweers.de/blog/2025/dpo-is-sft/): If you want the gradient-level view of preference optimization.
  LoRA initialization — if you like empirical checks of small training details.
  nanochat’s gpt.py — if you want to understand the model code underneath all this.

## BibTeX

Copy

```
@misc{rl-for-llms-2026,
  author = {Alexander Weers},
  title = {State of RL for reasoning LLMs},
  url = {https://aweers.de/blog/2026/rl-for-llms/},
}
```

## References

1. Training language models to follow instructions with human feedback

   Long Ouyang et al.
   Advances in neural information processing systems, 2022
   [↩](#cite-ouyang2022training "Back to citation")
2. Openai o1 system card

   Aaron Jaech et al.
   arXiv preprint arXiv:2412.16720, 2024
   [↩](#cite-jaech2024openai "Back to citation")
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

   DeepSeek-AI
   2025
   [↩](#cite-deepseekai2025deepseekr1incentivizingreasoningcapability "Back to citation")
4. Proximal policy optimization algorithms

   John Schulman et al.
   arXiv preprint arXiv:1707.06347, 2017
   [↩](#cite-schulman2017proximal "Back to citation")
5. Simple statistical gradient-following algorithms for connectionist reinforcement learning

   Ronald Williams
   Machine learning, 1992
   [↩](#cite-williams1992simple "Back to citation")
6. Trust region policy optimization

   John Schulman et al.
   International conference on machine learning, 2015
   [↩](#cite-schulman2015trust "Back to citation")
7. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://openreview.net/forum?id=2a36EMSSTp)

   Qiying Yu et al.
   The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025
   [↩](#cite-yu2025dapo "Back to citation")
8. Deepseekmath: Pushing the limits of mathematical reasoning in open language models

   Zhihong Shao et al.
   arXiv preprint arXiv:2402.03300, 2024
   [↩](#cite-shao2024deepseekmath "Back to citation")
9. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs

   Arash Ahmadian et al.
   Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024
   [↩](#cite-ahmadian2024back "Back to citation")
10. [Understanding R1-Zero-Like Training: A Critical Perspective](https://openreview.net/forum?id=5PAF7PAY2Y)

    Zichen Liu et al.
    Second Conference on Language Modeling, 2025
    [↩](#cite-liu2025understanding "Back to citation")
11. Minimax-m1: Scaling test-time compute efficiently with lightning attention

    Aili Chen et al.
    arXiv preprint arXiv:2506.13585, 2025
    [↩](#cite-chen2025minimax "Back to citation")
12. [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710)

    Fahim Tajwar et al.
    2026
    [↩](#cite-tajwar2026maxrl "Back to citation")
13. Rethinking the Trust Region in LLM Reinforcement Learning

    Penghui Qi et al.
    arXiv preprint arXiv:2602.04879, 2026
    [↩](#cite-qi2026rethinking "Back to citation")
14. The art of scaling reinforcement learning compute for llms

    Devvrit Khatri et al.
    arXiv preprint arXiv:2510.13786, 2025
    [↩](#cite-khatri2025art "Back to citation")
15. Group sequence policy optimization

    Chujie Zheng et al.
    arXiv preprint arXiv:2507.18071, 2025
    [↩](#cite-zheng2025group "Back to citation")
16. The lessons of developing process reward models in mathematical reasoning

    Zhenru Zhang et al.
    Findings of the Association for Computational Linguistics: ACL 2025, 2025
    [↩](#cite-zhang2025lessons "Back to citation")
17. R-prm: Reasoning-driven process reward modeling

    Shuaijie She et al.
    Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025
    [↩](#cite-she2025r "Back to citation")
18. PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference

    Rituraj Sharma et al.
    arXiv preprint arXiv:2603.02479, 2026
    [↩](#cite-sharma2026prism "Back to citation")
19. Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models

    Yixiu Mao et al.
    arXiv preprint arXiv:2603.10887, 2026
    [↩](#cite-mao2026dynamics "Back to citation")
20. Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes

    Amrith Setlur et al.
    arXiv preprint arXiv:2601.18795, 2026
    [↩](#cite-setlur2026reuse "Back to citation")
21. POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration

    Yuxiao Qu et al.
    arXiv preprint arXiv:2601.18779, 2026
    [↩](#cite-qu2026pope "Back to citation")
22. Learning to reason without external rewards

    Xuandong Zhao et al.
    arXiv preprint arXiv:2505.19590, 2025
    [↩](#cite-zhao2025learning "Back to citation")
23. Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text

    Ximing Lu et al.
    arXiv preprint arXiv:2601.22975, 2026
    [↩](#cite-lu2026golden "Back to citation")
24. Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?

    Chuxuan Hu et al.
    arXiv preprint arXiv:2506.19733, 2025
    [↩](#cite-hu2025breaking "Back to citation")
25. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?

    Yang Yue et al.
    arXiv preprint arXiv:2504.13837, 2025
    [↩](#cite-yue2025does "Back to citation")

[← Back to blog](/blog/)

© 2026 Alexander Weers | Built with [Lectern](https://github.com/aweers/lectern)

[Privacy](/privacy/)
[Impressum](/impressum/)