title: From PCA to LoRA: Why Fine-Tuning Could Have Been Parallel All Along
description: A note on a 1933 deflation convention, the errors it accumulates when applied to LoRA fine-tuning, and a recent proof that we needn't have been so patient after all.

# From PCA to LoRA: Why Fine-Tuning Could Have Been Parallel All Along

## I.The Inheritance

Harold Hotelling, writing in 1933, proposed a procedure of unimpeachable elegance: to find the principal direction of a matrix, do not invert a large object — simply extract the leading rank-one component, subtract its outer product, and ask the same question of the residual. He called it *deflation*.

The procedure has aged into invisibility. It powers principal component analysis. It animates the power method. It underwrites the rank-one updates that have, more recently, become the standard method of fine-tuning very large language models — the technique known as LoRA.

The two problems are not, however, identical, and it is worth being precise about how they differ before we go further. The two optimisation objectives **PCA** (Hotelling, 1933) — find the best rank-\\(r\\) approximation of a single matrix \\(\mathbf{X}\\): \\\[\min\_{\mathbf{U}\,:\,\mathrm{rank}(\mathbf{U})\,\leq\,r}\; \tfrac{1}{2}\,\\|\mathbf{X} - \mathbf{U}\\|\_F\^2.\\\] **Low-rank linear regression** (this paper) — given inputs \\(\mathbf{X} \in \mathbb{R}\^{d\times n}\\) and outputs \\(\mathbf{Y} \in \mathbb{R}\^{m\times n}\\), find a low-rank operator \\(\mathbf{W} \= \mathbf{B}\mathbf{A}\\) that maps the former to the latter: \\\[\min\_{\mathbf{B}\,\in\,\mathbb{R}\^{m\times r},\;\mathbf{A}\,\in\,\mathbb{R}\^{r\times d}}\; \tfrac{1}{2}\,\\|\mathbf{Y} - \mathbf{B}\mathbf{A}\mathbf{X}\\|\_F\^2.\\\] When \\(\mathbf{Y} \= \mathbf{X}\\), the second reduces to a rectangular variant of the first; in LoRA, however, \\(\mathbf{Y}\\) is the desired output of the weight update applied to \\(\mathbf{X}\\), and the two problems are distinct.Hotelling's procedure addresses *low-rank matrix approximation*: given a single matrix \\(\mathbf{X}\\), find its best rank-\\(r\\) approximation. The problem the present paper studies, the one that underlies LoRA, is *low-rank linear regression*: given input features \\(\mathbf{X}\\) and target outputs \\(\mathbf{Y}\\), find a low-rank operator \\(\mathbf{W}\\) such that \\(\mathbf{Y} \approx \mathbf{W}\mathbf{X}\\), with \\(\mathbf{W}\\) factored as a sum of \\(r\\) rank-one components. The deflation idea — extract a leading rank-one component, subtract, ask the same question of the residual — survives the move from one problem to the other. The constants in the analysis, and the perturbation theorem at its centre, do not.

What we have inherited from Hotelling, and have rarely thought to question, is the *order* of operations. The procedure is sequential. One component at a time, in descending importance. The \\(k\\)th component depends on the \\((k-1)\\)th, which depended on the \\((k-2)\\)th, and so on, in a chain that reaches all the way back to the start. It works, of course. Procedures that have been in continuous use for ninety-two years tend to.

## II.The Convention

The sequential order, on inspection, was a matter of convenience and not necessity. There is no mathematical principle that requires the first component to be finished before the second begins. The order is simply how Hotelling described it, and how every implementation since has reproduced it.

The cost of this convention shows up in inexactness. In an ideal world, the rank-one solver returns the exact leading component of the residual, the residual passed to the second component is therefore correct, and so on down the chain. In any real world, the rank-one solver is approximate. The first component is *almost* right; the residual passed to the second is *almost* correct; the second compounds the first's slip with its own; the third inherits both. Errors do not merely persist along this chain. They *propagate*.

This phenomenon has been studied. A note on the literature Vandchali et al. (2024) made the error-propagation phenomenon precise for sequential rank-one deflation; Liao et al. (2023) showed that in the worst case the kth-component error inherits a factor inversely proportional to the smallest spectral gap among components *1* through *k*.In a well-conditioned problem with small individual errors, this is tolerable. In a poorly-conditioned problem with realistic solver tolerance, it is not.

There has been no satisfactory remedy. One could iterate sequential deflation, but that only refines each component against a fixed and corrupted target. One could decompose jointly, but then the rank-one structure — the very thing that makes the procedure efficient and interpretable — is lost. What if the convention itself were the problem?

## III.A Quiet Insight

Imagine, instead, that the \\(r\\) rank-one components were assigned to \\(r\\) workers, who would run simultaneously. At every communication round, each worker would broadcast its current best estimate of its assigned component to all the others, and would, in turn, receive theirs. The \\(k\\)th worker, upon receiving the latest estimates of components \\(1\\) through \\(k-1\\), would rebuild its own deflation target — the residual against which it trains — from those latest estimates rather than from any frozen ancestor.

What changes? The deflation target ceases to be a single bad inheritance and becomes a moving good one. As predecessors converge to their true values, the residual handed to the \\(k\\)th worker converges to the *correct* residual. The \\(k\\)th worker, training against a now-correct target, converges to the *correct* \\(k\\)th component. Early imprecision, instead of propagating downstream and freezing the descendants of every miscomputation, simply *resolves* — the moment its source has resolved.

We call this property self-correction, and it is the central claim of the paper. From PCA to regression Liao et al. (2025) proved a version of this for the symmetric eigenvalue case — the original principal-component setting in which Hotelling first wrote. The contribution of the present paper is to show that the same logic survives the move to bilinear regression, the setting in which low-rank adaptation operates. The proof is not quite a translation: Wedin's theorem replaces Davis-Kahan, and the constants change.The deflation mismatch, which in sequential deflation is fixed and generally nonzero after each component is finalised, shrinks across rounds in our setting — and not merely asymptotically, but at an exponential rate.

Round ℓ  8

Spectral gap  0.40

k \= 1 k \= 2 k \= 3 k \= 4 k \= 5

Drag the round slider. In the parallel mode, all five worker errors slide downward in concert; each worker enters its decay regime a few rounds after its predecessor. In the sequential mode, each worker's error *freezes* as soon as its predecessor finishes, locking in whatever imprecision was present at that moment. The spectral-gap slider controls how close together the singular values of the underlying matrix are; smaller gaps lengthen the warm-up but the contraction still occurs.

## IV.The Algorithm

The bare parallel-deflation backbone is the first of three mechanisms. Around it, AdaPaD adds two more, each addressing a different practical concern.

*The backbone, with staggered activation.* Worker \\(k\\) begins its formal participation at round \\(\ell \= k\\). Before that round, no predecessor has even an approximate estimate, and deflating against an empty residual would be meaningless. In the conventional reading, the \\(k\\)th worker is therefore idle for rounds \\(1\\) through \\(k-1\\).

*Advance learning.* That idleness is wasted. The \\(k\\)th worker, during its formal idle period, may privately compute against the current snapshot of the predecessor estimates — refining its component in a way that no one will see, but that ensures, by the time it does enter the visible record at round \\(k\\), it enters as a comparatively informed participant. The training target during advance learning is the *same* as during active learning: the current best estimate of the residual. The only thing that changes is whether the result is broadcast. We find this matters: the leading constant in the convergence bound improves by roughly a third when advance learning is enabled.

*Dynamic rank discovery.* The final mechanism removes from the user the burden of choosing the rank in advance. AdaPaD maintains a per-module rank counter (initialised at one) and a shared rank budget \\(B\\). Every \\(\Delta\\) steps, the modules whose parameters are *both* large in magnitude *and* still moving — measured by an exponential moving average of the parameter-gradient product, plus its variability — receive new capacity. Modules whose updates have stabilised do not. The rank distribution that emerges is heterogeneous: in our GLUE runs with DeBERTaV3-base, the attention-output and feedforward-intermediate projections in the final transformer layer reach the maximum rank of four, while many earlier-layer modules stabilise at rank one. The rank distribution becomes an *output* of training, not an input.

## V.On Proof

The argument is by strong induction on the component index. We sketch it.

*Base case.* The first component has no predecessor, and its deflation target — the full residual — is correct from the outset. Its error therefore decays at the rank-one solver's intrinsic contraction rate \\(\mathcal{F}\_1 \in (0, 1)\\), a quantity that depends on the local conditioning of the problem. This is, in essence, the classical analysis of power iteration: geometric decay, with no surprise.

*Inductive step.* Assume that all components with index less than \\(k\\) have entered their geometric-decay regimes and that their errors are bounded by component-specific envelopes. The \\(k\\)th worker's deflation target is built from those predecessor estimates. The distance between that target and the *ideal* target — the residual that would have been used had all predecessors converged exactly — is bounded by the sum of the predecessor errors. As that sum shrinks, the perturbation seen by the \\(k\\)th worker shrinks.

This is the point at which the move from PCA to regression bites. In the symmetric eigenvalue case, the perturbation theorem of choice is Davis-Kahan, which bounds the angle between true and perturbed top eigenvectors. In the rectangular case — which is what arises when the data matrix is not square, as it is not in regression — the appropriate tool is Wedin's theorem, the rectangular analogue. The Wedin bound is similar in spirit but differs in detail; the constants change.

A careful surrogate sequence — an upper-bounding piecewise definition of the error envelope, valid after a warm-up period satisfying a Lambert-W condition — closes the induction.

Under standard contraction and spectral-gap assumptions, for every component index \\(k\\) and every round \\(\ell\\) past its warm-up \\(s\_k\\), \\\[ G\_{k,\ell} \;\leq\; 6 \, (\ell - s\_k \+ 2) \, m\_k\^{\ell - s\_k \+ 1}, \\\] where \\(m\_k \in (0,1)\\) is a recursive effective contraction rate that compounds modestly with \\(k\\). The leading factor is geometric; the linear prefactor is mild.

The deflation mismatch, by direct substitution of the theorem into the residual identity, satisfies the same kind of vanishing bound — which is the formal statement of self-correction. The full proof, with the surrogate sequences and warm-up bookkeeping the induction requires, is in the paper's appendix on [arXiv](https://arxiv.org/abs/2605.10741).

## VI.The Evidence

We have run AdaPaD on the eight tasks of the GLUE benchmark with DeBERTaV3-base, the standard 184-million-parameter encoder. We compare at matched parameter budgets of approximately \\(0.34\\)M trainable parameters, against three families of baselines: fixed-rank LoRA at \\(r \= 2\\); the adaptive-rank methods AdaLoRA, SoRA, and dEBORA from their published numbers; and our reproduction of IncreLoRA, the closest algorithmic cousin.

[Table I. GLUE development set, DeBERTaV3-base, matched parameter budgets. AdaPaD discovers rank dynamically (\(r_{\max}=4\)). Published numbers are cited from the respective papers; ★ denotes our reproduction with the authors' code. Bold: best in column. Avg. is the unweighted mean across the eight tasks.]
| Method | #Params | CoLA | RTE | MRPC | STS-B | SST-2 | QNLI | QQP | MNLI | Avg. |
|----|----|----|----|----|----|----|----|----|----|----|
| LoRA, r \= 2 | 0.34M | 63.56 | 79.06 | 78.86 | 90.47 | 94.95 | 94.03 | 91.61 | 90.30 | 85.36 |
| AdaPaD | 0.34M | 72.26 | 86.28 | 91.67 | 91.86 | 96.33 | 94.36 | 91.55 | 90.37 | 89.34 |
| IncreLoRA ★ | 0.34M | 71.85 | 85.92 | 90.20 | 91.68 | 96.22 | 93.50 | 91.94 | 90.59 | 88.99 |
| AdaLoRA | 0.32M | 70.04 | 87.36 | 90.44 | 91.63 | 95.80 | 94.49 | 91.78 | 90.66 | 89.03 |
| SoRA (\\(r\_{\max}\=4\\)) | 0.47M | 71.05 | 86.04 | 90.20 | 91.76 | 95.57 | 93.92 | 92.06 | 90.38 | 88.87 |
| dEBORA | 0.40M | 68.72 | 83.75 | 90.16 | 90.84 | 95.29 | 93.42 | 91.88 | 90.01 | 88.01 |

AdaPaD achieves the highest average across all compared methods (\\(89.34\\), ahead of AdaLoRA at \\(89.03\\) and our IncreLoRA reproduction at \\(88.99\\)). Head-to-head against IncreLoRA at matched seed and budget, AdaPaD wins on six of eight tasks; the two losses, QQP and MNLI, are the two largest datasets, where the bulk of training occurs after the rank budget fills and the residual difference between AdaPaD's phase-split scheme and IncreLoRA's joint scheme reasserts itself.

On the question of speed: four NVIDIA H200 GPUs, components sharded round-robin, a single all-gather per communication round, rank \\(64\\). Per-batch wall-clock speedup is \\(3.62\times\\) over the single-GPU baseline, close to the ideal of \\(4\times\\). End-to-end, accounting for the staggered activation in the warm-up phase, the speedup is \\(2.41\times\\). Communication overhead is below one per cent of total time.

In synthetic settings — controlled spectra, controlled noise levels — we have verified each of the theorem's quantitative claims. Empirical contraction rates track the recursive bound; warm-up periods track the Lambert-W condition; the noise floor matches the closed-form generalisation bound to within sampling error. The full validation is in the paper's appendix.

## VII.What This Buys Us

Three things are now possible that were not before.

*Time.* The wall-clock cost of decomposing a matrix into \\(r\\) rank-one components is bounded by the longest worker, not the sum across workers. For practitioners who run such decompositions repeatedly — within a fine-tuning loop, say, or as part of a continual-learning pipeline — the saving compounds with every cycle.

*Quality, by way of vanishing algorithmic error.* The total generalisation error of any rank-one deflation method decomposes naturally into three terms: a *truncation* term (the unavoidable cost of approximating a higher-rank target with a fixed-rank model), a *statistical* term (the unavoidable cost of finite data), and an *algorithmic* term (the cost of the procedure's own imprecision). In sequential deflation, the algorithmic term is fixed and generically nonzero. In parallel deflation with self-correction, the algorithmic term decays to zero with the number of rounds. Two of the three terms become tight.

*Capacity discovery.* For the practitioner adapting a large pretrained model to a new task, the rank is no longer a parameter to be guessed and then tuned. The procedure discovers, for each adapter module, the amount of capacity it can use, and stops growing it when further growth no longer helps. In our experiments the discovered ranks cluster between one and four; in different domains, with different data scales, they would cluster elsewhere. We do not have to know in advance where.

## VIII.Open Questions

We acknowledge what the proof does not do.

The convergence analysis is for the bilinear regression model. The objective of full LoRA fine-tuning is a composition of bilinear adapter outputs with the pretrained network's nonlinear forward pass and a task-specific head. The empirical results confirm that AdaPaD's behaviour persists into this nonlinear regime, but the formal extension — via a neural-tangent-kernel argument, or local linearisation around a converged backbone — is the natural next step.

Synchronous communication is assumed throughout. Real distributed systems do not provide it for free. Bounded-staleness asynchrony, with provable guarantees, is open.

The bounds degrade when spectral gaps approach zero. The algorithm slows but does not fail; the warm-up periods grow as the gap shrinks. A more aggressive analysis — replacing the worst-case Wedin bound with a problem-adaptive variant — is plausible, and would tighten the rate in the regime where it currently most needs tightening.

Each of these is, we hope, a foothold for the next paper rather than a barrier to this one. Hotelling did not write the last word on deflation in 1933; we do not pretend to write it now.

## A Reader's Glossary

AdaPaD
:   *Adaptive Parallel Deflation.* The algorithm of the present paper. Trains rank-one components in parallel with self-correction, advance learning, and per-module rank discovery.
Advance learning
:   Practice in which a not-yet-activated worker privately refines its rank-one component against the current snapshot of predecessor estimates. Its result is not broadcast until the worker's formal activation round.
Contraction rate
:   A factor \\(m \in (0,1)\\) by which an error is multiplied each round. The lower the rate, the faster the convergence.
Davis-Kahan
:   A perturbation theorem for the eigenvectors of symmetric matrices. The PCA analogue of Wedin's theorem.
Deflation
:   The procedure, originating with Hotelling (1933), of extracting rank-one components from a matrix one at a time by subtracting each previously extracted component before extracting the next.
Deflation target
:   The residual that the \\(k\\)th worker trains against. In sequential deflation, this is fixed once each predecessor finishes. In parallel deflation, it is recomputed every round from the most recent predecessor estimates.
LoRA
:   *Low-Rank Adaptation.* A parameter-efficient fine-tuning method that adapts a pretrained weight matrix \\(\mathbf{W}\\) by training a small low-rank update \\(\Delta\mathbf{W} \= \mathbf{B}\mathbf{A}\^\top\\), where \\(\mathbf{B}\\) and \\(\mathbf{A}\\) are much smaller than \\(\mathbf{W}\\).
PEFT
:   *Parameter-Efficient Fine-Tuning.* The family of methods that adapt large pretrained models by training only a small fraction of their parameters. LoRA and its variants are members of this family.
Rank-one component
:   A matrix of the form \\(\mathbf{b}\mathbf{a}\^\top\\), where \\(\mathbf{b}\\) and \\(\mathbf{a}\\) are column vectors. The simplest non-trivial low-rank matrix.
Self-correction
:   The property that errors in early rank-one components diminish as the components that follow refine the targets seen by their predecessors. The defining advantage of parallel over sequential deflation.
Spectral gap
:   The difference between consecutive singular values of a matrix. Larger gaps make the leading direction easier to estimate.
Warm-up period
:   The number of communication rounds, denoted \\(s\_k\\) for component \\(k\\), before that component's error envelope enters its exponential-decay regime.
Wedin's theorem
:   A perturbation theorem for the singular vectors of rectangular matrices. The regression analogue of Davis-Kahan.
