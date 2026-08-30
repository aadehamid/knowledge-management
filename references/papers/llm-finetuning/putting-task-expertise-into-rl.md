title: Putting Task Expertise into RL Achieves State-of-the-Art Performance on Text-to-SQL
description: An expert-verified training set and a reshaped reward let RLVR on Tinker train a single model past human-level text-to-SQL accuracy, at a fraction of frontier cost.
author: Thinking Machines Lab
keywords: thinky thinkingmachines machine learning deep learning ai

# Putting Task Expertise into RL Achieves State-of-the-Art Performance on Text-to-SQL

Many industries rely on relational databases that are queried with SQL. Most SQL is machine-written, but humans alone likely write billions of custom SQL queries each monthBased on our internal estimates and publicly available data, such as from [Snowflake filings](https://www.sec.gov/Archives/edgar/data/1640147/000164014722000023/snow-20220131.htm). prompted by business questions. They are quite good at it — humans score 92.96% on [BIRD](https://arxiv.org/pdf/2305.03111), a realistic benchmark for translating natural-language questions into SQL.

However, AI performance on text-to-SQL has lagged behind. [LLM scores on the BIRD leaderboard](https://bird-bench.github.io/) improved from just below 70% in 2024 to 82% today. Frontier models like GPT-5.6 Sol Ultra and Claude Fable 5 can score in the mid-80s, albeit at a cost that is prohibitive for high-volume applications. This isn’t for lack of training data: SQL is widely represented in the internet content used in LLM pretraining. The challenge for AI is in navigating the ambiguous questions and highly-contextual schema that characterize real-world examples.

A common approach for improving AI performance on tasks people understand well is building agentic scaffolding. Systems such as [OpenHands](https://github.com/OpenHands/OpenHands), [AI co-scientist](https://arxiv.org/pdf/2502.18864), and [MetaGPT](https://github.com/FoundationAgents/MetaGPT) decompose a task into stages, each with its own prompt or model call. [Text-to-SQL scaffolds](https://link.springer.com/chapter/10.1007/978-981-92-3438-7_27) follow the same pattern. A schema-linking stage narrows thousands of columns down to a candidate set.Real-world, enterprise data systems (including databases, data warehouses, and data lakehouses) contain up to [millions of columns](https://www.ibm.com/think/insights/data-consumption-challenges). Answering business questions often involves understanding which columns to use. Academic benchmarks are simpler. A generation stage samples queries. A self-correction stage repairs execution errors. A selection stage votes among the survivors. Every component is a separate call, and the orchestration is typically tuned for a given benchmark.

*State-of-the-art text-to-SQL scaffolding from \[1, 2, 3\]. The three systems differ in their component inventory but share a design premise: the model is held fixed and task performance is improved by increasing the number and structure of calls made to it.*

Scaffolding is an attempt to get over the limitations of model reasoning by making it adhere to a sequence of steps that mirrors how a human would approach the task. And yet, the best scaffolded models still lag 11 points behind humans on SQL. Human professionals acquire their skills with repeated experience, not by being handed a list of instructions — the same should be true of LLMs. Experience with the task should be used to train better reasoning about queries and databases into the model, instead of simply updating the prompts it receives from the scaffold.

In this blog post we describe fine-tuning a model that achieves human-level accuracy on text-to-SQL without scaffolding, using reinforcement learning with verifiable rewards (RLVR) on [Tinker](https://thinkingmachines.ai/tinker/).

Extracting the correct answer with SQL is a verifiable task that can be trained in a straightforward manner. However, the performance gap of SQL models suggested that the standard recipe can be improved. Our approach has two crucial improvements: an expert-verified training set purged of the label errors that could [poison RLVR](https://arxiv.org/abs/2603.16140), and a reward-shaping technique that targets two common failure modes of RLVR in this domain.

The trained model, [ReViSQL-K2.6](https://arxiv.org/abs/2603.20004), exceeds the human mark of 92.96% when picking from 16 samples (SC-16)SC refers to self-consistency selection, where we group concurrently generated SQL queries by their execution results and randomly select a query from the majority group. Neither component is part of a traditional agentic scaffold: sample generation simply draws multiple outputs from the same model prompt, without separately prompted intermediate steps, while majority voting requires no additional model calls. at a cost of $0.56 per task. It is more accurate than Fable 5 and GPT-5.6 Sol Ultra at 12–15% of their cost, and much more accurate than any scaffolded model on the leaderboard.

*Our single-model method, ReViSQL-K2.6, achieved higher accuracy than frontier models and prior open-source scaffolded pipelines on the expert-verified Arcwise-Plat-SQL benchmark. With 16-sample self-consistency (SC-16), ReViSQL-K2.6 exceeded the 92.96% human proxy for the first time.*

Code, data, and training recipes are at [github.com/uiuc-kang-lab/ReViSQL](https://github.com/uiuc-kang-lab/ReViSQL). We further describe our methods in detail in our [technical report](https://arxiv.org/abs/2603.20004).

## Curating high-quality training data {#curating-high-quality-training-data}

RLVR is effective at improving domain-specific reasoning, but it is sensitive to data with incorrect labels. In RLVR, the scalar reward is the entire learning signal for the training step. Mislabeled instances reverse the signal, degrading learning significantly. Our research found that [algorithmic tweaks cannot compensate for this loss](https://ddkang.substack.com/p/noisy-data-breaks-rlvr) — cleaning up the data is crucial for RLVR to work well.

We found existing text-to-SQL data to be extremely noisy.The noisiness of public SQL datasets was noted by many others such as [Pourreza and Rafiei (2023)](https://arxiv.org/abs/2310.18538) and [Wretblad et al. (2024)](https://arxiv.org/abs/2402.12243). [Our analysis](https://arxiv.org/abs/2507.02825) has shown that many widely known benchmarks contain large amounts of noise. We sampled 2.5k instances from BIRD Train, a training dataset for text-to-SQL. Our audit found errors in every component of the dataset: the questions, the external knowledge supplied, and in more than half of the “golden SQL queries” that the model’s answer is compared against.

| Error type | Share of audited instances |
|----|----|
| Gold SQL query incorrect | 52.1% |
| Natural language question flawed | 26.2% |
| External knowledge entry wrong | 18.2% |
| Unanswerable given the schema (discarded) | 1.5% |
| At least one of the above | 61.1% |

Table 1: Annotation error rates across 2.5k instances sampled from BIRD Train. Categories overlap, so the total is not a sum.

We cleaned up the training set in a multi-stage process. First, an LLM (OpenAI’s o3) and a human expert reviewed each instance and flagged errors. The expert review found that the LLM auditor was precise in catching annotation mistakes (90.6% precision) but only caught 24.5% of the errors flagged by humans. The errors and proposed fixes from this first stage were sent to a different expert for verification. Where the verifier disagreed with the initial auditor, the sample was sent back for additional loops of conflict resolution.

We released the cleaned-up training set to the community as [BIRD-Platinum](https://github.com/uiuc-kang-lab/ReViSQL/blob/main/data/bird-plat-2.5k-v1.json).

*The BIRD-Platinum data correction pipeline with human experts.*

We suspected that the evaluation dataset, BIRD Mini-Dev, likewise contained annotation errors. A first cleanup pass of BIRD Mini-Dev was done by [Arcwise](https://github.com/AlibabaResearch/DAMO-ConvAI/issues/39#issuecomment-2303283506) and corrected errors in 32.3% of instances. We did a second pass that confirmed the vast majority of Arcwise’s flags and found many more, bringing the total detected error rate in BIRD Mini-Dev to 52.8%. The evaluation set with the gold query errors cleaned up was released as [Arcwise-Plat-SQL](https://www.vldb.org/pvldb/vol19/p931-jin.pdf).

## BIRD-Platinum lifts RLVR above prior best models {#bird-platinum-lifts-rlvr-above-prior-best-models}

We fine-tuned Kimi-K2.6 with RLVR on BIRD-Platinum to produce ReViSQL-K2.6. Training on verified data alone lifted ReViSQL-K2.6 well above both frontier generalist LLMs and the leading open-weight fine-tuned text-to-SQL models on Arcwise-Plat-SQL, with an accuracy score of 88.55%. This shows that the annotation errors in standard training data were the binding constraint on RLVR for text-to-SQL.

*ReViSQL-K2.6, fine-tuned on BIRD-Platinum, achieves the highest accuracy on an expert-verified variant of BIRD, outperforming GPT-5.6 Sol Ultra, Claude Fable 5, and the strongest open-weight fine-tuned text-to-SQL models (Infly-RL-SQL-32B, OmniSQL-32B, XiYanSQL-32B, Arctic-R1-7B).*

To demonstrate that this approach generalizes to other models and eval sets we didn’t touch, we fine-tuned Qwen3-235B-A22B with RLVR on BIRD-Platinum and the original BIRD Train. We tested the model on two new text-to-SQL benchmarks, widely considered more difficult than BIRD:

- [**Spider2-SQLite**](https://arxiv.org/abs/2503.02240): a variant of the [Spider2](https://arxiv.org/abs/2411.07763) benchmark that contains complex queries with 5.2× as many tokens on average as Arcwise-Plat-SQL.
- [**Spider2-Snow**](https://arxiv.org/abs/2411.07763): a variant of Spider2 that uses the [Snowflake SQL dialect](https://docs.snowflake.com/en/sql-reference-commands).

*Qwen3-235B-A22B fine-tuned on BIRD-Platinum outperforms the same model trained on the original BIRD Train across Arcwise-Plat-SQL, Spider2-SQLite, and Spider2-Snow by 16%, 12%, and 14%, respectively.*

Training on the more carefully curated BIRD-Platinum improves the model’s accuracy by 16% on Arcwise-Plat-SQL, 12% on Spider2-SQLite, and 14% on Spider2-Snow compared to BIRD Train. This indicates that our verified data produces a more transferable learning signal across benchmarks and SQL dialects.

## Accurate reward signal for text-to-SQL RLVR {#accurate-reward-signal-for-text-to-sql-rlvr}

Training on clean data brought the fine-tuned model closer to human parity, but a gap of over 4% remained. We looked at the cases where the model failed to identify patterns, which led us to examine the reward function used in training.

Standard text-to-SQL RLVR assigns a reward of 1 when the generated query returns the same result as the gold query on the benchmark database. This mirrors the scoring used in evaluation, but doesn’t fully capture the general behavior we want the model to learn. We focused on two ways result-based reward can diverge from the intended behavior and amended the reward function to address them.

### Divergence 1: execution match is not semantic equivalence {#divergence-1-execution-match-is-not-semantic-equivalence}

The standard result-based reward checks the generated query’s output on a single database instance, but this doesn’t guarantee that the result would hold for a different one. A wrong join key or a dropped predicate can be missed when the particular database instance does not expose the error. Only queries that are semantically equivalent are guaranteed to produce the same result on any database instance.

We test for the semantic equivalence of SQL queries with [VeriEQL](https://arxiv.org/abs/2403.03193), a solver that uses bounded verification and incurs negligible CPU costs relative to the total training costs (less than 0.1%). In a pilot training run, we found that 32.8% of positive result-based rewards were given to queries that weren’t fully equivalent to the correct one. That means that nearly one time in three, the reward reinforced the wrong query.

SELECT AVG( order\_total ) FROM ( SELECT o.order\_id, SUM(i.price) AS order\_total FROM orders o JOIN items i ON o.order\_id \= i.order\_id WHERE o.customer \= 'Lucas...' GROUP BY o.order\_id );

SELECT AVG( i.price ) FROM orders o JOIN items i ON o.order\_id \= i.order\_id WHERE o.customer \= 'Lucas...'

*A false-positive reward. The generated query averages line-item prices instead of order totals, yet matches the gold result because each order has a single line item. Result-based grading rewards it as correct.*

We updated the reward signal by downweighting it in cases where the query was accepted by execution matching but wasn’t equivalent according to VeriEQL. With an additional source of verification, the reward pushes training towards the correct query that generalizes to different database instances.

### Divergence 2: outcome rewards are blind to provided knowledge {#divergence-2-outcome-rewards-are-blind-to-provided-knowledge}

BIRD-style problems provide external knowledge alongside the question in a prompt. A result-based reward conditions only on the final result, which means it cannot distinguish a model that read the provided information from one that guessed correctly based on its pretraining priors. For example, because “sodium \= 0” and “sodium \< 5” produce the same result set, a result-based reward cannot distinguish a model that correctly uses external knowledge to choose “sodium \= 0” from one that memorizes or hallucinates “sodium \< 5.” In the absence of a gradient pushing models to incorporate external knowledge, they tend to default to the prior. In a pilot analysis on a validation set, 24.2% of failures were traced to the model ignoring the necessary information that was supplied.

SELECT CAST(SUM( CASE WHEN sodium \= 0 THEN 1 ELSE 0 END ) AS REAL) \* 100 / COUNT(\*) FROM recipes r WHERE r.source \= 'California...'

SELECT CAST(SUM( CASE WHEN sodium \< 5 THEN 1 ELSE 0 END ) AS REAL) \* 100 / COUNT(\*) FROM recipes r WHERE r.source \= 'California...'

*A failure from ignoring external knowledge. The external knowledge specifies that “sodium-free” means sodium \= 0, but the generated query applies the threshold sodium \< 5 instead. Because outcome-based rewards condition only on the final result, they provide no signal that the model disregarded the provided evidence and wrote a semantically incorrect SQL query.*

We address this with rule-based process rewards. The model must emit a requirement block that translates each external-knowledge entry into an explicit query constraint, and a verification block that audits the generated query against those constraints, with penalties for non-compliance. Our process reward encourages the model to ground its reasoning and generation in the provided knowledge rather than blindly relying on its pre-trained prior. The rewards are rule-based rather than model-graded, making the scoring cheap and free from contamination by a judge model’s priors.

## Training recipe {#training-recipe}

We provide our training recipe for ReViSQL-K2.6. This recipe can also be reproduced using our [code built on the Tinker APIs](https://github.com/uiuc-kang-lab/ReViSQL).

| Component | Setting |
|----|----|
| Base model | moonshotai/Kimi-K2.6 |
| Training data | [BIRD-Platinum](https://github.com/uiuc-kang-lab/ReViSQL/blob/main/data/bird-plat-2.5k-v1.json) |
| Train/validation split | 85:15 |
| RL objective | CISPO |
| Batch size | 64 |
| Group size | 16 |
| Learning rate | 5×10^−5^ |
| LoRA rank | 32 |
| Maximum number of input tokens | 32,768 |
| Maximum interaction turns between the model and the environment (a database instance) | 5 |
| Maximum number of output tokens per turn | 3,072 |
| VeriEQL reward shaping | An execution match refuted by VeriEQL receives a penalty of 0.2 |
| Process reward shaping | Non-compliance with the required external knowledge analysis receives a penalty of 0.1 |
| Checkpoint selection | Highest validation accuracy |

Table 2: Training configuration.

## Results {#results}

We present ReViSQL-K2.6, the model fine-tuned on Tinker with verified data and both reward modifications. Under greedy decoding (single sample, temperature \= 0) our model achieves an accuracy of 91.37% on Arcwise-Plat-SQL at a cost of $0.035 per task. This is an 8.4 point improvement over [OpenSearch](https://arxiv.org/abs/2502.14913), the strongest prior open-source pipeline, at a 37% lower cost. The cost advantage is a direct consequence of removing auxiliary scaffolding around the model.

If ReViSQL-K2.6 votes among 16 candidates generated with temperature \= 1, accuracy rises to 92.97% at a cost of $0.56 per task. This is the first time a text-to-SQL AI system has exceeded the human benchmark to our knowledge.

*ReViSQL-K2.6 on Arcwise-Plat-SQL with 1, 4, 8, 16, and 32 samples against five open-source pipelines. Baselines are built on GPT-5.2 (GenaSQL, OpenSearch, SHARE) or XiYanSQL-QwenCoder-32B-2412 (CSC-SQL, Contextual). The 92.96% human level is marked with a line. Our model exceeds every pipeline by 8 to 22 points at comparable or lower cost per query.*

## Conclusion {#conclusion}

When AI shows poor performance in a domain-specific task, the common response is to add scaffolding around the task execution. This approach improved performance for text-to-SQL models somewhat, but it ultimately hits the ceiling imposed by the base model’s capability. This doesn’t mean that scaffolding is worthless. Rather, it suggests that the task knowledge contained in the scaffold belongs in the training signal, the same training that would uplift the model’s overall capacity.

The common thread with our previous work on [AI trained for financial judgment](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/) is that custom models can outperform the frontier on a wide range of tasks involving expert taste and judgment, often at a fraction of the cost. This requires expert judgment to be part of the training process: in identifying how AI fails, labeling the training data, and aligning training to the intended behavior.

In the case of text-to-SQL we saw major improvements from both a rigorous cleanup of the data and the shaping of the reward function to teach the model the correct skill. This took more effort upfront, but the resulting model achieves better performance at lower cost than both humans and scaffolded models. Putting task expertise into task-specific training is ultimately what scales.

## Citation {#citation}

Please cite this work as:

```
Zhu, Yuxuan et al., "Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering", arXiv:2603.20004 (2026).
```

BibTeX:

```
@article{zhu2026revisql,
  title = {Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering},
  author = {Zhu, Yuxuan and Jin, Tengjun and Choi, Yoojin and Kang, Daniel},
  journal = {arXiv preprint arXiv:2603.20004},
  year = {2026}
}
```
