title: Flow generation through natural language: An agentic modeling approach \(2026\) - Shopify
description: We fine-tuned Qwen3-32B into a tool-calling agent that generates Flow automations: 2.2x faster, 68% cheaper, with a weekly retraining flywheel.

# Flow generation through natural language: An agentic modeling approach (2026) - Shopify

If you're building AI products on top of closed models, anyone with an API key can get similar capabilities. Lasting differentiation comes from proprietary data, the training recipe, the infrastructure, and the speed of iteration.

Shopify has something most companies don't: a product surface where millions of merchant interactions directly signal whether the model's output is any good. That feedback loop is the foundation, but only if you keep learning from it.

We fine-tuned a tool-calling agent to turn natural language into a Shopify Flow for [Sidekick](https://www.shopify.com/ca/sidekick "Shopify Sidekick"), our AI commerce assistant. It's 2.2x faster, 68% cheaper, and outperforms closed models.

Along the way, we found lessons no paper warned us about. Data preprocessing decisions, from representation design to formatting details, that compound to swing accuracy by double digits. Silent infrastructure failures that degrade your model with zero warnings and take days to trace. Benchmark parity that masks a 35% gap once real users show up.

This post covers the problems we faced, how we fixed them, and what to look for if you're doing the same.

![Data pipeline > Flywheel](https://cdn.shopify.com/s/files/1/0779/4361/files/image2_a0d3c058-3cb2-4b24-87df-8ea9685879f7.png?v=1776796283)

## Building the training dataset

Shopify Flow is an automation platform where store owners build workflows from triggers, conditions, and actions. For store owners who aren't engineers, building the right workflow from a blank canvas is daunting. Sidekick generates it from plain English.

![Shopify Admin showing Flow](https://cdn.shopify.com/s/files/1/0779/4361/files/image7.png?v=1776796370)

### The cold start problem

Fine-tuning required training data, but since the feature hadn't been deployed yet, there were no production conversations to learn from.

We reverse-engineered user intent from existing production workflows. Thousands of anonymized store owners had already built workflows manually in Flow. We sampled those and filtered for quality: workflows that had run at least once in the last seven days, from merchants with two or more qualifying workflows, with one example per descriptor to ensure diversity across workflow types.

With a set of validated workflows, we worked backwards:

1.  **Sample a workflow.** Pick a popular, validated workflow from production.
2.  **Generate a user query.** Use a stronger LLM to produce a plausible natural-language request that would lead to this workflow.
3.  **Construct the tool trajectory.** Build the full multi-turn sequence of tool calls that an ideal agent would execute to arrive at this workflow. This was the bulk of the engineering effort.

We fine-tuned Qwen3-32B on this synthetic dataset and evaluated it against a benchmark of 300 hand-crafted examples covering the breadth of expected Flow usage. An [LLM evaluation framework](https://shopify.engineering/building-production-ready-agentic-systems "LLM judge on Shopify Engineering Blog") compares the generated workflow against the expected one for semantic correctness, and validates syntactic correctness programmatically.

We looked at three metrics:

-  **Semantic correctness:** Does the generated workflow do what it's supposed to? An LLM judge compares the output against the expected workflow.
-  **Syntactic correctness:** Are there errors that would cause it to fail? Malformed conditions, incorrect references, invalid configurations. Checked programmatically.
-  **Latency:** Time from request to workflow delivery.

If you're building an agent without interaction data, start with the output artifacts your users already produce and work backwards from them. That is often the right first step before your metrics have caught up. As shown in the table above, there is still a meaningful gap to close. Our second lesson, which we discuss below, is that teaching the model to generate Flows in Python can help narrow that gap further.

### Training in-distribution: the Python DSL

Shopify Flow workflows are represented internally in a JSON-based domain-specific language (DSL) designed for backend parsing, validation, and execution. That format is ideal for production systems, but it's a poor fit for LLMs. Conditional, program-like logic that would normally appear as code is embedded in deeply nested JSON, a pattern that's rare in pretraining data.

Rather than forcing the model to learn Flow's native format from scratch, we reformulated the task in a representation closer to the model's training distribution. Workflows are programs, so we taught the model to write them as Python.

A transpiler converts the JSON DSL into semantically equivalent Python:

Same workflow, same semantics, but the model now generates Python instead of a data format. Python is far closer to code and logical reasoning, and it makes up a large share of pretraining data. The fine-tuned model draws on familiar patterns: decorators, if/else logic, variables, for loops, and function calls.

With the same training data, switching from the JSON DSL to the Python DSL improved syntactic correctness by 22 points and semantic correctness by 13 points. Moving the target format from out-of-distribution to in-distribution turned the problem from "learn a new language and the task" into "learn the task."

Making this work required building a round-trip transpiler between Python and Flow's JSON representation to handle the full complexity of Flow logic without losing meaning in either direction.

Reliability was backed with extensive tests. We round-trip tested every workflow merchants created through Sidekick in production: converting from JSON to Python and back to JSON, then verifying the output matched the original exactly. Any mismatch was caught before it could reach training data. This process ran continuously across all production workflows, giving us confidence the transpiler handled the full range of real-world patterns.

At inference time, the model writes Python. The transpiler converts it to JSON for the Flow backend. Store owners never see Python, and the backend never has to understand it. Python is the model's internal language.

Prior work has explored Python as an intermediate representation ([SPEAC](https://arxiv.org/pdf/2406.03636 "SPEAC"), [LLMLift](https://arxiv.org/pdf/2406.03003 "LLMLift"), [WorkflowLLM](https://arxiv.org/pdf/2411.05451 "WorkflowLLM")), but via prompting or without a round-trip transpiler. What distinguishes this approach is the full loop: fine-tuning on Python combined with a transpiler back to the production DSL, without changing any downstream systems.

If you're training a model on a custom DSL, consider translating it into a language the model already knows. This helps separate learning the format from learning the task. As the results show, the gap narrows, but there is still room for improvement. At that point, the next step is to bring the system into production, learn from real usage, and incorporate real user feedback.

### Mirroring the production environment

Representation was one half of the data problem. The other half was making sure the model's training data matched exactly what it would see in production.

We knew training data should match production. What we didn't expect was how sensitive the model is to the *degree* of match. Every difference we closed, no matter how minor, improved eval scores:

-  **Tool naming and ordering:** Training data used the full prefixed name `flow_app_agent_task_search`. At inference, the same tool was called `task_search`. Functionally identical, but the model treated them as different tools. Removing the prefix from training data to match inference improved accuracy. The order in which the tools appeared in the system prompt also mattered. Shuffle the order between training and serving, and performance drops.
-  **Tool response format:** Tool responses return JSON objects with multiple fields. In the training data, we sorted keys alphabetically. If production returned them in a different order, or included an extra field, the model noticed. Any drift between what the training data showed and what production APIs actually returned degraded accuracy.
-  **System prompt and tool descriptions:** Tool descriptions in production changed frequently as the product team iterated on behavior. Every update had to be reflected in the training data, or the model's behavior drifted. Keeping both in sync was an ongoing process, not a one-time fix.

None of these are about the logic of the task. They are formatting details. The model treats every token as a signal, whether you intended it or not.

### Optimizing the tool-calling stack

When an agent calls tools, every response becomes part of the context. Context grows, latency grows, cost grows. Worse, irrelevant context dilutes the signal. The model reasons less accurately when it’s processing information it won't use.

We restructured our tool interfaces to minimize context at each step. Instead of returning full details for every result upfront, tools return lightweight summaries first. The model scans the summaries, selects what it needs, then retrieves full details only for those necessities. Two cheap calls instead of one expensive one.

For example, Flow has hundreds of available triggers, conditions, and actions. A search might return 100 matches. Rather than loading the full configuration schema for each one, `task_search` returns just names and descriptions. The model picks the 2-3 it actually needs, then calls `task_configuration` to get the full schema only for those. The context stays small, the reasoning stays focused.

![Merchant request > Shopify Flow workflow created](https://cdn.shopify.com/s/files/1/0779/4361/files/image1_92d92421-357a-4ef5-805f-569ab8a67ad0.png?v=1776796570)

## Making training fast

As our data pipeline grew, so did a tension: more training data improved accuracy but slowed each run. Slower runs meant fewer iterations, and fewer iterations meant slower improvement. We needed a way to use all the data and still retrain weekly.

We built the infrastructure to make both possible. Qwen3-32B trains on two nodes of H200 GPUs with Fully Sharded Data Parallel (FSDP). A full training run takes 12 hours, fast enough for weekly retraining with multiple experimental runs in between.

The full pipeline, from data collection through training, evaluation, and deployment, runs on [Tangle](https://shopify.engineering/tangle "Tangle on Shopify Engineering Blog"), Shopify's open-source ML experimentation platform. Tangle composes each step into a single reproducible workflow with intelligent caching. Only the affected steps re-run when one part changes.

![Tangle dashboard: Shopify Flow](https://cdn.shopify.com/s/files/1/0779/4361/files/image3_1a0e8f26-d7ba-4ff9-a63d-39bf8454449f.png?v=1776796646)

CometML tracks every run. HuggingFace hosts datasets and checkpoints. CentML serves the model in production. Weekly retraining runs without manual intervention.

![Tangle pipeline](https://cdn.shopify.com/s/files/1/0779/4361/files/image5.png?v=1776796678)

## Evaluation: benchmarks aren't ground truth

Synthetic data got us to parity on offline benchmarks. By every metric we tracked, the fine-tuned model was ready for production. We deployed it to 1% of traffic to see how it held up.

At 1% traffic, the fine-tuned model's workflow activation rate (whether store owners actually turn on the workflows Sidekick generates) came in 35% lower than the prompt-based agent. The benchmark covered what we expected merchants to ask. It didn't cover what they actually asked: editing existing workflows, handling email configurations, working with third-party integrations, and asking questions about Flow without intending to create a workflow.

The model performed well in-domain, but real traffic quickly surfaced out-of-distribution requests that our synthetic data had not covered. The low-traffic early deployment showed us exactly where to focus next. Activation rate was our first production signal, but it turned out to be noisy: it reflects merchant behavior, not model quality. We therefore optimized for a domain-expert-calibrated [LLM judge](https://shopify.engineering/building-production-ready-agentic-systems "LLM Judge"), which we describe next, while keeping activation rate as a guardrail to ensure we did not regress.

## Flywheel: from catching up to pulling ahead

### Closing the gap

The 1% deployment showed us exactly where the model was falling short. We needed a system that could diagnose those gaps, fix them, and retrain fast. Not once, but continuously.

We built an LLM-based judge that scores each conversation across the workflow lifecycle: whether the assistant correctly understood the merchant's intent, chose a Flow solution only when appropriate, selected the right components, and gave clear next steps. The judge grades each facet separately rather than treating quality as a single pass/fail outcome. To calibrate it, we collected human annotations on hundreds of conversations and tuned it until its scores aligned with human judgment, then validated against production activation rate.

A tagging system classifies every workflow along multiple dimensions: which triggers it uses, what conditions it checks, which actions it invokes, and whether it involves third-party integrations. Comparing performance across tagged slices pinpoints exactly where the model struggles. When performance drops on a particular slice, we know what kind of data to add.

The judge and tagging system together form the diagnostic layer. The fixes were concrete:

- Email workflows accounted for 25% of failures, so we added email-specific examples
- Diverse condition patterns accounted for 16%
- Workflow editing, which was something synthetic data had never covered

The following diagram shows our progress in Flow modeling, with quality improving steadily over time as measured by our LLM judge:

![LLM judge score over each month](https://cdn.shopify.com/s/files/1/0779/4361/files/LLM_Judge_score_over_each_month.png?v=1776860795)

### Continuous improvement

Closing the gap was the first test. Staying ahead is the real goal.

Every production conversation becomes a training signal. We sample high-quality examples: conversations where merchants actually activated the workflow afterwards. The judge scores them, and high-scoring conversations are routed into the training pool automatically. Low-scoring ones are quarantined for review.

The loop runs weekly:

1. Ingest production conversations
2. Score with the LLM judge
3. Route high-quality examples into training; quarantine low-quality for review
4. Identify gaps through tagged slice analysis
5. Retrain and deploy

The system improves as production traffic shifts, freeing the team to focus on expanding coverage and fixing systematic gaps rather than hand-curating data. The approach is similar in spirit to Karpathy's [Autoresearch](https://shopify.engineering/autoresearch "Autoresearch on Shopify Engineering Blog"), an automated loop that evaluates, keeps what works, discards what doesn't, and iterates—but applied to production data curation rather than training code.

## What's next

The flywheel is running, but the race between in-house and closed-source models doesn't stop. Every few months, a new frontier model raises the bar. The only way to stay ahead is to keep compounding: better data, better training, better evaluation, faster iteration. Here's where we're pushing next.

**Simulation environments.** A sandbox where the model can generate workflows and receive structured feedback on whether they would succeed, without impacting real merchants. The model writes test cases and runs them against a simulated Flow environment, creating a setting for verifiable rewards. This opens the door to distillation from stronger teacher models and on-policy optimization.

**From off-policy to on-policy.** Everything so far is off-policy: the model learns from curated examples collected after the fact. With verifiable rewards from the simulation environment, the next step is policy optimization where the model learns from its own generated trajectories. The goal is a model that discovers better strategies, not one that only replicates what it's seen.

**From manual calibration to self-improving evaluation.** Today, the LLM judge is calibrated against human annotations and production activation rate. But merchant behavior shifts, new integrations launch, and new workflow patterns emerge faster than manual recalibration can keep up. Automating judge calibration against live production signals is the next evaluation challenge.

## Results in production

The fine-tuned Flow agent now serves the majority of our production traffic.

No single technique got us here. Each stage built on the last. Synthetic data generation needed the Python DSL to close the accuracy gap. The DSL needed production mirroring to hold up in the real environment. Production mirroring needed infrastructure stable enough to trust. And when benchmarks said we were ready but production said otherwise, the flywheel closed the gap in two weeks.

## When does this generalize?

This approach applies when:

1.  **The task requires tool calling.** The model must reason, act, and incorporate external results, not just generate text.
2.  **The output format is a custom DSL** that doesn't appear in pretraining data, and its semantics can be expressed in a language the model already knows.
3.  **A round-trip transpiler is feasible** between the in-distribution representation and the production format.
4.  **A production feedback loop is available.** Synthetic data gets you started, but real-world data is what gets you to production quality.

Within Sidekick, this pattern is already being applied to other skills. The recipe is the same: isolate the skill, fine-tune the tool-calling model, and build the loop for continuous improvement.

Six months ago, this system ran on a frontier model we didn't control. Now it runs on a model we trained, on infrastructure we own, improving from data only we have, at 68% lower cost. The version running right now is already worse than the one retraining behind it.

We started on rented ground. This is what the first mile of owned ground looks like.

---

This article contains contributions from Nicolas Bertagnolli, Joe Lin, Han Li, Mingyu Zhao, Jason Liu, LinKai Ma, Yuxuan Wang, Matt Koenig, Lingyun Wang, Agentic Foundation Modeling Team.
