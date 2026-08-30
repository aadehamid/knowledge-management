Adversarial review of TWO summary pages against their sources. REPORT-ONLY.

=== PAGE 1: fsdp-qlora-answerai-2024-full ===
---
type: summary
title: "FSDP+QLoRA: training a 70B model at home (Answer.AI)"
tags: [llm-finetuning, fsdp, qlora, quantization, memory]
sources:
  - id: doc-you-can-now-train-a-70b-language-m
    resource: ../../Raw/doc-you-can-now-train-a-70b-language-model-at-home-answerai.md
    title: "You can now train a 70b language model at home – Answer.AI"
updated: 2026-08-30
description: "the full story of combining FSDP and QLoRA: the engineering problems, the HQQ discovery, and the bitsandbytes bugs fixed along the way"
generated: { by: "agent:deepseek-v4-pro-0813", at: "2026-08-30T14:00:11Z" }
---

# FSDP+QLoRA: training a 70B model at home (Answer.AI)

Answer.AI's first release: FSDP + QLoRA, training a 70B model on two 24GB gaming GPUs
(RTX 3090/4090) — collaboration between Answer.AI, Tim Dettmers (U Washington), and
Hugging Face. The motivation framing: gaming GPUs "have similar performance to the data
center GPUs that cost over 10x more" but max out at 24GB; "all the big industry labs have
the 10x more expensive hardware already, so they don't really have the incentive."

## The two halves

- **QLoRA** (Dettmers): quantized base (4-bit, untrainable) + trainable fp16 LoRA adapters.
  Previously trained a 65B model (130GB unquantized) on a 48GB card. Limitation: 4-bit
  70b = 35GB, still larger than 24GB per card.
- **FSDP** (Meta): shards parameters across GPUs with all-gather/reduce-scatter, all GPUs
  active simultaneously (unlike `device_map='auto'` naive model parallelism, "only one GPU
  is ever active at a time... 7/8 of the compute is wasted").

## The engineering problems, each recorded because they recur

1. **FSDP syncs only PyTorch parameters/buffers**; quantization libraries store "quantization
   state" metadata in dictionaries — it gets lost when FSDP moves shards. Fix: move
   quantization state into the layers.
2. **FSDP only supports floating-point storage**; quantized weights are integers. Fix:
   selectable storage datatype matching the computation type.
3. **Loading a quantized model larger than one GPU** was impossible — loading/quantization
   itself required the whole model on one GPU. Fix: load and discretize one layer at a time.
4. **bitsandbytes CPU-offloading bug**: each time an offloaded weight was copied back to
   GPU it was **re-quantized, "effectively turning the pretrained model into random
   weights"**. Fixed by tracking which parameters were already quantized.
5. **bitsandbytes super-linear memory growth** with sequence length — "eventually resulting
   in even higher memory usage than without quantization". No bnb fix at publication; led
   to the HQQ discovery below.

## The HQQ discovery

Unsloth's Daniel Han pointed them to HQQ: bitsandbytes 4-bit assumes normally-distributed
parameters and is fast but less accurate; GPTQ/AWQ optimize quantization parameters against
calibration data (more accurate, "hours or even days" to process). **HQQ is "50x faster to
process a 70b model compared to GPTQ, yet is more accurate than it"** — and needed nearly
the same integration steps as bitsandbytes. FSDP+HQQ support merged within days.

## Extra techniques layered on

Gradient checkpointing (recompute rather than store activations), CPU offloading ("not
very useful to the GPU rich... but for our use case, it's absolutely necessary"), Flash
Attention 2.

Also stated: support landed upstream in Accelerate, Transformers, TRL, and PEFT; the code
was incorporated into Axolotl and used to train Mixtral.

Related: [QLoRA](../entities/QLoRA.md) · FSDP — the FSDP page lives in the Transformer from Scratch vault, outside this bundle


=== SOURCE 1: Answer.AI FSDP+QLoRA ===
---
url: "https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html"
title: You can now train a 70b language model at home – Answer.AI
source_type: doc
type: Source
fetch_status: ingested
fetched_at: 2026-08-29
wiki_refs:
  - Wiki/summaries/fsdp-qlora-answerai-2024-full.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/entities/QLoRA.md
  - Wiki/summaries/fsdp-qlora-answerai-2024-full.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/entities/QLoRA.md
  - Wiki/summaries/fsdp-qlora-answerai-2024-full.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/entities/QLoRA.md
  - Wiki/summaries/fsdp-qlora-answerai-2024-full.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/entities/QLoRA.md
---

title: You can now train a 70b language model at home – Answer.AI
description: We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.

# You can now train a 70b language model at home

## Summary

Today, we’re releasing Answer.AI’s first project: a fully open source system that, for the first time, can efficiently train a 70b large language model on a regular desktop computer with two or more standard gaming GPUs (RTX 3090 or 4090). This system, which combines FSDP and QLoRA, is the result of a collaboration between Answer.AI, Tim Dettmers (U Washington), and Hugging Face’s Titus von Koeller and Sourab Mangrulkar.

This system will help the open source community release better models. Teknium, the creator of the extremely popular OpenHermes models and datasets, with over half a million downloads, said:

> “*With this capability we can take huge models to new heights locally, and gigantic, hundreds of billions of parameter models are now accessible by small labs.*”

At Answer.AI we made this our first project because it’s a key foundation of our north star: helping make useful AI available to everyone. Just being able to use *other* people’s models is not enough. We want everyone to be able to create their *own* personalized models, so that they are in control of their own AI systems.

## Background

### The big idea

There are two very different levels of hardware used to train deep learning models. There is the data center class hardware, such as H100s and A100s, costing [hundreds of thousands of dollars](https://shop.lambdalabs.com/deep-learning/servers/blade/customize). Then there are desktop computers containing gaming GPUs, such as dual 4090s, costing [under $10,000](https://shop.lambdalabs.com/gpu-workstations/vector/customize) (and which can be assembled from 2nd hand parts for less than half the price of a pre-built system).

But here’s the key point: the gaming GPUs have similar performance to the data center GPUs that cost over 10x more! It would be great if we could use these 10x cheaper (but nearly as fast) cards to train large language models, but we can’t, because they have much less memory. The best currently available data center cards have 80GB RAM, whilst gaming cards max out at 24GB RAM. Since only the largest models produce the best results, creating the best models has been largely inaccessible to most people.

We realized that there’s actually no intrinsic reason for this. The super fast hardware is there, waiting to be used – we just need a way to feed it with the model and the data in a way that meets its memory constraints. The obvious question is: why hasn’t this been done then? All the big industry labs have the 10x more expensive hardware already, so they don’t really have the incentive to figure this out.

The big idea here is simple: figure out how to use these cheaper, lower-memory gaming GPUs to train the best available open source models. So the goal is this: train[^1] a 70 billion parameter (70b) model using only gaming GPUs, which means our per-GPU memory will be at most 24GB. It’ll be a challenge, because each parameter normally takes 16 bits (2 bytes), so that’s 70\*2\=140GB to even store the weights – and that’s without including all the other data such as activations, gradients, and optimization state!

### Why this project?

Answer.AI is a very unusual type of organization – a for-profit R&D lab closer in spirit to [19th century electricity labs](https://www.answer.ai/posts/2024-01-26-freaktakes-lessons.html) than to today’s AI research groups. Figuring out how to make large model training inexpensive and accessible is just the kind of thing Eric Ries and Jeremy Howard hoped we’d be able to do when the organization was [launched at NeurIPS](https://www.answer.ai/posts/2023-12-12-launch.html) last year.

Solving this problem is hard. It requires understanding many separate libraries (e.g bitsandbytes, PEFT, Transformers, Accelerate, and PyTorch), and computer science and math concepts (e.g discretization, distributed computing, GPU programming, linear algebra, SGD concepts such as gradient checkpointing), and how they all interact.

Academia is full of brilliant people that solve hard problems. But academia hasn’t solved this particular problem. That’s because it’s difficult for university researchers to justify spending time on this kind of work. Combining existing tools and techniques together isn’t generally considered “novel” enough to result in publication in a high impact journal, but that’s the currency that academics need. Furthermore, academics are generally expected to become highly specialized within their field, making it challenging to bring together so many pieces into a single solution.

And, of course, big tech companies are also full of brilliant people that solve hard problems. But this particular problem, training models with consumer GPUs, isn’t a problem they need to solve – they’ve already bought the big expensive GPUs! Many startups are also full of brilliant people that solve hard problems! But, as [Eric Ries explains](https://ltse.com/about/mission), “today’s financial market forces businesses to prioritize short-term gains over everything else”. It’s extremely hard for a startup to justify to investors why they’re spending their funds on open source software and public research.

Whilst academia, big tech, and startups had good reasons for not solving this problem, these are [the exact reasons](https://www.answer.ai/posts/2023-12-12-launch.html) that this problem was a great fit for Answer.AI. Everyone who works at the company has built the kinds of systems that we had to work with on this problem, so we were able to understand how all the pieces fit together. People who love to both deeply understand the foundations of software and AI, and also love to hack at fun and interesting end-to-end systems are the kinds of people who are drawn to Answer.AI, and vice versa.

The problems we choose to solve together are selected by the same people that will do the solving. So we tend to pick up projects that involve bringing together multiple ideas together to create practically useful solutions. And because we’re a public benefit company with a charter to produce *long term* benefit from AI, open source software and public research are directly in line with our mission.

### QLoRA: Train bigger models on a single GPU

Two projects have been released recently that took the first critical steps towards making this a reality: QLoRA (by [Tim Dettmers et al](https://arxiv.org/abs/2305.14314)), and FSDP (by Meta’s [PyTorch team](https://engineering.fb.com/2021/07/15/open-source/fsdp/)).

QLoRA is a simple but brilliant combination of two critically important advances in modern neural networks: *quantization*, and *LoRA*. Quantization is a technique where, instead of using 16 or even 32 bits to store the weights of a neural network, 4 (or even fewer) bits are used. There are only 16 possible values of a 4 bit number, but [Dettmers and Zettlemoyer showed](https://arxiv.org/abs/2212.09720) that this can be enough in the large language models that are popular today. Tim Dettmers made these 4-bit “quantized” models easy to create, thanks to his bitsandbytes library, and recently Hugging Face has stepped in to help [maintain and document](https://huggingface.co/docs/bitsandbytes/main/en/index) this library, particularly thanks to the initiative of Titus von Koeller.

Unfortunately, once a model is quantized, it can not be trained any further with regular approaches – with just 16 possible values, the gradient descent method used for model training will observe zero gradients nearly everywhere, so it can’t make any updates to the quantized weights. This is a major problem, because it means that quantization can only be used for inference, not for continued pre-training or fine-tuning. Whilst inference is useful and important, it’s really just *consuming* models. But we want everybody to be able to *contribute* to *creating* models!

The trick to avoiding this limitation is to use [LoRA](https://arxiv.org/abs/2106.09685) – “Low-Rank Adaptation of Large Language Models”. LoRA doesn’t train the whole large language model at all, but instead adds “adaptors”, which are very small matrices (generally smaller than 1% of the full model) that are trained, whilst keeping the rest of the model constant. If you’ve played with models like Stable Diffusion, you will have probably seen these adapters many times; it’s how those models are generally shared, and why they are so small and fast to download.

Tim realized that LoRA can be combined with quantization: use a quantized base model, which is not changed at all by the training, and add trainable LoRA adaptors that are not quantized. This combination is *QLoRA*. Tim’s team was able to use this to, for the first time, train a model that (unquantized) is larger than the GPU: they trained a 65b model (which is 130GB unquantized) on a 48GB card.

Hugging Face stepped in once again here, creating the [PEFT](https://huggingface.co/blog/peft) library, which made LoRA training far simpler, and also integrating it directly with bitsandbytes to allow anyone to use QLoRA with just a few lines of code. The Hugging Face team has been working tirelessly behind the scenes to ensure that the open source community can use these technologies to train their models. If you’ve ever used Transformers to load a 4-bit model using a single function argument, then you’ve got them to thank (and even if you haven’t, you’ve almost certainly used the work of folks that have built their model with this ecosystem).

QLoRA didn’t quite slay the problem we set out to solve, to train a 70b model on 24GB cards, but it got closer than anything before. When quantized to 4 bits (which is 0.5 bytes), the 70b model takes 70/2 \= 35 GB, which is larger than the 24GB gaming GPUs we want to use.

There are other limitations to QLoRA. A 48GB card is very expensive, and training a 65b model only just fits on such a card. That can be a problem, because we need to store lots of other things too, including the activations[^2], gradients, and optimization state of the model during training. If there’s not much memory left over after loading the model weights, there’s not enough working memory to support training.

For instance, one of the benefits of language models is that we can use them to “chat” with, or understand, or analyze long documents or conversations. To make models that can handle long sequences like that, we need to show them examples of long sequences during training. The longest sequence used in training is called the “sequence length”. Trying to use anything but a short sequence length will cause an error when training a 65b QLoRA model on a 48GB card, because there isn’t enough memory to store all the information about the sequence; nearly all the memory is used just to store the model itself.

Furthermore, if the model can only look at a single sequence at a time, it’s going to take a really long time to get through all the data in our training set. So instead we want to be able to “batch” a few sequences together at a time. The number of sequences included is the “batch size”. When there’s very little space left on the GPU after loading the model weights, we can only use very small batch sizes, resulting in extremely slow training.

### FSDP: Scale training to multiple GPUs

One obvious solution to the problem of the RAM limitations of a single consumer GPU, is to use more than one GPU! A very common approach in the open source community is to simply place a few layers of the model on each card. So then to train, you run the first few layers on the first GPU, then the next few on the second GPU, and so forth. For instance, a 70b (140GB) model could be spread over 8 24GB GPUs, using 17.5GB on each. There’s even a convenient setting in Hugging Face Transformers, `device_map=’auto’`, which you may well have used; that’s what this is actually doing behind the scenes. This does the job, but there’s a giant downside: only one GPU is ever active at a time, as all the others wait for their “turn”. That means that ⅞ of the compute is wasted.

*Distributed Data Parallel* (DDP) was previously the gold standard approach to training models across multiple GPUs efficiently. This requires keeping the full model on each GPU – if you have a small model (e.g. a 2b model, which takes 4GB RAM) you can simply load the whole thing onto each GPU separately, and have each GPU then churn through training examples in parallel. So for instance, if you had 4 GPUs, that’s a 4x training speedup. But DDP doesn’t work if the model doesn’t fit onto a GPU, with enough room to spare for the data needed for the training process.

So we need something that can split a model across GPUs (like `device_map=’auto’`) and also use them in parallel (like DPP). This is where Meta’s [Fully Sharded Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) (FSDP) library comes in. It “shards” a large model, by splitting its parameters across multiple GPUs, allowing all the GPUs to be used simultaneously. When a layer of the neural network is calculated on a particular GPU during training, all the required shards are copied there. Then, the calculation is made, and finally the copied parts are deleted from that GPU. Whilst this sounds terribly inefficient, actually by being smart about copying the data of the next layer at the same time the current layer is busy calculating, it’s possible for this approach to result in no slowdown compared to DDP.

FSDP’s ability to bring the performance of DDP to models that are larger than any one GPU has been a revelation. For instance, a 70b (70 billion parameter) unquantized model takes 140GB of RAM (because each parameter is stored as 16 bits, which is 2 bytes), but even NVIDIA’s H100 card (which costs around $40,000 for a single card!) falls short of what’s needed, with its 80GB RAM. But with FSDP, four H100 GPUs can be combined for a total of 320GB RAM.

(Mind you, such a machine is going to set you back around $150,000…)

## Bringing FSDP and QLoRA together

At Answer.AI our north star is making useful AI more accessible. $150,000 to create your own high-quality personalized model definitely doesn’t count as accessible! So the first project we embarked on was to make it possible to use a desktop with consumer gaming GPUs to efficiently train a 70b model. We figured that if we could use QLoRA to reduce the size of a model by around 400% (so a 70b model would fit into 35GB RAM), and then we used FSDP to shard that across two or more 24GB consumer cards, that would leave enough RAM left over to train a model.

### First steps

Jeremy and Tim in late 2023 discussed the idea of bringing FSDP and QLoRA together. Tim connected Jeremy with Titus von Koeller, and Jeremy and Titus worked together to try, explore, understand, and document the issues that occurred when the two libraries were combined.

Answer.AI’s Johno Whitaker put together an important first step: a simple standalone test script which allowed us to more deeply understand the problem, and test solutions. A key breakthrough came in early 2024 when Answer.AI’s Benjamin Warner and Titus independently came up with a key idea: store the quantized parameters in a selectable data type, where that storage data type is the same data type as the “computation type” of the model[^3].

Benjamin had this prototyped within 24 hours of developing the idea, but then we discovered another problem: FSDP was not copying the quantization information needed for each shard to use the model! That’s because FSDP is quite opinionated on the subset of data it will sync between GPUs[^4]. We realized that if we quantized the model on each GPU the missing metadata would remain untouched on all GPUs. Furthermore, we had to move the “quantization state” (the information necessary to (de)quantize the parameters) from the parameters into the layers, in order to ensure they were not removed when FSDP moved shards.

Once we had those issues resolved, we were able to successfully train our first batch of data with a quantized model using FSDP! Benjamin and Answer.AI’s Kerem Turgutlu were able to package this up with all the tests and refactoring needed into a [pull request](https://github.com/TimDettmers/bitsandbytes/pull/970) for bitsandbytes. We’re extremely grateful to the maintainers of the bitsandbytes project, who were very responsive in shepherding our PR through their processes.

### Mission accomplished, nearly

At this point, we once again figured that we’d have things tied up pretty quickly, and once again we under-estimated the complexity of the task! The first thing we realized was that it still wasn’t actually possible to load a quantized model that’s larger than a single GPU, since the loading and quantization process itself required the whole model to be put on one GPU.

Jeremy had spent a few weeks carefully studying Meta’s fantastic [Llama-Recipes](https://github.com/facebookresearch/llama-recipes) project, which was the best complete FSDP fine tuning implementation he had found, and by closely tracking how it worked together with bitsandbytes, along with Hugging Face’s PEFT, Transformers, and Accelerate projects, he managed to construct a minimal standalone script which manually complete all the steps needed to fine tune a model.

Benjamin realized that with some tweaks it would be possible to do the loading and discretization one layer at a time, thus avoiding the need to ever have the whole model on a single GPU. He also figured out how to prevent the PEFT library from moving the quantization state to CPU. Kerem wrote a custom implementation of the LoRA algorithm so that it could work with Benjamin’s changes.

And with that, we were able to fine tune a 70b model on dual 3090 gaming GPUs for the first time!

To make this work, we were benefiting not just from FSDP and QLoRA, but also from a vast array of clever techniques developed over the last couple of years by the academic and open source communities. We used:

- [Gradient checkpointing](https://arxiv.org/abs/1604.06174) (also known as activation checkpointing) to avoid storing full gradients, instead saving activations at a number of ‘checkpoints’ throughout the model and then re-computing gradients by re-running the forward computation step as needed
- [CPU offloading](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload) to store weights in CPU RAM rather than on the GPU when they are not in use, drastically reducing the GPU memory required. This technique isn’t very useful to the “GPU rich” using H100 GPUs, which have highly optimized ways to pass weights to each other. But for our use case, it’s absolutely necessary, since gaming GPUs and motherboards don’t have these systems
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) to efficiently calculate attention using a memory-optimized Cuda kernel.

Making these work together with FSDP and QLoRA wasn’t always straightforward. For instance, when using CPU offloading, Benjamin discovered and fixed a problem in bitsandbytes. Each time an “offloaded” weight was copied back to the GPU, it was being automatically re-quantized, which effectively turning the pretrained model into random weights! We made a pull request to bitsandbytes that kept track of which parameters had already been quantized, so we could avoid the redundant computation.

After all this work, we were very pleased to see that we could train large models with consumer GPUs. Jeremy had run detailed benchmarking of the original llama-recipes across a range of GPU hardware, and Kerem developed a comprehensive benchmarking system for the new project. In comparing the two we realized that we were still not able to use the sequence lengths or batch sizes we’d hoped – for some reason more memory was being used than we expected.

When we looked closely, it turned out that it wasn’t due to our FSDP/QLoRA integration at all – but actually as we increased seqlen in bitsandbytes even without FSDP, the memory usage went up super-linearly, eventually resulting in even higher memory usage than without quantization! It turns out that we’re [not the first](https://github.com/RahulSChand/gpu_poor/issues/1) people to discover this problem. We don’t have a bitsandbytes solution yet (but it’s being investigated), but it did lead us to an exciting discovery…

### Discovering HQQ

We love collaborating with like minded people, so when we saw the amazing work being done by Daniel Han on [Unsloth](https://unsloth.ai/) we wanted to learn more, and see whether we might be able to help each other out. We asked Daniel if there were other interesting projects in this space that we should be watching, and he pointed us to [HQQ](https://mobiusml.github.io/hqq_blog/).

To explain HQQ, we’ll need to give a bit of background first… The 4-bit quantization done by bitsandbytes uses a simple, fast, and clever approach where each group of parameters is normalized to a consistent range, and then each parameter is placed in a bucket, where the bucket breakpoints are based on an assumption that the parameters are normally distributed. This has the benefit that quantization is nearly instant, but because real model parameters will not exactly fit the assumed distribution, accuracy can suffer.

Other approaches such as GPTQ and the more recent AWQ go in a different direction, where the quantization parameters are optimized based on the actual behavior of the model when representative data is passed to it.[^5] These methods tend to produce more accurate models, potentially with even fewer than 4 bits per parameter; but they have the downside that the optimization process can take hours or even days for each model.

HQQ combines the best of both worlds. It is 50x faster to process a 70b model compared to GPTQ, yet is more accurate than it. Kerem decided to investigate whether HQQ would work well with FSDP.

He discovered that making HQQ and FSDP work well together took nearly the exact same steps as was required for bitsandbytes, and as result he had a complete working example completed within days. The [mobius.ml folks](https://www.mobiuslabs.com/) couldn’t have been more responsive and helpful in ensuring that our PR was successfully merged – so we are now delighted to announce that FSDP works with HQQ too!

## How to use FSDP/QLoRA

To use FSDP, you will of course need more than one GPU. If you don’t have access to such a system, you can rent a dual 3090 box from the [Runpod Community Cloud](https://www.runpod.io/) for around $0.60/hour. There are many other providers to choose from; [cloud-gpus](https://cloud-gpus.com/) is a great place to see what’s on offer.

You’ll need to install the latest version of Transformers, PEFT, and bitsandbytes (and HQQ if you’re using that). Then, clone [our repo](https://github.com/AnswerDotAI/fsdp_qlora/tree/main) and follow the README there. Running `python train.py --help` will show the available options. To train llama2-7b on the included alpaca dataset on two 24GB cards you might run:

> `python train.py --train_type qlora --dataset alpaca --batch_size 8 --gradient_accumulation_steps 2 --output_dir qlora_output --log_to wandb`

We’ve crammed everything required into this single file to make it easier to see what is going on and modify things if necessary.

You should treat this script as an alpha/preview release. Whilst we’ve used it to successfully train a variety of practically useful models on a range of hardware, it’s still early days. If you’re not comfortable with testing and debugging models, we’d suggest holding off for a few months whilst the community more fully tests the approach.

**Update:** We’re excited to see that support is [already underway in the Hugging Face ecosystem](https://huggingface.co/docs/peft/main/en/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus) via changes to [Accelerate](https://github.com/huggingface/accelerate/pull/2544), [Transformers](https://github.com/huggingface/transformers/pull/29587), [TRL](https://github.com/huggingface/trl/pull/1416) and [PEFT](https://github.com/huggingface/peft/pull/1550). Our code has also been incorporated into the Axolotl finetuning library and [used to train Mixtral](https://twitter.com/winglian/status/1766192708102562222) and other models.

## A first step

Originally we’d planned to show some benchmarks in this article, along with guidance based on the benchmark results about how to best take advantage of FSDP/QLoRA. However, we had to keep delaying posting this, because we kept making major improvements every few days! So we’ll follow up with a benchmarking and recommendations article in the coming weeks.

What we’ve shown here is just the first step. We have lots of ideas for improvements, and we’re sure that the open source community will have many other ideas we haven’t even thought of yet!

We hope that seeing this proof of concept, that it’s possible to scale resource-efficient QLoRA training across inexpensive gaming GPUs, will help bring more attention to the problem of bringing down the cost of model training. It’s in everyone’s interest to make AI more accessible – and to enable more people to not only consume, but also build, valuable models.

[^1]: Throughout this article “training” can refer to either pre-training, or fine-tuning.

[^2]: Strictly speaking, we don’t generally store the actual activations, but rather just the intermediate pieces needed to recalculate them on demand by using gradient checkpointing.

[^3]: FSDP only supports floating point types but most quantization libraries store quantized weights in integer types. A selectable storage datatype resolved this discrepancy.

[^4]: FSDP only sypports syncing PyTorch parameters and buffers, while most quantization libraries store “quantization state” metadata in dictionaries.

[^5]: This is very important as calibration data bias is another major issue one can face using these data dependent methods.


=== PAGE 2: distributed-training-guide ===
---
type: summary
title: Distributed training guide (DeepSpeed ZeRO, FSDP)
tags: [llm-finetuning, distributed-training, deepspeed, fsdp, memory]
sources:
  - id: doc-everything-about-distributed-train
    resource: ../../Raw/doc-everything-about-distributed-training-and-efficient-finetuning-sumanths-personal.md
    title: "Everything about Distributed Training and Efficient Finetuning - Sumanth's Personal Website"
  - id: doc-you-can-now-train-a-70b-language-m
    resource: ../../Raw/doc-you-can-now-train-a-70b-language-model-at-home-answerai.md
    title: "You can now train a 70b language model at home – Answer.AI"
updated: 2026-08-30
description: the ZeRO stage ladder, FSDP sharding, the parallelism zoo, and the practical optimization ordering
generated: { by: "agent:deepseek-v4-pro-0813", at: "2026-08-30T14:00:11Z" }
---

# Distributed training guide (DeepSpeed ZeRO, FSDP)

Sumanth Hegde's synthesis: DeepSpeed ZeRO, FSDP, and efficient fine-tuning optimizations,
aimed at "upping your game from just using a home server with a couple of 3090s to a GCP
container with 8xA100 80GBs."

## The memory math that motivates everything

AdamW needs **12 bytes per parameter** (weights + momentum + variance, mixed precision) —
so Falcon 40B is ~74GB for weights alone in BF16, and ~480GB of training state.

## The ZeRO ladder (all data parallelism; communication costs vs DDP's 2Ψ)

| Stage | Shards | Memory | Communication |
|---|---|---|---|
| Baseline DDP | nothing | full copies | 2Ψ |
| ZeRO-1 | optimizer state | 4x reduction (example) | same as DDP |
| ZeRO-2 | optimizer + gradients | 8x | same as DDP |
| ZeRO-3 | + parameters | N-way reduction | 1.5x DDP (extra all-gather) |

ZeRO-1/2 still require the entire model on one GPU; ZeRO-3 removes that limit ("as long as
there are sufficient number of devices"). Caveat: ZeRO-1/2 RAM usage during model
initialization grows huge at 40B+; ZeRO-3 fixes it.

Extensions: **ZeRO-R** (activation partitioning), **ZeRO-Offload** (optimizer work to CPU,
V100-era 40 TFLOPS for 10B), **ZeRO-Infinity** (+ NVMe offload, 10-100T+ parameters on one
DGX-2 node), **ZeRO++** (int8 weight quantization for all-gather, hierarchical partitioning
hpZ, int4 gradient quantization — 4x less communication than ZeRO-3).

## FSDP

Same idea as ZeRO-3 (sharding strategies from the same paper lineage): all-gather to
collect shards for a layer, compute, discard. **Hybrid sharding** = shard within nodes,
replicate across nodes ("similar to hierarchical partitioning in ZeRO++") — attractive
multi-node when ZeRO-2/1 aren't options.

His claim on durability: ZeRO/FSDP "will likely be replaced only by a strategy that is
very similar" — the ease-of-use (no architecture changes, no forward-pass modifications)
is the point; PP/TP require model changes and are messy in HuggingFace.

## The efficient-finetuning toolbox

Mixed precision (BF16 preferred — no loss scaling needed vs FP16), PEFT (LoRA over all
linear layers can match FFT per QLoRA; his IA3 experiments on GPT-2 matched LoRA with <1/10
the parameters, "needs more community experimentation"), Flash Attention 2 (220+ TFLOPS on
A100; exact, not approximate), gradient checkpointing (~20% slowdown, activations O(N) →
O(√N)), quantization (QLoRA "enabl[es] finetuning of 60B+ parameter models on a single GPU
with 48GB vRAM" at much worse throughput from the de-quantization step), gradient
accumulation.

## His practical ordering (the citable part)

1. BF16/FP16 by default (BF16 "without any overflow issues")
2. LoRA on all linear layers
3. Flash Attention if the GPU supports it
4. Gradient checkpointing (may be unnecessary with Flash Attention, per Tri Dao)
5. Multipack sampler in the dataloader
6. Multi-GPU: "BF16 + LoRA + Gradient Checkpointing + DeepSpeed ZeRO 3 first"
7. Quantization only under very limited memory (QLoRA works with ZeRO 1/2 only)
8. 8+ GPUs: ZeRO-3 (ZeRO-2 hits CPU RAM limits at 40B+ — Falcon-40B needs >1.5TB CPU RAM
   for initialization)
9. Multi-node: ZeRO-3 with hierarchical partitioning, or FSDP hybrid sharding
10. CPU/disk offloading as the last resort (ZeRO-Infinity > ZeRO-Offload)
11. Scale learning rate with effective batch size (fine-tuning; pretraining is the opposite
    per GPT-3/BLOOM papers)

Also noted: monitor `htop` for RAM OOM and `nvidia-smi` for data-preprocessing bottlenecks.

The FSDP+QLoRA combination this guide references is documented from Answer.AI's perspective in [FSDP+QLoRA full](fsdp-qlora-answerai-2024-full.md).

Related: [FSDP+QLoRA](../summaries/fsdp-qlora-answerai-2024-full.md) · [QLoRA](../entities/QLoRA.md) · [Unsloth MoE kernels](../summaries/unsloth-moe-kernels.md)


=== SOURCE 2: distributed training guide ===
---
url: "https://sumanthrh.com/post/distributed-and-efficient-finetuning/"
title: "Everything about Distributed Training and Efficient Finetuning - Sumanth's Personal Website"
source_type: doc
type: Source
fetch_status: ingested
fetched_at: 2026-08-29
wiki_refs:
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/summaries/distributed-training-guide.md
  - Wiki/summaries/distributed-training-guide.md
---

title: Everything about Distributed Training and Efficient Finetuning | Sumanth's Personal Website
description: A deep dive into distributed training and efficient finetuning - DeepSpeed ZeRO, FSDP, practical guidelines and gotchas with multi-GPU and multi-node training
author: Sumanth R Hegde

There’s been an insane amount of interest in large language models (LLMs) these days, with a very special open source community of hackers figuring out the best way to finetune, serve and run inference on consumer-grade hardware. A number of excellent open-source codebases have popped up to meet these needs, notably [FastChat](https://github.com/lm-sys/FastChat/), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) and [LLama.cpp](https://github.com/ggerganov/llama.cpp), with the 🤗HuggingFace ecosystem being at the center of it all. I wanted to write this post to focus on the nitty gritty details of distributed training strategies, specifically DeepSpeed and FSDP, along with a summary of different efficient finetuning methods, with special focus on multi-GPU and multi-node training. The trend right now is clear: We’re going to be using more and more compute, and thus more GPUs with bigger models. So, understanding these topics is important in this context, especially so when you’re trying to up your game from just using a home server with a couple of 3090s, to, say, a GCP container with 8xA100 80GBs. This is also relevant for startups/ companies who are trying to get into fine-tuning their own language models. For actual large scale training taken up by the big tech companies, there’s plenty of material, mostly from [Stas Bekman](https://github.com/stas00/ml-engineering), who led the training for BLOOM-176B, and there’s very little use for [GPU-poors](https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini) in focusing on that. This is also more of a *synthesis* of ideas from different excellent resources out there already, with the main focus on what you can do in the 🤗HuggingFace ecosystem, along with practical considerations, mostly from a number of sources online (it’s not pretty) and some learnt from my internship in Summer 2023. In short, I hope to answer the following questions:

1. What do we care about with distributed training and performance? What happens under the hood with DeepSpeed and FSDP?
2. What hardware setup do I need for different distributed training strategies? What are the caveats?
3. What are the various efficient finetuning optimizations? What are the tradeoffs?
4. What are some practical guidelines that can capture all the salient finetuning optimizations, in order to train large models in a multi-GPU and multi-node setting?
5. What open-source codebases can I use right now? What are the pros and cons?

# Table of Contents {#table-of-contents}

1. [Distributed Training Basics](#distributed-training)
2. [ZeRO-powered Data-Parallelism](#zero-powered-data-parallelism)
    1. [Baseline](#baseline)
    2. [ZeRO Stage 1](#zero1)
    3. [ZeRO Stage 2](#zero2)
    4. [ZeRO Stage 3](#zero3)
    5. [ZeRO-R](#zero-r)
    6. [ZeRO-Offload](#zero-offload)
    7. [ZeRO-Infinity](#zero-infinity)
    8. [ZeRO\+\+](#zero)
3. [Fully-Sharded Data Parallel](#fully-sharded-data-parallel)
    1. [Full Sharding](#full-sharding)
    2. [Hybrid Sharding](#hybrid-sharding)
4. [Implementations](#implementations)
    1. [How can you use DeepSpeed and FSDP?](#how-can-you-use-deepspeed-and-fsdp)
    2. [What about Pipeline and Tensor Parallelism?](#what-about-pipeline-parallelism-and-tensor-parallelism)
    3. [Are DeepSpeed and FSDP Here to Stay?](#are-deepspeed-zero-and-fsdp-here-to-stay)
5. [Efficient Finetuning](#efficient-finetuning)
    1. [Mixed Precision Training](#mixed-precision)
    2. [Parameter-Efficient Fine Tuning](#parameter-efficient-fine-tuning-peft)
    3. [Flash Attention](#flash-attention)
    4. [Gradient/Activation Checkpointing](#gradient-activation-checkpointing)
    5. [Quantization](#quantization)
    6. [Gradient Accumulation](#gradient-accumulation)
    7. [So wait, should I always try to increase batch size?](#so-wait-should-i-always-try-to-increase-batch-size)
6. [Practical Guidelines](#practical-guidelines)
    1. [The Ultimate Summary](#the-ultimate-summary)
    2. [Additional Guidelines](#additional-guidelines)
7. [More on DeepSpeed and FSDP](#more-on-deepspeed-and-fsdp)
    1. [Multi-node with DeepSpeed ZeRO-3](#multi-node-with-deepspeed-zero-3)
    2. [DeepSpeed Memory Requirements](#deepspeed-memory-requirements)
    3. [Usage with 🤗Accelerate](#usage-with-%f0%9f%a4%97accelerate)
8. [Open-Source Codebases](#open-source-codebases)
    1. [FastChat](#fastchat)
    2. [Axolotl](#axolotl)
    3. [Useful Fine-tuning Guides](#useful-fine-tuning-guides)
9. [The End](#the-end)

# Distributed Training {#distributed-training}

This is a very broad topic with much talked about, so I won’t cover everything. When it comes to training/finetuning LLMs, typically you’re dealing with large model sizes (10B\+) and large dataset sizes (1T\+ tokens while pretraining, 1M\+ in supervised fine-tuning). Our ultimate goal in getting done with training as fast as possible is to maximize *throughput*, i.e we want to be able to process as many samples per second as we can. LLMs require a LOT of GPU vRAM to train, not just because of the large model weights (Falcon 40B, with 40B parameters, needs around 74GB just for model weights in BF16), but also because of optimizer states - with vanilla AdamW, you need [12 bytes per parameter](https://blog.eleuther.ai/transformer-math/) to store a copy of the model weights, the momentum and the variance parameters. This is where we need smart distributed training strategies, where each GPU worker only deals with a fraction of training state and data.

The main parallelism strategies are:

1. Data Parallelism(DP): Each GPU worker gets a fraction of the total mini-batch of data, and computes the gradients on that fraction of the data. The gradients are then averaged across all workers, and the model weights are updated. In it’s most basic form, like the one in PyTorch DDP, each GPU stores a copy of the model weights, optimizer state and gradients for the fraction of the data it’s working on.
2. Model Parallelism/ Vertical Model Parallelism (MP): In model parallelism, models are *vertically sliced*, with different layers of the model placed on different GPU workers. Consider the case where a single model with 12 layers is placed on 3 GPUs.

```

---------------  ---------------  -----------------
1 | 2 | 3 | 4 |  5 | 6 | 7 | 8 |  9 | 10 | 11 | 12 |
---------------  ---------------  -----------------
```

An improvement is Pipeline Parallelism (PP), which gives you the illusion of parallelism by overlapping computation for different micro-batches of data. This is just like the classic pipeline in computer architecture. From the [GPipe](https://blog.research.google/2019/03/introducing-gpipe-open-source-library.html) blog:

> To enable efficient training across multiple accelerators, GPipe partitions a model across different accelerators and automatically splits a mini-batch of training examples into smaller micro-batches. By pipelining the execution across micro-batches, accelerators can operate in parallel.

![Gpipe](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/gpipe_hu5259366ef2017461648ee030acaf09a9_92604_e2a94cd2b52af3654d7313c0901250a3.webp){width=760 height=464}

3. Tensor Parallelism (TP): In tensor parallelism, each GPU processes only a slice of a tensor by *horizontally slicing* the model across GPU workers. Each worker processes the *same batch* of data, computing the activations for the part of the weights they have, exchanging parts that each other needs, with each worker computing the gradients for the slice of the weights it has.

You can have a combination of the various parallelism strategies above for even better throughput gains. That said, let’s take a closer look at two improvements for data-parallel training: Zero Redundancy Optimizer and the closely related Fully Sharded Data-Parallel strategies.

A comment: I’ll be loosely using the term “GPU worker” to refer to individual processes that run on each GPU. It’s not very precise, but for a DP setting I find it convenient and more approachable.

*Further reading* :

- Efficient Training on multiple GPUs: [https://huggingface.co/docs/transformers/perf\_train\_gpu\_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)
- How to Train Really Large Models on Many GPUs?: [https://lilianweng.github.io/posts/2021-09-25-train-large/](https://lilianweng.github.io/posts/2021-09-25-train-large/)

# ZeRO-powered Data-Parallelism {#zero-powered-data-parallelism}

This is one of the most efficient and popular strategies for distributed training at the moment. DeepSpeed’s [ZeRO](https://arxiv.org/abs/1910.02054), or Zero Redundancy Optimizer, is a form of *data parallelism* that massively improves on memory efficiency. The main idea is that the ZeRO exploits memory redundancy in data-parellel training and the latest improvements in fast inter-GPU communication to improve throughput, with some increase in communication volume, depending on the stage. ZeRO actually has two components - ZeRO-DP (data pallelelism) and ZeRO-R (residual memory). The DeepSpeed team has also put forth a number of followup optimizations that make ZeRO even more compelling - ZeRO-Offload/Infinity (offloading computation to CPU/ NVMe disk) and ZeRO\+\+ (with flexible multi-node training and quantized weights).

ZeRO-DP can be best visualized in this diagram (from [DeepSpeed’s blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)):![Different Stages of ZeRO](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/zero_huadd67ed54d9443c1208b20053d79cf40_169124_471cf0dc7dafc83066586bb6825317a0.webp){width=760 height=313}

The different methods, while training a 7.5B parameter model on 64 GPUs, have the following performance:

1. **Baseline**: PyTorch DDP
2. **ZeRO Stage 1/ $P\_{os}$** : 4x memory reduction (example specific) with the same communication volume as the Baseline (no additional inter-GPU communication)
3. **ZeRO Stage 2/ $P\_{os\+g}$** : 8x memory reduction (example specific) with the same communication volume as the Baseline.
4. **ZeRO Stage 3/ $P\_{os\+g}$**: 64x memory reduction (example specific), with 1.5x communication volume as the Baseline (this 1.5x is across different hardware setups and model sizes).

## Baseline {#baseline}

Simple data parallelism as implemented in PyTorch DDP. Each GPU worker has a copy of the model weights, optimizer state and gradients. After a backward pass, the gradients are averaged across all workers (the all-reduce step), and the model weights are updated.

*Comment on communication volume*: For understanding the benefits of ZeRO, I think it’s important to understand what communication volume means exactly. In typical DP, you have an [all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) step where all the workers send out the gradient array that they have, and then receive gradient arrays from the other workers to get the average value. From the ZeRO paper:

> State-of-art implementation of all-reduce uses a two-step approach, where the first step is a reduce-scatter operation, which reduces different part of the data on different process. The next step is an all-gather operation where each process gathers the reduced data on all the process. The result of these two steps is an all-reduce. Both reduce-scatter and all-gather are implemented using a pipelined approach, that results in a total data movement of Ψ elements (for a data with Ψ elements) for each. Therefore, the standard DP incurs 2Ψ data movement during each training step.

In this case, “data” is our gradients, and process refers to individual worker processes running on each GPU. All in all, what I want to convey here is that, if you have $\Psi$ parameters, you incur a communication cost of $2\Psi$ with plain DP.

##  ZeRO Stage 1 /$P\_{os}$ (Optimizer State Partioning) {#a-namezero1a-zero-stage-1-p_os-optimizer-state-partioning}

Here, only the optimizer state is partitioned/ sharded across GPU workers, with model weights and gradients replicated across all workers. After a backward pass, you have a regular all-reduce step to get the average gradient value across all workers. Now, each worker updates the optimizer state in it’s partition. Recall the Adam equations:

![Adam](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/adam_huac431a47f6c52fc1330aa63b76add502_16794_1f03b6c6fc7bb3abc459d1e830ffc7d8.webp){width=443 height=139}

![Source](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/adam2_hu1fc459b19b8966306ca16fdf1d6e68eb_23061_77d4b9d612fc287e0378908bd569f482.webp "Source"){width=360 height=128}

($w$, $g$, $v$ and $m$ correspond to weight, gradient, velocity and momentum respectively). Notice that these are all element-wise operations, with no dependency across different slices of weights (after computing gradients, that is).

*Communication volume*: Okay, so you might say, you first have an “all-reduce” operation that communicates the updated gradients to all the GPUs and then, after updating optimizer state for their partition, each GPU must still get the updated weights from the other GPUs. Doesn’t this increase communication volume? This detail isn’t clear even in the ZeRO paper! From my understanding, this implementation is in fact the same for both ZeRO Stage 1 and Stage 2:

- All-reduce for gradients consists of two components - [reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#reducescatter) and [all-gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allgather).
- With both DeepSpeed ZeRO 1 and 2, you have the usual reduce-scatter, where different parts of the gradient are reduced on different GPUs. After this, each GPU will compute the updated weights corresponding to the optimizer partition it has, and then, you only need one all-gather operation, this time to communicate the updated model parameters to all other GPUs. Thus you have a reduction in memory consumption at the same communication volume (which is to say, for free)!

##  ZeRO Stage 2 / $P\_{os\+g}$ (Optimizer State \+ Gradient Partitioning) {#a-namezero2a--zero-stage-2--p_osg-optimizer-state--gradient-partitioning}

Here, both the optimizer state and the gradients are partitioned/sharded across workers. This means that, not only are two GPU workers looking at a different micro-batch of data during training, they are also maintaining gradients for a subset of model parameters. The key insight here is that each worker is updating its partition of optimizer state, and thus the gradients (or rather, the reduced/averaged gradients) needed for a worker is simply the one corresponding to that state partition. Coming to the implementation, as mentioned above, DeepSpeed effectively performs a reduce-scatter operation, where gradients corresponding to different workers are averaged at that worker (instead of the typical all-reduce for all parameters). This means that you now have even more memory savings, again at the same communication volume - that is, there are no extra costs with respect to data movement compared to DDP!

**Note**: With Both ZeRO Stage 1 and 2, you still need the entire model to fit on 1 GPU. There are also caveats with RAM usage - model initialization takes up a huge amount of RAM as you increase number of processes/ GPUs and go to large model sizes (40B\+). ZeRO 3 improves on this.

##  ZeRO Stage 3 / $P\_{os\+g\+p}$ (Optimizer State \+ Gradient \+ Parameter Partitioning) {#a-namezero3a--zero-stage-3--p_osgp-optimizer-state--gradient--parameter-partitioning}

This to me is the most interesting ZeRO stage. Stage 3 partitions the model parameters across workers, in addition to optimizer state and gradients. A helpful visualization from DeepSpeed:

![Visualization for different training states with 4 GPUs](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/zero3_hu10d2649002074154929bd913d53be750_244157_7a732c0570a6411191a4dd4e19fd5512.webp "Visualization for different training states with 4 GPUs"){width=760 height=316}

![Training states being sharded with DeepSpeed ZeRO-3](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/zero32_hu7aa6fee1bee98a9a6f27be487ebacdef_188133_4ad2663f2c6ccaadd253e92cf4bcc0e2.webp "Training states being sharded with DeepSpeed ZeRO-3"){width=687 height=365}

To borrow an example from [Stas Bekman’s guide](https://github.com/stas00/ml-engineering/tree/master/model-parallelism), suppose you have the following 3-layer model and 4 GPUs:

```fallback
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
a3 | b3 | c3
```

With DeepSpeed ZeRO3, your GPUs will be populated as follows:

```fallback
GPU 0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU 1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU 2:
La | Lb | Lc
---|----|---
a2 | b2 | c2

GPU 3:
La | Lb | Lc
---|----|---
a3 | b3 | c3
```

Each layer of the model is *horizontally sliced*, with each worker storing a part of the weight tensors. During the forward and backward pass (recall that each GPU worker is still seeing a different micro-batch of data), different GPU workers exchange the parts of the each layer that they have (*parameter communication on-demand*), and compute activations/ gradients. The rest is similar to ZeRO Stage 2. The communication volume for ZeRO-3 can be easily seen to be 1.5x times baseline DDP: for every training step, you have an extra all-gather operation for model parameters in the forward pass. The amount of data being moved in this operation is again $\Psi$ (per GPU), and thus total communication volume is $\Psi$ (parameter all-gather) \+ $\Psi$ (gradients reduce-scatter) \+ $\Psi$ (all-gather, for updated parameters) \= $3\Psi$ \= 1.5x DDP. This is pretty impressive considering that the memory consumption has been cut down by the number of GPU workers $N$. Another key observation from the ZeRO paper:

> ZeRO powers DP to fit models with arbitrary size as long as there are sufficient number of devices to share the model states.

That is, you are now no longer limited by the per-GPU vRAM for data parallel (DP) training, as long as you have enough GPUs (easier said than done, I know).

## ZeRO-R {#zero-r}

I won’t go much into this, but ZeRO-R improves on ZeRO-DP by focusing on memory consumption by activations, and managing memory fragmentation. ZeRo-R reduces activation memory footprint by partitioning the activations as well. It also makes some more improvements in managing temporary buffers, which you can think of as memory allocated for storing intermediate results during gradient accumulation and reduction across workers.

## ZeRO-Offload {#zero-offload}

ZeRO-Offload is an optimization technique that can offload optimizer and computation from GPUs to the host CPU. At the time it was released in January 2021, ZeRO-Offload could achieve 40 TFLOPS on 1 NVIDIA V100 GPU (V100 32 GB vRAM, with a maximum throughput of 130 TFLOPS) for a 10B parameter model. With Pytorch DDP, the maximum is 30 TFLOPS, which you get can with a 1.4B parameter model, which is the largest you can run without running out of memory. The main problem with offloading computation to the CPU is that this is multiple orders of magnitude slower than GPU (based on throughput). The smart strategy adopted in ZeRO-Offload is that only the less intensive computations ( $\< O(MB)$, where $M$ is the model size and $B$ is the batch size) are offloaded to CPU so that the total compute complexity stays the same ($O(MB)$). In practice, this means that operations like norm calculations, weight updates, etc can be done on CPU, while forward and backward pass matrix mults need to be done on GPU. ZeRO-Offload works with all stages of ZeRO (1, 2 and 3).

[Here’s](https://docs.it4i.cz/dgx2/introduction/) the specification for the DGX-2 node used in their experiments, which has 16 V100 GPUs. Note that if you’re in the ZeRO-2 setting, ZeRO-Offload will still be limited by the available per-GPU memory i.e fitting the entire model on each GPU can be the bottleneck.

## ZeRO-Infinity {#zero-infinity}

ZeRO-Infinity is an improvement over ZeRO-Offload which came up in April 2021, by allowing offloading to disk (NVMe memory), and making some improvements to CPU offloading. ZeRO-Infinity was shown to fit models with 10-100T\+ parameters (Trillion!) for training on just one DGX-2 node. ZeRO-Infinity does this by exploiting CPU and NVMe memory simultaneously. Here’s a visualization from the paper:

![A Snapshot of ZeRO Infinity for 4 data-parallel ranks (GPUs). The figure depicts the state during a backward pass. The Partitioned/sharded parameters are moved from slow memory (CPU+ NVMe) to GPU and then collected to form the full layer. After gradients are computed, they are aggregated, re-partitoned, and then offloaded to slow memory.](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/zero_inf_hu55f9609ca49eb5602aa98c4578684cbb_77411_116d611594cca8137e70f19d3d18b4a2.webp "A Snapshot of ZeRO Infinity for 4 data-parallel ranks (GPUs). The figure depicts the state during a backward pass. The Partitioned/sharded parameters are moved from slow memory (CPU+ NVMe) to GPU and then collected to form the full layer. After gradients are computed, they are aggregated, re-partitoned, and then offloaded to slow memory."){width=755 height=486}

Unlike ZeRO-Offload, ZeRO-Infinity is specifically built on top of ZeRO-3. In their evaluations of model speed on 512 GPUs across 32 DGX-2 nodes, the authors showed that ZeRO-Infinity trains up to 20 trillion parameter models with throughput of up to 49 TFlops/GPU, while using alternative parallelism strategies like 3D parallelism only allows you to train 40x smaller models. There are some bandwidth requirements for ZeRO-Infinity to be a competitive choice, namely for NVMe-CPU and CPU-GPU communication.

In terms of the differences between ZeRO-Offload and ZeRO-Infinity, here is a comment from the DeepSpeed team:

> DeepSpeed first included offloading capabilities with ZeRO-Offload, a system for offloading optimizer and gradient states to CPU memory within ZeRO-2. ZeRO-Infinity is the next generation of offloading capabilities accessible to ZeRO-3. ZeRO-Infinity is able to offload more data than ZeRO-Offload and has more effective bandwidth utilization and overlapping of computation and communication.

By default, ZeRO-Infinity’s optimizations play in when you offload with ZeRO-3, while ZeRO-Offload is used when offloading with Stage 1/2.

*Further Reading*:

1. ZeRO-Offload/Infinity Tutorial: [https://www.deepspeed.ai/tutorials/zero-offload/](https://www.deepspeed.ai/tutorials/zero-offload/)
2. ZeRO-Offload - Democratizing Billion-Scale Model Training : [https://arxiv.org/abs/2101.06840](https://arxiv.org/abs/2101.06840)
3. ZeRO-Infinity - Breaking the GPU Memory Wall for Extreme Scale Deep Learning: [https://arxiv.org/abs/2104.07857](https://arxiv.org/abs/2104.07857)

## ZeRO\+\+ {#zero}

ZeRO\+\+ is a recent improvement to ZeRO-3 from the DeepSpeed team. Key improvements:

1. Quantized weights (qwZ) : Reduces all-gather parameter communication volume by half by quantizing model weights to int8.
2. Hierarchical Partitioning (hpZ): Hierarchical partitioning is a hybrid partitioning scheme that can help in multi-node settings with DeepSpeed ZeRO 3. In this case, you can have model parameter sharding happening within a node, and then have replication across nodes. This means that you don’t have the same amount of memory savings as classic ZeRO-3 running for the full setup, but you avoid expensive inter-node parameter communication overhead, thereby improving throughput in general. I much prefer the term “hybrid sharding” used in FSDP to “hierarchical partitioning” though, and we will revisit this when we look at FSDP below.
3. Quantized gradients (qgZ): Enables even more savings in communication volume by replacing fp16 with int4 quantized data during gradient reduce-scatter ops (Recall: this is the gradient gather \+ averaging step in ZeRO 2/3 with sharded gradients).

Overall, ZeRO\+\+ reduces communication volume by 4x with these three improvements, compared to ZeRO-3.

*Further reading*:

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- ZeRO tutorial: [https://www.deepspeed.ai/tutorials/zero/](https://www.deepspeed.ai/tutorials/zero/)
- ZeRO\+\+: Extremely Efficient Collective Communication for Giant Model Training: [https://arxiv.org/abs/2306.10209](https://arxiv.org/abs/2306.10209)
- ZeRO\+\+ tutorial: [https://www.deepspeed.ai/tutorials/zeropp/](https://www.deepspeed.ai/tutorials/zeropp/)

# Fully-Sharded Data Parallel {#fully-sharded-data-parallel}

Fully-Sharded Data Parallel (FSDP) is another data-parallelism technique aimed at improving memory efficiency with limited communication overhead, and thus throughput. FSDP’s sharding strategy is based on ideas from [Xu *et al*](https://arxiv.org/abs/2004.13336) and ZeRO. FSDP has two sharding strategies: Full Sharding and Hybrid Sharding.

## Full Sharding {#full-sharding}

This is mostly the same as ZeRO-3 where you have parameters, optimizer state and gradients being sharded across workers/ devices. From the FSDP blog, this is a pretty low-level visualization of different operations involved with 2 devices:![FSDP](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/fsdp_hufdef68c740eef12ec5f56b917cc75b3b_570277_23edc3a4ddb434e2ed7c2c86bfbf3d55.webp){width=760 height=343}

As you can see, each worker/device holds only a subset of the weights, and you have *parameter communication on-demand* to compute intermediate activations and gradients. From [PyTorch’s FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html):

> In forward path

> - Run all\_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
> - Run forward computation
> - Discard parameter shards it has just collected

> In backward path

> - Run all\_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
> - Run backward computation
> - Run reduce\_scatter to sync gradients
> - Discard parameters

Here, “rank” is a GPU worker.

Another helpful visualization, especially in contrasting full vs hybrid sharding is from the paper:![FSDP Full Sharding](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/fsdp_full_hu8d905b123872d7aaa51082b363e35f7a_54649_2179b392b3f27bee4679d064f8b85385.webp){width=658 height=480}

## Hybrid Sharding {#hybrid-sharding}

![FSDP Hybrid Sharding](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/fsdp_hybrid_hu194794a445ebe2d8654a955631b65a32_59139_74060ebf66a5761f939f69d20155a451.webp){width=629 height=410}

Hybrid sharding consists of both *sharding* and *replication*. What this means is that, given a number of workers/GPUs $W$, the sharding happens only across subsets of size $F$, with replication across different subsets. Let’s make this more specific. Suppose you wish to do multi-node training across 2 nodes with each being a `a2-ultragpu-8g` node in GCP. You have 8xA100 GPUs in each node, with a total of 16 workers. You can use hybrid sharding to shard model parameters inside each node, and then have replication across nodes. What this means is that in each forward/backward pass, you have similar all-gather and reduce-scatter operations within each node, namely to get model parameters from other GPUs (intra-node) and compute intermediate activations and gradients. Now, you further have another all-gather across nodes to get an averaged gradient value for the total mini-batch of data being processed in that training step. This is especially attractive when you’re forced to deal with sharding parameters (i.e ZeRO 2/1 is not an option) and you’re in a multi-node setup. This is similar to the “hierarchical partitioning” feature in ZeRO\+\+.

# Implementations {#implementations}

## How can you use DeepSpeed and FSDP? {#how-can-you-use-deepspeed-and-fsdp}

One of the main advantages of DeepSpeed ZeRO/ FSDP is that you get the kind of memory savings and throughput in data \+ tensor parallelism while actually being only in a data-parallel setting. This means that you do not need any ad-hoc architecture changes, or change your forward pass with messy `.to()` device castings, or any customizations. So ZeRO / FSDP really works across different architectures (which is why, lo and behold, we get good integrations). ZeRO is implemented in Microsoft’s DeepSpeed library and is integrated into the 🤗 Accelerate library. FSDP is a part of PyTorch itself, and again has an integration in the 🤗 Accelerate library. You can thus use either of these strategies, from the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) API (which uses Accelerate in the backend), or directly with Accelerate.

## What about Pipeline Parallelism and Tensor Parallelism? {#what-about-pipeline-parallelism-and-tensor-parallelism}

Pipeline Parallelism (PP) and Tensor (TP) currently require architecture changes and/or changes in the forward pass of the model. If you wish to use PP with DeepSpeed, [you’d need to make changes to the model architecture definition](https://www.deepspeed.ai/tutorials/pipeline/#alexnet). This, coupled with the fact that the 🤗Transformers library implements a dozen odd features for each model makes the situation very messy for an integration. Of course, you already have naive model parallelism with `device_map="auto"`. This is a VERY BAD strategy overall to use multiple GPUs, and only relevant if you can’t fit the model on one GPU (might as well use one GPU otherwise). If you really do want PP and TP, the best option for now seems to be to use [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and stick to the models they support (BERT, GPT-2, T5, Llama). You can also make use of ZeRO-powered DP \+ DeepSpeed PP \+ Megatron TP in [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed), but only for training models based on BERT, GPT-2 and T5. At some point, [there was an effort](https://github.com/huggingface/transformers/issues/8771), led by Stas Bekman, in trying to figure out how to get PP and TP implemented in 🤗HuggingFace, but that didn’t pan out. From [Bekman’s engineering blog](https://github.com/stas00/ml-engineering/tree/master/model-parallelism):

> 🤗 Transformers status: as of this writing none of the models supports full-PP. GPT2 and T5 models have naive MP support. The main obstacle is being unable to convert the models to nn.Sequential and have all the inputs to be Tensors. This is because currently the models include many features that make the conversion very complicated, and will need to be removed to accomplish that.

**Update 01/19/2024**: A few months later, we now have 3D parallelism support for 🤗 Transformer models with 🤗 [nanotron](https://github.com/huggingface/nanotron). I’m yet to try it out, but the library looks great!

## Are DeepSpeed ZeRO and FSDP here to stay? {#are-deepspeed-zero-and-fsdp-here-to-stay}

DeepSpeed ZeRO and PyTorch FSDP are *mostly* going to stay, or rather, will likely be replaced *only* by a strategy that is very similar. The main reason I believe so is easy of use. The only constant in our world of LLMs is change. With new models, architectures, attention implementations, positional embedding improvements, etc coming every day, the ability to swap out one architecture for another, and in a matter of hours, be able to launch a training run of a 40B\+ parameter model on 10M samples is important. Even if a new DP \+ PP \+ TP strategy ends up squeezing out higher throughput than ZeRO-powered DP, we’re likely not going to see much adoption. I think we’re also likely to see more focus and thus throughput optimizations, and perhaps even custom compute cluster configurations coming up geared towards the pure data-parallel category for the same reason. So we won’t be doing too bad sticking to only DP.

(Of course, this is not relevant for the Very Large Language Model training taken up by OpenAI, Anthropic, etc)

# Efficient Finetuning {#efficient-finetuning}

Another hot topic! I will simply list out some of the most popular optimizations:

## Mixed Precision {#mixed-precision}

This is now a no-brainer with large model training. In short, weights, activations and gradients are stored in half-precision formats while you have a “master copy” of the weights in FP32/ single-precision. The two half-precision formats commonly used are BF16 ("[Brain Float 16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)", developed by Google Brain) and FP16. FP16 needs additional loss-scaling in order to prevent gradient underflow, while BF16 doesn’t seem to have these issues.

*Further reading*:

1. Mixed Precision Training: [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
2. Performance and Scalability: How To Fit a Bigger Model and Train It Faster: [https://huggingface.co/docs/transformers/v4.18.0/en/performance](https://huggingface.co/docs/transformers/v4.18.0/en/performance)

## Parameter-Efficient Fine-Tuning (PEFT) {#parameter-efficient-fine-tuning-peft}

PEFT methods aim to reduce the memory requirements during finetuning, by freezing most of the model weights and having a subset/ a small number of additional parameters as trainable. The most popular PEFT method is [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora), where you finetune a low-rank version of weight updates to your model parameters. Another effective PEFT method is [IA$\^3$](https://huggingface.co/docs/peft/conceptual_guides/ia3), which injects trainable vectors into key, value and feedfoward layers in a transformer-based architecture. With both LoRA and IA$\^3$, the added weights/vectors can be *merged* with the base weights, meaning that, at inference time, there are no additional computations (addition/multiplication). The downside is that performance *can* be lesser than when you perform full finetuning. This however, has rapidly changed, and LoRA-based approaches can infact *match* full fine-tuning performance, if you add trainable weights to ALL linear layers (See [QLoRA](https://arxiv.org/abs/2305.14314)). With respect to IA $\^3$, in my experiments with small models like GPT-2 (770M), I’ve found that IA $\^3$ can match LoRA’s performance with less than 1/10th the number of trainable parameters. This still needs more community experimentation though, especially at the scale of LLama-2-7B or Falcon-40B.

*Further reading*:

- Low-Rank Adapatation of Large Language Models: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- A Conceptual Guide to LoRA in 🤗 PEFT: [https://huggingface.co/docs/peft/conceptual\_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora)
- A Conceptual Guide to IA$\^3$ in 🤗 PEFT: [https://huggingface.co/docs/peft/conceptual\_guides/ia3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- Efficient Fine-tuning of T5-XXL with LoRA: [https://www.philschmid.de/fine-tune-flan-t5-peft](https://www.philschmid.de/fine-tune-flan-t5-peft)

## Flash Attention {#flash-attention}

Flash attention is a *fast* (you get speedup!), *memory-efficient* (while saving memory!), *exact* (with no approximations!) *IO-aware* (by accounting for reads/writes across different levels of GPU memory!) attention algorithm. With FlashAttention 2 (it’s only been here since July 2023), you can get 220\+ TFLOPS on an A100 80GB (which has a maximum of 315 TFLOPS). To put that into perspective, when Flash Attention 1 came along in mid-2022, it had the best throughput you could get, which was up to 124 TFLOPS. Further, with some of the DeepSpeed ZeRO papers we looked at before, the achievable peak throughput was estimated to be around 70 TFLOPS in 2021 on a Tesla V100!

Currently, Flash Attention supports Ampere, Ada, or Hopper NVIDIA GPUs (A100, RTX 3090, H100,etc), and only half precision datatypes bf16/fp16. To use flash attention with 🤗Transformers, you only need [one flag change](https://x.com/younesbelkada/status/1705258148045750343) for LLama and Falcon (pass `use_flash_attention=True` to `AutoModel`). For the rest of the 🤗 models, this requires manually changing the attention function used in the `forward` methods to flash-attention’s high throughput version, [but this is fast developing](https://github.com/huggingface/transformers/issues/26350). Flash Attention should be integrated into PyTorch’s `scaled_dot_product_attention` soon, so that you don’t have to rely on monkey patches. (This was supposed to happen with v2.1, but will hopefully happen in the near future)

Flash Attention v1.0 has also been available in 🤗[Optimum](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#decoder-models) for some time, but you can’t have padding tokens, which makes it very restrictive.

*Further reading*:

- ELI5: Flash Attention: [https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning - [https://tridao.me/publications/flash2/flash2.pdf](https://tridao.me/publications/flash2/flash2.pdf)

## Gradient/ Activation Checkpointing {#gradient-activation-checkpointing}

Typically, in each forward pass, all the intermediate activations are retained in memory, as they are needed to compute the backward pass. Gradient/Activation checkpointing is a technique to reduce memory consumption by only retaining a subset of intermediate activations, and recomputing the rest as needed. The tradeoff involved is in the additional recomputation time. A good rule of thumb from HuggingFace is that gradient checkpointing slows down training by 20%. The memory requirement for activations, when you have N model layers, drops off from $O(N)$ to $O(\sqrt{N})$. Of course, with total memory consumption, the drop-off would not be this great, because you have a lot more than activations being stored.

*Further reading*:

- Fitting large networks into memory: [https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
- Performance and Scalability: How To Fit a Bigger Model and Train It Faster: [https://huggingface.co/docs/transformers/v4.18.0/en/performance](https://huggingface.co/docs/transformers/v4.18.0/en/performance)

## Quantization {#quantization}

This one’s a ✨ hacker favourite ✨! I’m going to be brief here. With quantization approaches, there are two kinds:

- *Post-Training Quantization (PTQ)* - These are approaches aimed at efficient *inference*. [LLM.int8()](https://arxiv.org/abs/2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323) come here.
- *Quantization-aware training*: In the original sense of the term, this meant to train a model from the start using quantized weights and activations, for use later during inference. This is what we want here - a strategy to train with quantized parameters. [QLoRA](https://arxiv.org/abs/2305.14314) comes in this category (kind of, since it does integrate quantization into the training process). The main idea with QLoRA is that it quantizes the base, pretrained model weights to 8/4 bits and then trains additional LoRA parameters in floating-point half/full precision. This is a very powerful strategy, enabling finetuning of 60B\+ parameter models on a single GPU with 48GB vRAM. The caveat, of course, is that you are stuck with much worse throughput. The reason is simple - you have an additional de-quantization step happening whenever you compute activations for a given layer. (The exact number is dependent on the hardware setup. For example, you might be able to run Falcon-40B training with DeepSpeed ZeRO 3 on 8xA100s without quantization, and training with quantization would be useless here, even if you can get a better batch size. This is not the same with just 2xA100s).

**Side Note**: The full QLoRA paper is worth reading. Beyond the fact that their approach enabled training a 65B model on 1 consumer GPU (this was the largest open source language model at the time), the paper also showed that LoRA-based training can *match* full fine-tuning (they added more trainable layers, but quantizing the base weights made it *more efficient* than the original LoRA configuration), and that dataset quality was paramount in supervised finetuning (450K FLAN samples is worse than 9K high-quality human-labelled samples).

*Further reading*:

- QLoRA: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- Quantize 🤗Transformer models: [https://huggingface.co/docs/transformers/v4.34.0/en/main\_classes/quantization](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization)

## Gradient Accumulation {#gradient-accumulation}

Gradient accumulation is a way to increase your effective batch size at some drop in throughput (and sometimes, for free!). This is best explained with an example. Let’s say you have batch size of 2, and a gradient accumulation step size of 4. With gradient accumulation, you would have your regular forward and backward passes in every training step, but, you would have an optimizer step (`optimizer.step()` in PyTorch) being called once every 4 training steps. What this means is that gradients get accumulated over 4 steps and then the model weights get updated using the average gradients across 4\*2 \= 8 samples. You’ve now got an increase in batch size, making weight updates less noisy, but with the same memory consumption. In fact, gradient accumulation with multi-GPU/ multi-node training gives you larger batch size along with *faster* training times. This is because, in normal training, you have gradients being averaged out locally (for the batch handled by a GPU) and you perform an all-reduce operation every training step. Now, with gradient accumulation, you perform this all-reduce operation at larger intervals ( which is `gradient_accumulation_steps`). Reducing the number of such all-reduce operations leads to lesser inter-worker (and possibly also inter-node) communication and better training times.

*Further reading*:

- Training with the DeepSpeed API: [https://www.deepspeed.ai/training/](https://www.deepspeed.ai/training/)

### So wait, should I always try to increase batch size? {#so-wait-should-i-always-try-to-increase-batch-size}

This is a tempting question after reading everything we’ve talked about. The answer: No! Remember that the goal is to train the best possible neural net you can get, as fast as possible for easy experimentation, using available hardware. So, even if you’re using let’s say only 75% of available GPU memory (60 GB/ 80GB in an A100), you might have reached the maximum possible throughput in your system - this means that further increases in batch size will lead to corresponding increases in per-step latency, resulting in no boost in throughput or even lesser throughput. Further, even if you’ve got a kickass hardware setup to train 40B\+ parameter models with large batch size, there’s more to simply increasing batch size with the above memory optimizations,because this can hurt convergence. It’s hard to pin down a good study on this for large models at scale. There’s a [pre-Transformer era paper](https://openreview.net/forum?id=H1oyRlYgg) which showed that large batch sizes can hurt generalization. DeepSpeed mentions that you can find better hyperparameter/ optimizer choices to make large batch sizes work, as evidenced by [this guide on 1-cycle learning rate schedule](https://www.deepspeed.ai/tutorials/one-cycle/). One more insight from a DeepSpeed author is that a global batch size is usually fixed for large scale training runs to achieve a targeted convergence rate ([source](https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)). All in all, more batch size $\neq$ better!

**How big is too big?** To provide some perspective, BLOOM-176B pretraining was done on 366B tokens with a global batch size of 2048. What is too big to hurt model convergence is not clear yet, and that too in the fine-tuning regime.

# Practical Guidelines {#practical-guidelines}

## The Ultimate Summary {#the-ultimate-summary}

Combining everything we’ve talked about, here are some practical guidelines while trying to experiment and finetune 10-100B\+ model parameter on 1M\+ size datasets (I have experience with DeepSpeed and not FSDP yet, so I’ll focus on that):

- BF16/ FP16 by default. BF16 comes with basically no other config parameters and usually without any overflow issues (as opposed to fp16, where you can get different results with different loss scaling factors and have more overflow issues because of smaller dynamic range), so it’s very convenient.
- Use LoRA with trainable parameters added to all the linear layers. If you want to follow QLoRA closely, you can use their utility function [here](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L248).
- Use Flash Attention if your GPU supports it. Currently, Flash Attention 2 is supported for Llama 2 and Falcon in HuggingFace, with other models requiring monkey patches.
- Use Gradient/Activation Checkpointing. This will reduce throughput slightly. If you’ve got Flash attention, gradient checkpointing might not be required. From the flash attention paper (See also: [Tri Dao’s recommendation](https://github.com/EleutherAI/gpt-neox/pull/725#issuecomment-1374134498)):

![Flash attention](https://sumanthrh.com/post/distributed-and-efficient-finetuning/post/distributed-and-efficient-finetuning/flash_hue82fc85cc5c167ea9842618f33fbb382_153888_a443256b242014a0ff2d4a1cc88c8dee.webp){width=760 height=158}

- Use an efficient sampler in your dataloader, like the [multi-pack sampler](https://github.com/imoneoi/multipack_sampler).
- If you have multiple GPUs, always try BF16 \+ LoRA \+ Gradient Checkpointing \+ DeepSpeed ZeRO 3 first.
- Use quantization when you have very limited GPU memory. QLoRA-style training currently works for DeepSpeed ZeRO 1/2 only. Thus, even though it is memory efficient when it comes to model parameters, you still have parameter redundancy with ZeRO 1/2, and you also get reduced throughput.
- With more and more GPUs (say 8 V100s or A100s), DS ZeRO-3 should be the way to go. DS ZeRO-2 is also good, but you can start to hit CPU RAM limits (during model initialization) because of model parameters being replicated across all workers. For example, with Falcon-40B model on a node with 8 GPUs, you need > 1.5 TB of CPU RAM, and you rarely get this in your cloud container instances (AWS, GCP, etc), so you’re screwed. Of course, this might not applicable for a scrappy home server where you don’t have NVLink for inter-GPU communication. DS ZeRO-3 has a lot more inter-GPU communication, which is why NVLink would matter.
- In a small-scale multi-node setup, with a few nodes, the best option seems to be DeepSpeed ZeRO-3 with hierarching partitioning enabled (or FSDP with hybrid sharding). If you’ve got Infiniband interconnect, you can mostly use plain DeepSpeed ZeRO-3 and push for larger model sizes as well.
- Gradient accumulation should be used if you’re still short on batch size after all the above optimizations. Training times with gradient accumulation can be faster with large models and multi-GPU/ multi-node settings.
- If you’re really short on GPU memory, then you would activate CPU/ disk offloading (and by disk, this has to be NVMe for ZeRO-Infinity). With the advent of Flash Attention 2, we need another study on the throughput gap between plain GPU-based training and GPU \+ NVMe/CPU-offloading. I suspect this gap is much higher than before now, and thus offload only if really constrained (which is why, this is the last optimization to try). ZeRO-Infinity is better than ZeRO-Offload, and you have to use ZeRO Stage 3 for this.
- Calculate the effective batch size and adjust hyperparameters accordingly. A general guideline is to scale up the learning rate with the effective batch size. This seems to hold true even for 100B\+ models, [as seen in OpenAI’s finetuning docs](https://platform.openai.com/docs/guides/legacy-fine-tuning/hyperparameters).
- Finally, when you do start training, monitor `htop` to check on RAM usage (sometimes RAM OOM can be an issue), along with `nvidia-smi` to make sure GPUs aren’t bottlenecked by data preprocessing (you should aim for close to 100% volatile GPU utilization, even if GPU memory usage is lesser).

**More on hyperparameters**: A comment on the learning rate scaling. Turns out that with larger and larger models, 100B\+ say, the general guideline is to *decrease* the learning rate during *pretraining* rather than increase, even though you use a larger batch size. This is evidenced by OpenAI’s [GPT-3 paper](https://arxiv.org/abs/2005.14165) and the [BLOOM paper](https://arxiv.org/abs/2211.05100). It is pretty hard to build intuition with such patterns, so we’re left with adopting a very empirical approach, and to quickly update all priors if experiments say otherwise.

## Additional Guidelines {#additional-guidelines}

There are a number of useful tips in 🤗Transformers’ documentation [on performance and scalability](https://huggingface.co/docs/transformers/v4.18.0/en/performance). If you’re especially interested in pitfalls related to building and managing a server at home, or how exactly NVLink matters, or just a deeper look at memory management, it is excellent. [Stas Bekman’s collection](https://github.com/stas00/ml-engineering) also has a number of debugging and performance related tips that might be useful.

# More on DeepSpeed and FSDP {#more-on-deepspeed-and-fsdp}

## Multi-node with DeepSpeed ZeRO-3 {#multi-node-with-deepspeed-zero-3}

With plain ZeRO-3, you need to make sure your inter-node parameter communication does not become a big bottleneck, since otherwise you will likely see very little throughput gains even with, say, 2 nodes vs a single node. From [Stas Bekman’s investigations](https://github.com/microsoft/DeepSpeed/issues/2928), we have some clarity on what kind of inter-node network you need to have:

For a model size $M$, with $N$ nodes, each node having $G$ GPUs, you have about $48 \* M/N$ bits of data being sent out per training step from each node. Even for 2 8xA100 nodes, with a 40B parameter model, this is 120GB of data, that each node communicates per training step! For large scale training (64GPUs\+), you really do need InfiniBand interconnect with 1000 Gbps. For smaller-scale multi-node training, you can get away with 100-400 Gbps.([source](https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)). With [Amazon EC2 P4d instances](https://aws.amazon.com/ec2/instance-types/p4/) (the nodes with 8xA100s meant for ML training), you typically have Elastic Fabric Adapter (EFA) as the network interface across nodes, and, according to the spec, you can get up to 400 Gbps of network bandwidth which is decent! The exact number you end up getting is \~ 340 Mbps, and one should plan for 80-85% of the maximum bandwidth listed in the spec \[[This is a practical tip from Stas](https://x.com/StasBekman/status/1710038192983019547?s=20)\]. EFA v2 used for AWS P5 instances with H100s is a crazy [8x faster](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p5-instances-powered-by-nvidia-h100-tensor-core-gpus-for-accelerating-generative-ai-and-hpc-applications/).

To measure the exact inter-node network bandwidth you’re getting, you can make use of Stas Bekman’s utility function [here](https://github.com/stas00/ml-engineering/blob/9a51114f8377350bfbf1764f23feac441e865401/multi-node/all_reduce_bench.py). This is specifically meant for benchmarking all-reduce, so it’s an accurate test for what you’ll see while training.

(Update: An inital version of this blog had EFA bandwidth numbers for an EC2 Inf instance, and not P4d, which turns out be much lesser. It has now been corrected)

**Recommendation**: When you don’t have Infiniband, it’s simply better to make use of ZeRO\+\+ and use hierarchical partitioning (hpZ). To enable this, you will set the `zero_hpz_partition_size` config parameter to the number of GPUs/ ranks per node. If you’re training with 2 nodes, each with 8xA100 GPUs, this means that `zero_hpz_partition_size` would be 8.

🤗**Accelerate/Transformer support**: It is currently unclear if hierarchical partitioning (hpZ) is supported in Accelerate’s DeepSpeed integration. It should be from the looks of it since Accelerate is supposed to integrate all the features of DeepSpeed ZeRO, and hpZ is a one parameter change in your DeepSpeed config file. I’ve raised an [issue](https://github.com/huggingface/accelerate/issues/2020) (10/01/2023), and will update this post if needed.

With FSDP, you will use the hybrid sharding strategy (`HYBRID_SHARD`). This really seems like a no-brainer for multi-node training (atleast small scale, without Infiniband), and [one user found out the hard way](https://github.com/pytorch/pytorch/issues/102434#issuecomment-1569776688). FSDP with hybrid sharding is already supported by 🤗Accelerate.

## DeepSpeed Memory Requirements {#deepspeed-memory-requirements}

When you’ve got a new infra setup and wish to try out DeepSpeed, you should definitely use [DeepSpeed’s memory estimators](https://deepspeed.readthedocs.io/en/latest/memory.html)

### DeepSpeed ZeRO 1/2 {#deepspeed-zero-12}

```python
deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(model,
  num_gpus_per_node=1, num_nodes=1, additional_buffer_factor=1.5)
```

Using the default buffer factor (which is an estimation factor that simply scales all the CPU and GPU memory estimates), you get the following results for a 3B parameter model on 1 node with 8 GPUs, from the DeepSpeed docs:

```fallback
python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("t5-3b"); \
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'

Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 8 GPUs per node.
SW: Model with 2851M total params.
  per CPU  |  per GPU |   Options
  127.48GB |   5.31GB | offload_optimizer=cpu
  127.48GB |  15.93GB | offload_optimizer=none
```

With Falcon-40b for example, the CPU RAM needed with ZeRO 1/2 comes out to be > 1.5TB. You don’t have to actually run your training/finetuning code to test this out - Simply run the above command with your model and hardware setup, and you’ll get an estimate. For FSDP, I did not find a similar memory estimator, so you can just use the DeepSpeed ZeRO-3 estimates for full-sharding with FSDP.

**Note**: One more comment on CPU RAM shortage. This can be tricky to debug, because your process will fail without any info in your logs. Adding pytorch distributed debug flags (`NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=INFO`) also does not help because this is a RAM problem. I’ve simply monitored `htop` (as mentioned previously) in the initial training stage and found out the hard way.

### DeepSpeed ZeRO 3 {#deepspeed-zero-3}

```fallback
python deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model, \
  num_gpus_per_node=1, num_nodes=1, additional_buffer_factor=1.5)
```

An example output, for a 3B parameter model on 1 node with 8 GPUs:

```fallback
python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("t5-3b"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'

Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 8 GPUs per node.
SW: Model with 2851M total params, 32M largest layer params.
  per CPU  |  per GPU |   Options
   71.71GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  127.48GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   63.74GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  127.48GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    1.47GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=1
  127.48GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=0
```

This also gives you an idea about different offloading strategies and initialization! `zero_init=1` will initialize the model weights in a memory scalable fashion, where the weights will be immediately partitioned across your workers once allocated. CPU RAM requirements can go through the roof when you use `zero_init=0` (Terabytes for 10B\+ models with multiple GPUs), so you should most definitely use `zero_init=1`. CPU Offloading in a multi-GPU setting is again RAM heavy, and can cause RAM OOM. Use the estimator and then, if really needed switch to NVMe offloading if CPU RAM is not enough.

## Usage with 🤗Accelerate {#usage-with-accelerate}

🤗Accelerate is meant to provide a unified interface for launching a variety of distributed training runs while giving you the flexibility of writing code in plain PyTorch. Currently, it looks like there are caveats if you’re thinking of using the same code to switch between FSDP and DeepSpeed. For example, with FSDP, you have to call `accelerator.prepare(model)` before instantiating the optimizer. I’m not sure whether the same method works normally with DeepSpeed (With DeepSpeed you can simply have one `.prepare()` call for everything). There are a few more caveats, which I won’t get to here, but you can have a look at some of the 🤗Accelerate docs below.

*Further reading*:

1. FSDP with 🤗Accelerate: [https://huggingface.co/docs/accelerate/usage\_guides/fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
2. Fine-tuning Llama 2 70B using PyTorch FSDP: [https://huggingface.co/blog/ram-efficient-pytorch-fsdp](https://huggingface.co/blog/ram-efficient-pytorch-fsdp)

# Open Source Codebases {#open-source-codebases}

Open source codebases have come a long way in just a few months. The main question I want to answer here is “If I want to start finetuning right now using open source codebases, what can I use? What should I keep in mind?”. As always, the devil is in the details. I’ll summarize some of the functionality available in two of the most popular and useful platforms, [FastChat](https://github.com/lm-sys/FastChat) and [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl).

## FastChat {#fastchat}

FastChat is a platform for finetuning, serving and evaluating LLM-based chatbots from [LMSys](https://lmsys.org/). The features:

### Serving {#serving}

You can serve models like Llama, Falcon, WizardLM using FastChat. ([List of supported models](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md#supported-models)) Adding new models for inference/serving seems to be very straightforward, with support for both causal (like Llama) and sequence-to-sequence models (like T5). They also support serving with CPU Offloading and different quantization schemes. Under the hood, FastChat uses the awesome [vLLM](https://vllm.ai/) library for efficient inference, which is another open-source project from LMSys.

### Finetuning {#finetuning}

- The supported models for finetuning seem to be Llama, T5 and Baichuan. I’m happy to be corrected here. The main finetuning script is only for Llama models, and has [Llama-specific magic numbers](https://github.com/lm-sys/FastChat/blob/e64ee0e9a0d2d1a987a59a1cfe24bf711f3dec09/fastchat/train/train.py#L133). There’s an additional finetuning script for T5 and Baichuan models. Training support for Falcon is still an [open issue](https://github.com/lm-sys/FastChat/issues/1588).
- This is specifically for supervised finetuning on (single/multi-turn) conversation data, aimed at getting a chatbot. Doing instruction-tuning on a FLAN-like dataset with (instruction, response) pairs should also be possible (single turn conversation). But mixing other dataset formats, like say a dataset for causal language modelling, or training on multiple datasets is not possible.
- LoRA and QLoRA-based training is supported. The [provided DeepSpeed configs](https://github.com/lm-sys/FastChat/tree/main/playground) are reference configs meant for a low-resource setting (Say 1 V100 32GB for finetuning Llama-13B) and thus have CPU offloading enabled by default. Make sure to modify for your hardware setup. For further reference, see the [training docs](https://github.com/lm-sys/FastChat/blob/e64ee0e9a0d2d1a987a59a1cfe24bf711f3dec09/docs/training.md).
- FastChat uses the 🤗Trainer API and like almost all open-source training packages available, supports only one homogeneous dataset (in this case, it also has to be conversational) for train and eval. This means you get to see training loss and evaluation loss for the combined dataset, but nothing else while monitoring your runs.

### Evaluation {#evaluation}

FastChat’s evaluation package is based on [MT-bench](https://arxiv.org/abs/2306.05685), a multi-turn conversation-based evaluation dataset. For scoring, they use an LLM-as-a-judge approach where you use a more powerful language model like GPT-3.5 or GPT-4 to score model outputs.

## Axolotl {#axolotl}

Axolotl is a massive open-source effort in finetuning language models, with the following salient features:

- Support for all sorts of causal language models, like Llama, MPT, Falcon, etc. Sequence-to-sequence models are not supported yet.
- Supports training on multiple datasets with a variety of formats. In my opinion, this is an extremely important feature that Axolotl has. The full list of dataset formats is too huge, but you can train on instruction-tuning datasets like FLAN, conversation-based datasets like ShareGPT, and simple completion-based datasets (plain text/code for causal language modelling). Axolotl also supports pre-tokenized datasets.
- LoRA, QLoRA support, along with support for the multipack sampler, which packs sequences of similar length together to avoid wasting compute on padding tokens.
- Axolotl also uses the 🤗Trainer API, and has a number of features for custom evaluation and logging. You can evaluate on MMLU, or a local benchmark dataset and log loss/ accuracy during training.
- Axolotl further supports both FSDP and DeepSpeed, mainly because they just let the Trainer handle this. Flash-Attention is also available for a number of models like Llama, [BTLM](https://huggingface.co/cerebras/btlm-3b-8k-base) and the new [Mistral](https://mistral.ai/news/announcing-mistral-7b/).
- Evaluating on different tasks like MMLU and MATH and visualizing loss curves separately is not supported, mainly because it’s hard to customize this with the Trainer.
- A nitpick here from me is that Axolotl has ended up customizing almost every single part of the Trainer, barring some methods like `prediction_step` and `training_step`. If they’ve gone low-level enough to [use `torch.distributed.gather`](https://github.com/OpenAccess-AI-Collective/axolotl/blob/90e0d673f76f84478856434deb6024c5c869a5ad/src/axolotl/utils/callbacks.py#L280C1-L280C61), I wonder why they didn’t just start off with writing everything in plain PyTorch \+ 🤗Accelerate, since they could have avoided bloat. The Trainer also gives you the flexiblity to choose from a dozen odd optimizers, etc, but you just need a small subset for supervised finetuning.

## Useful Fine-tuning Guides {#useful-fine-tuning-guides}

A few recently published guides on finetuning from 🤗HuggingFace are worth a look:

- Fine-tuning Llama 2 70B on 2 8xA100-ultra-80GB nodes using 🤗PEFT and FSDP: [https://huggingface.co/blog/ram-efficient-pytorch-fsdp](https://huggingface.co/blog/ram-efficient-pytorch-fsdp)
- Fine-tuning Falcon 180B on 2 8X100-ultra-80GB nodes using 🤗PEFT and DeepSpeed: [https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99](https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99)
- Fine-tuning Falcon 180B on 1 8xA100-ultra-80GB node using 🤗PEFT and DeepSpeed: [https://www.philschmid.de/deepspeed-lora-flash-attention](https://www.philschmid.de/deepspeed-lora-flash-attention)

# Going Very Large {#going-very-large}

First, a comment on how large you can go with just 1 node with 8xA100s. With a number of the above optimizations, you can currently finetune Falcon 180B on a [DGX](https://www.nvidia.com/en-us/data-center/dgx-a100/) node with 8 A100-80GB GPUs. From [HuggingFace’s guide](https://www.philschmid.de/deepspeed-lora-flash-attention), you can finetune a 180B parameter model, for 3 epochs on the [dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset with 15k samples, with an effective batch size of 64 and a sequence length of 2048 tokens, in 153 minutes. The exact throughput number is unclear because text chunking is done as a preprocesssing step, and of course we need an estimate for time per training step (since total training time includes the time for evaluation and checkpointing), but it looks be > 5 samples per second, which is pretty good for a 180B model.

For more on managing large scale training runs, we turn to Stas Bekman. I’ve mentioned this before but I’ll put this here again: The open-source community has a lot to thank [Stas Bekman](https://github.com/stas00) for, among other things, his [excellent writeups on large-scale training](https://github.com/stas00/ml-engineering). Debugging large scale training runs, monitoring, diving deeper into network setups for large scale multi-node, etc are all found in the above repo. Some of the material on distributed training and the recommendations there might be a little outdated, however.

# The End {#the-end}

If you’ve made it through the whole post, good job - this was 7000\+ words. Hope you’ve found this useful!


Check: the five engineering problems in order — is the causal chain right? The ZeRO ladder numbers? The 12-vs-16 bytes/parameter discrepancy — did I represent both sides fairly?

Output: findings HIGH/MEDIUM/LOW, then verdict.