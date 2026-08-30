title: RLHF learning resources in 2024
description: A list for beginners and wannabe experts and everyone in between.
author: Nathan Lambert

# RLHF learning resources in 2024

I’ve given a lot of effort into sharing information on Reinforcement Learning from Human Feedback (RLHF). I figured I would categorize them in one place for people who come to me or Interconnects looking to learn about the topic.

This was inspired by [my recent appearance on Latent Space](https://www.latent.space/p/rlhf-201) , which we called RLHF 201. Doing this made me realize, once again, how few resources there are out there for going deeper on RLHF other than often confusing research papers. The slides for this talk are available [here](https://docs.google.com/presentation/d/1ke59UrJk79m-cZYCBynOucDCyBjzcZxDGDMxeFGs19k/edit?usp=sharing) . Compared to my last lecture, I added a bunch of the underlying math, made figures cleaner, and added commentary on evaluation. The [previous generation of slides I used at Stanford](https://docs.google.com/presentation/d/1T6X8ZlwrBek14wGfKljLxikwkTBDdM88r0AZ6NiodU4/edit?usp=sharing) are also good, and they have a longer introduction.

Generally, the goal for this post is to give people with different learning styles the tools to learn more in their way of choice. I’ve split it up by video mediums (talks and podcasts), technical mediums (code and models or datasets), and text (which is mostly blog posts). Almost all of these link to papers within them, if you’re looking to go into more detail.

This list is obviously biased towards my stuff and is not a review, so plenty of things I’ve seen aren’t included.. It’s meant to give entry points for people wishing to go deeper on the subject. **If you send me things that you think should be added and why, I’ll happily take a look.**

Generally, I’ll give a very light description as to why I like every piece of content.

![](https://substackcdn.com/image/fetch/$s_!snCj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F31db6f45-8fc3-466d-ac8b-5df527258463_2000x1143.png){width=1456 height=832}

## Video

### Tutorials and overviews

- December 2022, [Reinforcement Learning from Human Feedback: From Zero to chatGPT](https://www.youtube.com/watch?v=2MBJOuVq380) . This was my first big lecture on RLHF. Included for posterity and remembering the excitement of the earliest days.
- April 2023, John Schulman’s Berkeley RLHF talk, “ [RL and Truthfulness](https://www.youtube.com/watch?v=hhiLw5Q_UFg) .” This is still the best talk on intuitions of why RLHF tuning on outputs from other models is risky from a capabilities point of view.
- July 2023, My [tutorial at ICML](https://icml.cc/virtual/2023/tutorial/21554) . A solid introduction with an hour on data from my colleague at Toloka AI.
- July 2023, John Schulman’s [Proxy Objectives in RLHF](https://icml.cc/virtual/2023/invited-talk/21549) at ICML. This talk has some revealing details on the outer loop training they do for ChatGPT.

### Research talks of mine

- March 2023: [Reinforcement Learning from Human Feedback: Open and Academic Perspectives](https://www.youtube.com/watch?v=8SgKDSX-Me0) ( [slides](https://docs.google.com/presentation/d/1jUipk6Mu8On2mWu_qfiocmbgnNux7SWp/edit#slide=id.p1) ): A decent introduction with advice on how academics can work in this area.
- August 2023: [Objective Mismatch in Reinforcement Learning from Human Feedback](https://www.youtube.com/watch?v=yrdUBwCnMr8) ( [slides](https://www.google.com/url?q=https://doi.org/10.5281/zenodo.8186168&sa=D&source=editors&ust=1695142915789145&usg=AOvVaw1VoXzDLJn26YshoTz3wu5Z) ): understand the fundamental tradeoffs and sources of uncertainty in RLHF.
- November 2023: [Bridging RLHF from LLMs back to control](https://www.youtube.com/live/ThgGAZF4hgI?si=2fM-bYOrM2ETmtm5&t=8848) ( [slides](https://docs.google.com/presentation/d/15i_7iqyUJwDMtyzzzSn83JLETET4Lz2Y7e4sDCAZhfM/edit#slide=id.g82736d3e0d_0_26) ). Make connections on current RLHF progress back to other fields that have been using RL for much longer!
- December 2023: [15min History of Reinforcement Learning and Human Feedback](https://www.youtube.com/watch?v=Mu_-FWIuhDA) ( [slides](https://docs.google.com/presentation/d/1kBEKvHBugXE5tSnwp2_kQ1VdV5bOP_LveY6gkTe_Vco/edit?usp=sharing) ). Answers: what are the core motivating fields of RLHF?
- December 2023: [Direct Preference Optimization (DPO): Easy to start, hard to master (maybe)](https://www.youtube.com/watch?v=YJMCSVLRUNs) ( [slides](https://docs.google.com/presentation/d/1gCnS6Sv2ynER3hB8TKyVJE4VxlcR8iGkg5iFIbZa4a8/edit#slide=id.g2a105985c94_0_0) ). Get enough info to know that we won’t have a DPO answer in 2024.
- Videos from the [New Orleans Alignment Workshop](https://www.alignment-workshop.com/nola-2023) (before NeurIPs) has a bunch of appealing talks. Anca’s was specifically recommended to me.

### Other podcasts

- January 2023: [TWIML Reinforcement Learning - RLHF, Robotic Pre-Training & Offline RL with Sergey Levine](https://www.youtube.com/watch?v=dvO_jR1B5rs) . This holds up really well when thinking about integrating long-term focuses of RL research into RLHF methods.
- September 2023: [Generating Conversation: RLHF and LLM Evaluations with Nathan Lambert (Episode 6)](https://www.youtube.com/watch?v=u8xxEkH3a5g) . I still thought this was one of my better podcast appearances for the year of 2023!
- January 2024: [RLHF 201 - with Nathan Lambert of AI2 and Interconnects](https://www.latent.space/p/rlhf-201) (video with slides [here](https://www.youtube.com/watch?v=3WU6fl7DHj0) ). In this, we discuss all the core topics in RLHF as we get ready for 2024. It was a great time to make this, and I think there’s a lot of details to learn from it if you have the basics.

## Research

The iteratively updated list of papers I come across in the area is [here](https://www.craft.me/s/NHvR6dsCVNNW8L) (which I want to update soon). It’s the basis for [this series](https://www.interconnects.ai/p/rlhf-lit-review-1-and-missing-pieces) , which I intend to continue.

I wrote two position / survey papers last fall covering what I expect to be the core themes unfolding in RLHF in the next few years. If you want a deeper take, I whole heartedly recommend them.

1. On reward models, the limitations of preferences, and more: *[The History and Risks of Reinforcement Learning and Human Feedback](https://arxiv.org/abs/2310.13595).*
2. On the fundamental tradeoffs of different RLHF pieces: *[The Alignment Ceiling: Objective Mismatch in Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2311.00168).*

There are two surveys of the area worth looking at too. 

1. *[Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217)*  serves as a critique of the RLHF perspective from a mostly AI Safety angle and with a focus on LLM techniques. 
2. *[A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925)*  covers a much broader base than most of the paper’s I’ve linked. It’s important to remember that RLHF is much bigger than just LLMs.

The [further reading section](https://huggingface.co/blog/rlhf#further-reading) of my first primary blog post on RLHF is a good place to start with the classics of the field, with the likes of InstructGPT, Anthropic’s work, etc. It’s quoted here:

> - [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)  (Zieglar et al. 2019): An early paper that studies the impact of reward learning on four specific tasks.
> - [Learning to summarize with human feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html)  (Stiennon et al., 2020): RLHF applied to the task of summarizing text. Also, [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862)  (OpenAI Alignment Team 2021), follow on work summarizing books.
> - [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)  (OpenAI, 2021): Using RLHF to train an agent to navigate the web.
> - InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)  (OpenAI Alignment Team 2022): RLHF applied to a general language model \[ [Blog post](https://openai.com/blog/instruction-following/)  on InstructGPT\].
> - GopherCite: [Teaching language models to support answers with verified quotes](https://www.deepmind.com/publications/gophercite-teaching-language-models-to-support-answers-with-verified-quotes)  (Menick et al. 2022): Train a LM with RLHF to return answers with specific citations.
> - Sparrow: [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375)  (Glaese et al. 2022): Fine-tuning a dialogue agent with RLHF
> - [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)  (OpenAI 2022): Training a LM with RLHF for suitable use as an all-purpose chat bot.
> - [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)  (Gao et al. 2022): studies the scaling properties of the learned preference model in RLHF.
> - [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)  (Anthropic, 2022): A detailed documentation of training a LM assistant with RLHF.
> - [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)  (Ganguli et al. 2022): A detailed documentation of efforts to “discover, measure, and attempt to reduce \[language models\] potentially harmful outputs.”
> - [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294)  (Cohen at al. 2022): Using RL to enhance the conversational skill of an open-ended dialogue agent.
> - [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241)  (Ramamurthy and Ammanabrolu et al. 2022): Discusses the design space of open-source tools in RLHF and proposes a new algorithm NLPO (Natural Language Policy Optimization) as an alternative to PPO.
> - [Llama 2](https://arxiv.org/abs/2307.09288)  (Touvron et al. 2023): Impactful open-access model with substantial RLHF details.

## Code

There’s a lot of code out there for RLHF. Not all of it is that easy to work with or learn from. I worked on the first two.

- [Alignment handbook](https://github.com/huggingface/alignment-handbook) is probably the cleanest to start with and build off of from a researcher’s point of view.
- [TRL](https://github.com/huggingface/trl) is the place that’s usually the fastest to implement minimal implementations of all the new algorithms. Lot’s of examples that can be run on single GPUs usually.
- [DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat) (paper is [here](https://arxiv.org/abs/2308.01320) ). While very different engineering setup, it is good to compare different ways of implementing the same stuff.
- [TRLX](https://github.com/CarperAI/trlx) , while kind of no longer supported, has some of the most in-detail logs on scaling algorithms like PPO.

## Models

Obviously there are way too many to do a thorough study of, but the most important open RLHF models and datasets of the last year to me are:

- [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) (led to [Tulu 2](https://huggingface.co/allenai/tulu-2-dpo-70b) , Stability’s model, Intel’s model, and more) was the spark that gave us the proliferation of DPO and generally useful RLHF models.
- [Starling](https://starling.cs.berkeley.edu/) was a recent model with great performance that intriguingly did ***not*** use DPO.
- [Llama 2](https://arxiv.org/abs/2307.09288) still has more details in their paper than most labs have tried with respect to RLHF.

## Datasets

- [UltraFeedback](https://arxiv.org/abs/2310.01377) : the dataset that gave us Zephyr et al. There’s even been more [research trying to improve the dataset and RLHF performance](https://argilla.io/blog/notus7b/) .
- [Open Assistant 1](https://huggingface.co/datasets/OpenAssistant/oasst1) : The community-generated instruction data that yielded the first wave of progress in open IFT training.
- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) : The first popular synthetic instruction data.
- [ShareGPT](https://sharegpt.com/) and variants: large datasets people are using to try and get ChatGPT-like abilities in open data.

## Evaluations

These three evaluations are the comprehensive set of how RLHF models are relatively ranked.

- [ChatBotArena](https://chat.lmsys.org/) : The crowd-sourced comparisons website that is the go-to source of model quality for open and closed models alike.
- [MT Bench](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) : A two turn chat evaluation also built by LMSYS, which is very well correlated with most real-world evaluations of LLMs.
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) : The first GPT4-as-a-judge tool to proliferate LLM-as-a-judge practices.

## Blog posts

### Interconnects posts

From the 2023 year in review post:

1. Feb. 27: [The RLHF battle lines are drawn](https://www.interconnects.ai/p/rlhf-battle-lines-2023)  covers the importance of RLHF to the LLM ecosystem, the costs of building it, and where the year will take us.
2. Apr. 26: [Beyond human data: RLAIF needs a rebrand](https://www.interconnects.ai/p/beyond-human-data-rlaif)  covers a new way of thinking about general RL fine-tuning of LLMs: RL from *computational feedback*  (RLCF). RLAIF is a variant of this.
3. Jun. 21: [How RLHF actually works](https://www.interconnects.ai/p/how-rlhf-works)  covers the high-level intuition about what RLHF changes in model behavior -- safety, formatting, reasoning, and more subtle things.
4. Aug. 2: [Specifying objectives in RLHF](https://www.interconnects.ai/p/specifying-objectives-in-rlhf)  covers the proxy objective problem in RLHF and why the new method Direct Preference Optimization (DPO) may not be the final solution.
5. Oct. 18: [Undoing RLHF and the brittleness of safe LLMs](https://www.interconnects.ai/p/undoing-rlhf)  covers why RLHF safety filters are not resistant during further training and how this shifts the LLM marketplace.
6. Oct. 25: [RLHF lit. review #1 and missing pieces in RLHF](https://www.interconnects.ai/p/rlhf-lit-review-1-and-missing-pieces)  covers recent papers and core themes of RL research not yet touched by RLHF.
7. Nov. 22: [RLHF progress: Scaling DPO to 70B, DPO vs PPO update, Tülu 2, Zephyr-β, meaningful evaluation, data contamination](https://www.interconnects.ai/p/rlhf-progress-scaling-dpo-to-70b)  covers empirical progress in RLHF in the second half of 2024.
8. Dec. 6: [Do we need RL for RLHF?](https://www.interconnects.ai/p/the-dpo-debate)  covers all things DPO and what it means for RLHF in the future.

And this year:

- [What is missing to reproduce the RLHF of GPT4](https://www.interconnects.ai/p/open-gpt4-limitations) ? The problems we likely won’t solve in Open RLHF this year.
- [Multimodal RLHF roundup](https://www.interconnects.ai/p/multimodal-rlhf) : The questions you should try and answer if you want to work on multimodal chat models.

### Other blogs of mine

- [Illustrating RLHF](https://huggingface.co/blog/rlhf) : The original post I learned the topic with, still a good introduction.
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama) : The full RLHF process on a specific dataset and domain.
- [Red-Teaming Large Language Models](https://huggingface.co/blog/red-teaming) : A general introduction to red-teaming.
- [What Makes a Dialog Agent Useful?](https://huggingface.co/blog/dialog-agents) : A general introduction to the difference between chat agents and instruction models.

## Other resources

- Things like `awesome-rlhf` on [GitHub](https://github.com/opendilab/awesome-RLHF) have a lot of links, but they’re not curated.
- The Assembly AI post [How RLHF Preference Model Tuning Works (And How Things May Go Wrong)](https://www.assemblyai.com/blog/how-rlhf-preference-model-tuning-works-and-how-things-may-go-wrong/) from Swyx.
- Chip Huyen’s post on [RLHF](https://huyenchip.com/2023/05/02/rlhf.html) (multiple recommendation).
- Karpathy’s “State of GPT” [section about reward models](https://youtu.be/bZQun8Y4L2A?si=pxnnS82uqgesyHmt&t=786) ( [slides](https://karpathy.ai/stateofgpt.pdf) )
- [N implementation details of RLHF](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) goes into the weeds trying to reproduce some of OpenAI’s original results in the area.

Please send me any other links you think deserve a chance to be included. I’m happy to keep updating this for a few weeks!
