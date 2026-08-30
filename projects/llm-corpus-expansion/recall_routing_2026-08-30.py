# Routing for the 64 Recall-only sources. Titles for older cards come from the
# Recall metadata queries; each line is url | title | source_type | subject.
ROUTES = """
https://www.aleksagordic.com/blog/matmul|Inside NVIDIA GPUs: Anatomy of high performance matmul kernels|blog|cuda
https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html|GPU Programming: Registers, Global, and Local Memory|doc|cuda
https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_types|Cornell Virtual Workshop: GPU Memory Types|doc|cuda
https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/|NVIDIA Ampere Architecture In-Depth|blog|cuda
https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/|NVIDIA Hopper Architecture In-Depth|blog|cuda
https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/|Using Shared Memory in CUDA C/C++|blog|cuda
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html|CUDA C++ Programming Guide|doc|cuda
https://enccs.github.io/gpu-programming/4-gpu-concepts/|GPU programming concepts (ENCCS)|doc|cuda
https://michalpitr.substack.com/p/gpu-programming|GPU Programming|blog|cuda
https://michalpitr.substack.com/p/optimizing-matrix-multiplication|Optimizing matrix multiplication|blog|cuda
https://www.microway.com/hpc-tech-tips/cuda-parallel-thread-management/|CUDA Parallel Thread Management|blog|cuda
https://www.microway.com/hpc-tech-tips/gpu-memory-types-performance-comparison/|GPU Memory Types - Performance Comparison|blog|cuda
https://www.microway.com/hpc-tech-tips/parallel-code-maximizing-your-performance-potential/|Parallel Code: Maximizing your Performance Potential|blog|cuda
https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture|What is a CUDA Device Architecture? GPU Glossary|doc|cuda
https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor|What is a Streaming Multiprocessor? GPU Glossary|doc|cuda
https://people.freebsd.org/~lstewart/articles/cpumemory.pdf|What Every Programmer Should Know About Memory|pdf|cuda
https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/|In Pursuit of High-Fidelity GPU Kernel Benchmarking|blog|cuda
https://www.georgeho.org/floating-point-deep-learning/|Floating-Point Formats and Deep Learning|blog|cuda
https://www.youtube.com/watch?v=CKmNpAO5rS4|Stanford CS149 Parallel Computing Lecture 2 - A Modern Multi-Core Processor|video|cuda
https://www.youtube.com/watch?v=F4bVSyz_jxo|Stanford CS149 Lecture 3 - Multi-core Arch Part II + ISPC Programming Abstractions|video|cuda
https://www.youtube.com/watch?v=LMk8nqIFXLo|ScaleML Series Day 5 - GPU Programming for Foundation Models|video|cuda
https://www.youtube.com/watch?v=h9Z4oGN89MU|How do Graphics Cards Work? Exploring GPU Architecture|video|cuda
https://www.youtube.com/watch?v=uyzqxIoiobU|Ex-NVIDIA Engineer: Why AI Is About to Get 1000x Cheaper|video|cuda
https://github.com/amitshekhariitbhu/llm-inference-engineering|Learn LLM Inference Engineering step by step - KV cache, PagedAttention, continuous batching, vLLM, SGLang|code|llm-inference-optimization
https://github.com/cfregly/ai-performance-engineering|cfregly/ai-performance-engineering - GPU optimization, distributed training, inference scaling|code|llm-inference-optimization
https://github.com/videlalvaro/inference-school|inference-school - a hands-on Swift and Metal course for building LLM inference on Apple silicon|code|llm-inference-optimization
https://livebook.manning.com/book/quantization-and-fast-inference/chapter-1/v-1#28|Quantization and Fast Inference, Chapter 1: Facing the Efficiency Wall|doc|llm-inference-optimization
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization|A Visual Guide to Quantization|blog|llm-inference-optimization
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/|Defeating Nondeterminism in LLM Inference|blog|llm-inference-optimization
https://modal.com/notebooks/modal-labs/charles-dev/nb-x2wXrLH7aqi7HGVQ8Fosh2|Inside vLLM - Anatomy of a High-Throughput LLM Inference System (Modal Notebook)|doc|llm-inference-optimization
https://www.youtube.com/watch?v=3TBT4WPkDaw|Inference, Serving, PagedAttention and vLLM|video|llm-inference-optimization
https://alexstrick.com/posts/2024-06-15-isafpr-first-finetune.html|Finetuning my first LLM(s) for structured data extraction with axolotl|blog|llm-finetuning
https://aweers.de/blog/2026/rl-for-llms/|State of RL for reasoning LLMs|blog|llm-finetuning
https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers|Fine-tuning with gpt-oss and Hugging Face Transformers|doc|llm-finetuning
https://docs.mistral.ai/resources/deprecated/finetuning|Mistral fine-tuning docs|doc|llm-finetuning
https://github.com/unslothai/unsloth|unslothai/unsloth - local UI to run and train LLMs and diffusion models|code|llm-finetuning
https://huggingface.co/blog/hf-skills-training|We Got Claude to Fine-Tune an Open Source LLM|blog|llm-finetuning
https://rlhfbook.com/|RLHF Book: Reinforcement Learning from Human Feedback and LLM Post-Training|doc|llm-finetuning
https://rlhfbook.com/c/06-policy-gradients|RLHF Book chapter 6: Policy Gradients|doc|llm-finetuning
https://www.philschmid.de/fine-tune-llms-in-2024-with-trl|How to Fine-Tune LLMs in 2024 with Hugging Face|blog|llm-finetuning
https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune|Qwen3 - How to Run and Fine-tune|doc|llm-finetuning
https://www.youtube.com/watch?v=pov3pLFMOPY|QLoRA: Quantization for Fine Tuning|video|llm-finetuning
https://huggingface.co/blog/moe|Mixture of Experts Explained|blog|transformers
https://sander.ai/2025/04/15/latents.html|Generative modelling in latent space|blog|transformers
https://mojolang.org/docs/manual/values/ownership/|Ownership - Mojo manual|doc|transformers
https://dataorigami.net/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/|Bayesian Methods for Hackers|doc|ml-foundations
https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework|Build a Neural Network Framework|course|ml-foundations
https://brandonrohrer.com/blog.html|Brandon Rohrer's blog|blog|ml-foundations
https://www.jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/|Markov Chain Monte Carlo Without all the Bullshit|blog|ml-foundations
https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-27-backpropagation-find-partial-derivatives/index.html|MIT 18.065 Lecture 27: Backpropagation - Find Partial Derivatives|video|ml-foundations
https://theorydish.blog/2021/12/16/backpropagation-≠-chain-rule/|Backpropagation is not the Chain Rule|blog|ml-foundations
https://towardsdatascience.com/comparing-performance-of-big-data-file-formats-a-practical-guide-ef366561b7d2/|Comparing Performance of Big Data File Formats: A Practical Guide|blog|ml-foundations
https://developer.ibm.com/articles/awb-token-optimization-backbone-of-effective-prompt-engineering/|Token optimization: the backbone of effective prompt engineering|blog|ai-engineering
https://jordivillar.com/blog/sql-is-all-you-need|SQL Is All You Need|blog|ai-engineering
https://www.llmstxt.new/|Generate llms.txt|doc|ai-engineering
https://slack.engineering/rebuilding-slack-com/|Rebuilding slack.com|blog|ai-engineering
https://huggingface.co/docling-project/SmolDocling-256M-preview|SmolDocling-256M-preview|blog|document-ai-retrieval
https://huggingface.co/PaddlePaddle/PaddleOCR-VL|PaddlePaddle/PaddleOCR-VL|blog|document-ai-retrieval
https://arxiv.org/pdf/2005.11401|Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks|pdf|document-ai-retrieval
https://buttondown.com/ainews/archive/ainews-cohere-command-r-anthropic-claude-tool-use/|AINews: Cohere Command R+, Anthropic Claude Tool Use, OpenAI Finetuning|blog|llm-landscape
https://x.ai/news/grok-os|Open Release of Grok-1|blog|llm-landscape
https://flashcards.dwarkesh.com/|Dwarkesh Podcast Flashcards|doc|llm-landscape
"""

# Dropped: no retrievable content, or already held under a pinned permalink.
DROPPED = {
 "https://authn.edx.org/login": "edX auth wall",
 "https://learning.edx.org/course/course-v1:MITx+6.008.1x+3T2016/home": "edX auth wall",
 "https://scs.hosted.panopto.com/Panopto/Pages/Auth/Login.aspx": "Panopto login page",
 "https://job-boards.greenhouse.io/anthropic?error=true": "job board, no knowledge content",
 "https://www.innerworkings.ai/login": "login page",
 "https://brilliant.org/": "site homepage, no article",
 "https://www.oreilly.com/library/view/pretrain-vision-and/9781804618257/B18942_TOC_ePub.xhtml": "paywalled TOC, already dropped once",
 "https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/llama-2/tiny-llama.yml": "404 on main; recovered at a pinned commit already in the corpus",
 "https://github.com/meta-llama/llama-cookbook/blob/main/examples/Prompt_Engineering_with_Llama_2.ipynb": "404 on main; recovered at a pinned commit already in the corpus",
 "https://www.youtube.com/watch?v=2s81mlFtHio": "Recall product tutorial, not subject material",
 "https://www.youtube.com/watch?v=7EJjdDLK4cg": "already in resources/sources/llm-inference-optimization/urls.txt",
 "https://vllm.ai/blog/2025-09-05-anatomy-of-vllm": "same article as aleksagordic.com/blog/vllm, already held",
 "https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm": "already held as blog.vllm.ai",
 "https://v0.app/": "already held as v0.dev",
 "https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf": "PDF of the CUDA guide already routed as HTML",
 # Out of scope for every vault in this knowledge base.
 "https://communityovercode.org/wp-content/uploads/2023/10/mon_dataeng_building-a-semantic-metrics-layer-using-calcite-julian-hyde.pdf": "data-engineering semantic layer, no vault covers it",
 "https://cube.dev/blog/universal-semantic-layer-capabilities-integrations-and-enterprise-benefits": "data-engineering semantic layer, no vault covers it",
 "https://motherduck.com/blog/semantic-layer-duckdb-tutorial/": "data-engineering semantic layer, no vault covers it",
 "https://www.ssp.sh/blog/rise-of-semantic-layer-metrics/": "data-engineering semantic layer, no vault covers it",
}
