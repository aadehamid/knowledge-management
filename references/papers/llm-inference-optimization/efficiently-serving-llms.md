title: Efficiently Serving LLMs
description: Understand how LLMs predict the next token and how techniques like KV caching can speed up text generation. Write code to serve LLM applications efficiently to multiple users.

# Efficiently Serving LLMs

Join our new short course, **Efficiently Serving Large Language Models**, to build a ground-up understanding of how to serve LLM applications from Travis Addair, CTO at Predibase. Whether you’re ready to launch your own application or just getting started building it, the topics you’ll explore in this course will deepen your foundational knowledge of how LLMs work, and help you better understand the performance trade-offs you must consider when building LLM applications that will serve large numbers of users.

You’ll walk through the most important optimizations that allow LLM vendors to efficiently serve models to many customers, including strategies for working with multiple fine-tuned models at once. In this course, you will:

- Learn how auto-regressive large language models generate text one token at a time
- Implement the foundational elements of a modern LLM inference stack in code, including KV caching, continuous batching, and model quantization, and benchmark their impacts on inference throughput and latency.
- Explore the details of how LoRA adapters work, and learn how batching techniques allow different LoRA adapters to be served to multiple customers simultaneously.
- Get hands-on with Predibase’s LoRAX framework inference server to see these optimization techniques implemented in a real world LLM inference server.

Knowing more about how LLM servers operate under the hood will greatly enhance your understanding of the options you have to increase the performance and efficiency of your LLM-powered applications.
