title: Paper page - RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture
description: Join the discussion on this paper page

# RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture

Authors:

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

 ,

## Abstract

The study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) and fine-tuning pipelines for adapting Large Language Models (LLMs) to agricultural datasets, showing significant improvements in accuracy and cross-geographical knowledge integration.

There are two common ways in which developers are incorporating proprietary and domain-specific data when building applications of Large Language Models (LLMs): [Retrieval-Augmented Generation (RAG)](https://huggingface.co/papers?q=Retrieval-Augmented%20Generation%20%28RAG%29) and [Fine-Tuning](https://huggingface.co/papers?q=Fine-Tuning). RAG augments the prompt with the external data, while [fine-Tuning](https://huggingface.co/papers?q=fine-Tuning) incorporates the additional knowledge into the model itself. However, the pros and cons of both approaches are not well understood. In this paper, we propose a pipeline for [fine-tuning](https://huggingface.co/papers?q=fine-tuning) and RAG, and present the tradeoffs of both for multiple popular LLMs, including [Llama2-13B](https://huggingface.co/papers?q=Llama2-13B), [GPT-3.5](https://huggingface.co/papers?q=GPT-3.5), and [GPT-4](https://huggingface.co/papers?q=GPT-4). Our pipeline consists of multiple stages, including extracting information from [PDFs](https://huggingface.co/papers?q=PDFs), generating questions and answers, using them for [fine-tuning](https://huggingface.co/papers?q=fine-tuning), and leveraging [GPT-4](https://huggingface.co/papers?q=GPT-4) for evaluating the results. We propose metrics to assess the performance of different stages of the RAG and [fine-Tuning](https://huggingface.co/papers?q=fine-Tuning) pipeline. We conduct an in-depth study on an agricultural dataset. Agriculture as an industry has not seen much penetration of AI, and we study a potentially disruptive application - what if we could provide location-specific insights to a farmer? Our results show the effectiveness of our dataset generation pipeline in capturing [geographic-specific knowledge](https://huggingface.co/papers?q=geographic-specific%20knowledge), and the quantitative and qualitative benefits of RAG and [fine-tuning](https://huggingface.co/papers?q=fine-tuning). We see an [accuracy](https://huggingface.co/papers?q=accuracy) increase of over 6 p.p. when [fine-tuning](https://huggingface.co/papers?q=fine-tuning) the model and this is cumulative with RAG, which increases [accuracy](https://huggingface.co/papers?q=accuracy) by 5 p.p. further. In one particular experiment, we also demonstrate that the fine-tuned model leverages information from across geographies to answer specific questions, increasing [answer similarity](https://huggingface.co/papers?q=answer%20similarity) from 47% to 72%. Overall, the results point to how systems built using LLMs can be adapted to respond and incorporate knowledge across a dimension that is critical for a specific industry, paving the way for further applications of LLMs in other industrial domains.
