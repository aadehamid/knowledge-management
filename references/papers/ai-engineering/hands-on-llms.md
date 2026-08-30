title: GitHub - iusztinpaul/hands-on-llms: 🦖 𝗟𝗲𝗮𝗿𝗻 about 𝗟𝗟𝗠𝘀, 𝗟𝗟𝗠𝗢𝗽𝘀, and 𝘃𝗲𝗰𝘁𝗼𝗿 𝗗𝗕𝘀 for free by designing, training, and deploying a real-time financial advisor LLM system \~ 𝘴𝘰𝘶𝘳𝘤𝘦 𝘤𝘰𝘥𝘦 + 𝘷𝘪𝘥𝘦𝘰 & 𝘳𝘦𝘢𝘥𝘪𝘯𝘨 𝘮𝘢𝘵𝘦𝘳𝘪𝘢𝘭𝘴
description: 🦖 𝗟𝗲𝗮𝗿𝗻 about 𝗟𝗟𝗠𝘀, 𝗟𝗟𝗠𝗢𝗽𝘀, and 𝘃𝗲𝗰𝘁𝗼𝗿 𝗗𝗕𝘀 for free by designing, training, and deploying a real-time financial advisor LLM system \~ 𝘴𝘰𝘶𝘳𝘤𝘦 𝘤𝘰𝘥𝘦 + 𝘷𝘪𝘥𝘦𝘰 &amp; 𝘳𝘦𝘢𝘥𝘪𝘯𝘨 𝘮𝘢𝘵𝘦𝘳𝘪𝘢𝘭𝘴 - iusztinpaul/hands-...

# GitHub - iusztinpaul/hands-on-llms: 🦖 𝗟𝗲𝗮𝗿𝗻 about 𝗟𝗟𝗠𝘀, 𝗟𝗟𝗠𝗢𝗽𝘀, and 𝘃𝗲𝗰𝘁𝗼𝗿 𝗗𝗕𝘀 for free by designing, training, and deploying a real-time financial advisor LLM system \~ 𝘴𝘰𝘶𝘳𝘤𝘦 𝘤𝘰𝘥𝘦 \+ 𝘷𝘪𝘥𝘦𝘰 & 𝘳𝘦𝘢𝘥𝘪𝘯𝘨 𝘮𝘢𝘵𝘦𝘳𝘪𝘢𝘭𝘴

As the world of GenAI and LLMs moves fast, too fast for educational content, it was easier to archive this course and create a new one from scratch.

Check out our new [LLM Twin open-source course](https://github.com/decodingml/llm-twin-course) for an improved experience in learning to build a production-ready LLM and RAG system.

---

- [1. Building Blocks](#1-building-blocks)
    - [1.1. Training Pipeline](#11-training-pipeline)
    - [1.2. Streaming Real-time Pipeline](#12-streaming-real-time-pipeline)
    - [1.3. Inference Pipeline](#13-inference-pipeline)
    - [1.4. Financial Q&A Dataset](#14-financial-qa-dataset)
- [2. Setup External Services](#2-setup-external-services)
    - [2.1. Alpaca](#21-alpaca)
    - [2.2. Qdrant](#22-qdrant)
    - [2.3. Comet ML](#23-comet-ml)
    - [2.4. Beam](#24-beam)
    - [2.5. AWS](#25-aws)
- [3. Install & Usage](#3-install--usage)
- [4. Lectures](#4-lectures)
    - [4.1. Costs](#41-costs)
    - [4.2. Ask Questions](#42-ask-questions)
    - [4.3. Video lectures](#43-video-lectures)
    - [4.4. Articles](#44-articles)
- [5. License](#6-license)
- [6. Contributors & Teachers](#7-contributors--teachers)

---

*Using the 3-pipeline design, this is what you will learn to build within this course* ↓

Training pipeline that:

- loads a proprietary Q&A dataset
- fine-tunes an open-source LLM using QLoRA
- logs the training experiments on [Comet ML's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) experiment tracker & the inference results on [Comet ML's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) LLMOps dashboard
- stores the best model on [Comet ML's](https://www.comet.com/site/products/llmops/?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry

The **training pipeline** is **deployed** using [Beam](https://docs.beam.cloud/getting-started/quickstart?utm_source=thepauls&utm_medium=partner&utm_content=github) as a serverless GPU infrastructure.

-> Found under the `modules/training_pipeline` directory.

- CPU: 4 Cores
- RAM: 14 GiB
- VRAM: 10 GiB (mandatory CUDA-enabled Nvidia GPU)

**Note:** Do not worry if you don't have the minimum hardware requirements. We will show you how to deploy the training pipeline to [Beam's](https://docs.beam.cloud/getting-started/quickstart?utm_source=thepauls&utm_medium=partner&utm_content=github) serverless infrastructure and train the LLM there.

Real-time feature pipeline that:

- ingests financial news from [Alpaca](https://alpaca.markets/docs/api-references/market-data-api/news-data/)
- cleans & transforms the news documents into embeddings in real-time using [Bytewax](https://github.com/bytewax/bytewax?utm_source=thepauls&utm_medium=partner&utm_content=github)
- stores the embeddings into the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github)

The **streaming pipeline** is **automatically deployed** on an AWS EC2 machine using a CI/CD pipeline built in GitHub actions.

-> Found under the `modules/streaming_pipeline` directory.

- CPU: 1 Core
- RAM: 2 GiB
- VRAM: -

Inference pipeline that uses [LangChain](https://github.com/langchain-ai/langchain) to create a chain that:

- downloads the fine-tuned model from [Comet's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry
- takes user questions as input
- queries the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) and enhances the prompt with related financial news
- calls the fine-tuned LLM for financial advice using the initial query, the context from the vector DB, and the chat history
- persists the chat history into memory
- logs the prompt & answer into [Comet ML's](https://www.comet.com/site/products/llmops/?utm_source=thepauls&utm_medium=partner&utm_content=github) LLMOps monitoring feature

The **inference pipeline** is **deployed** using [Beam](https://docs.beam.cloud/deployment/rest-api?utm_source=thepauls&utm_medium=partner&utm_content=github) as a serverless GPU infrastructure, as a RESTful API. Also, it is wrapped under a UI for demo purposes, implemented in [Gradio](https://www.gradio.app/).

-> Found under the `modules/financial_bot` directory.

- CPU: 4 Cores
- RAM: 14 GiB
- VRAM: 8 GiB (mandatory CUDA-enabled Nvidia GPU)

**Note:** Do not worry if you don't have the minimum hardware requirements. We will show you how to deploy the inference pipeline to [Beam's](https://docs.beam.cloud/getting-started/quickstart?utm_source=thepauls&utm_medium=partner&utm_content=github) serverless infrastructure and call the LLM from there.

[![architecture](https://github.com/iusztinpaul/hands-on-llms/raw/main/media/architecture.png)](https://github.com/iusztinpaul/hands-on-llms/blob/main/media/architecture.png)

We used `GPT3.5` to generate a financial Q&A dataset to fine-tune our open-source LLM to specialize in using financial terms and answering financial questions. Using a large LLM, such as `GPT3.5` to generate a dataset that trains a smaller LLM (e.g., Falcon 7B) is known as **fine-tuning with distillation**.

→ To understand how we generated the financial Q&A dataset, [check out this article](https://open.substack.com/pub/paulabartabajo/p/how-to-generate-financial-q-and-a?r=1ttoeh&utm_campaign=post&utm_medium=web) written by [Pau Labarta](https://github.com/Paulescu).

→ To see a complete analysis of the financial Q&A dataset, check out the [dataset\_analysis](https://github.com/iusztinpaul/hands-on-llms/blob/main/dataset_analysis) subsection of the course written by [Alexandru Razvant](https://github.com/Joywalker).

[![EDA](https://github.com/iusztinpaul/hands-on-llms/raw/main/media/eda_prompts_dataset.png)](https://github.com/iusztinpaul/hands-on-llms/blob/main/media/eda_prompts_dataset.png)

Before diving into the modules, you have to set up a couple of additional external tools for the course.

**NOTE:** You can set them up as you go for every module, as we will point you in every module what you need.

`financial news data source`

Follow this [document](https://alpaca.markets/docs/market-data/getting-started/) to show you how to create a FREE account and generate the API Keys you will need within this course.

**Note:** 1x Alpaca data connection is FREE.

`serverless vector DB`

Go to [Qdrant](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) and create a FREE account.

After, follow [this document](https://qdrant.tech/documentation/cloud/authentication/?utm_source=thepauls&utm_medium=partner&utm_content=github) on how to generate the API Keys you will need within this course.

**Note:** We will use only Qdrant's freemium plan.

`serverless ML platform`

Go to [Comet ML](https://www.comet.com/signup?utm_source=thepauls&utm_medium=partner&utm_content=github) and create a FREE account.

After, [follow this guide](https://www.comet.com/docs/v2/guides/getting-started/quickstart/) to generate an API KEY and a new project, which you will need within the course.

**Note:** We will use only Comet ML's freemium plan.

`serverless GPU compute | training & inference pipelines`

Go to [Beam](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github) and create a FREE account.

After, you must follow their [installation guide](https://docs.beam.cloud/getting-started/installation?utm_source=thepauls&utm_medium=partner&utm_content=github) to install their CLI & configure it with your Beam credentials.

To read more about Beam, here is an [introduction guide](https://docs.beam.cloud/getting-started/introduction?utm_source=thepauls&utm_medium=partner&utm_content=github).

**Note:** You have \~10 free compute hours. Afterward, you pay only for what you use. If you have an Nvidia GPU >8 GB VRAM & don't want to deploy the training & inference pipelines, using Beam is optional.

When using Poetry, we had issues locating the Beam CLI inside a Poetry virtual environment. To fix this, after installing Beam, we create a symlink that points to Poetry's binaries, as follows:

```
export COURSE_MODULE_PATH=<your-course-module-path> # e.g., modules/training_pipeline
 cd $COURSE_MODULE_PATH
 export POETRY_ENV_PATH=$(dirname $(dirname $(poetry run which python)))

 ln -s /usr/local/bin/beam ${POETRY_ENV_PATH}/bin/beam
```

`cloud compute | feature pipeline`

After, download and install their [AWS CLI v2.11.22](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configure it](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with your credentials.

**Note:** You will pay only for what you use. You will deploy only a `t2.small` EC2 VM, which is only `~$0.023` / hour. If you don't want to deploy the feature pipeline, using AWS is optional.

Every module has its dependencies and scripts. In a production setup, every module would have its repository, but in this use case, for learning purposes, we put everything in one place:

Thus, check out the README for every module individually to see how to install & use it:

1. [q\_and\_a\_dataset\_generator](https://github.com/iusztinpaul/hands-on-llms/blob/main/modules/q_and_a_dataset_generator)
2. [training\_pipeline](https://github.com/iusztinpaul/hands-on-llms/blob/main/modules/training_pipeline)
3. [streaming\_pipeline](https://github.com/iusztinpaul/hands-on-llms/blob/main/modules/streaming_pipeline)
4. [inference\_pipeline](https://github.com/iusztinpaul/hands-on-llms/blob/main/modules/financial_bot)

We strongly encourage you to clone this repository and replicate everything we've done to get the most out of this course.

In each module's video lectures, articles, and README documentation, you will find step-by-step instructions.

Happy learning! 🙏

The GitHub code (released under the MIT license) and video lectures (released on YouTube) are entirely free of charge. Always will be.

The Medium lessons are released under Medium's paid wall. If you already have it, then they are free. Otherwise, you must pay a $5 monthly fee to read the articles.

If you have any questions or issues during the course, we encourage you to create an [issue](https://github.com/iusztinpaul/hands-on-llms/issues) in this repository where you can explain everything you need in depth.

Otherwise, you can also contact the teachers on LinkedIn:

- [Paul Iusztin](https://www.linkedin.com/in/pauliusztin/)
- [Pau Labarta](https://www.linkedin.com/in/pau-labarta-bajo-4432074b/)

`To understand the entire code step-by-step, check out our articles` ↓

- [Lesson 1: The LLMs Kit: Build a Production-Ready Real-Time Financial Advisor System Using Streaming Pipelines, RAG, and LLMOps](https://medium.com/decodingml/the-llms-kit-build-a-production-ready-real-time-financial-advisor-system-using-streaming-ffdcb2b50714)

- [Lesson 2: Why you must choose streaming over batch pipelines when doing RAG in LLM applications](https://medium.com/decoding-ml/why-you-must-choose-streaming-over-batch-pipelines-when-doing-rag-in-llm-applications-3b6fd32a93ff)
- [Lesson 3: This is how you can build & deploy a streaming pipeline to populate a vector DB for real-time RAG](https://medium.com/decodingml/this-is-how-you-can-build-deploy-a-streaming-pipeline-to-populate-a-vector-db-for-real-time-rag-c92cfbbd4d62)

- [Lesson 4: 5 concepts that must be in your LLM fine-tuning kit](https://medium.com/decodingml/5-concepts-that-must-be-in-your-llm-fine-tuning-kit-59183c7ce60e)
- [Lesson 5: The secret of writing generic code to fine-tune any LLM using QLoRA](https://medium.com/decodingml/the-secret-of-writing-generic-code-to-fine-tune-any-llm-using-qlora-9b1822f3c6a4)
- [Lesson 6: From LLM development to continuous training pipelines using LLMOps](https://medium.com/decodingml/from-llm-development-to-continuous-training-pipelines-using-llmops-a3792b05061c)

- [Lesson 7: Design a RAG LangChain application leveraging the 3-pipeline architecture](https://medium.com/decodingml/design-a-rag-langchain-application-leveraging-the-3-pipeline-architecture-46bcc3cb3500)
- [Lesson 8: Prepare your RAG LangChain application for production](https://medium.com/decodingml/prepare-your-rag-langchain-application-for-production-5f75021cd381)

This course is an open-source project released under the MIT license. Thus, as long you distribute our LICENSE and acknowledge our work, you can safely clone or fork this project and use it as a source of inspiration for whatever you want (e.g., university projects, college degree projects, etc.).
