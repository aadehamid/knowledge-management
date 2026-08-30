title: Distilabel Docs
description: Distilabel is an AI Feedback \(AIF\) framework for building datasets with and for LLMs.
author: Argilla, Inc.

# Distilabel

### Synthesize data for AI and add feedback on the fly!

 [ ![CI](https://img.shields.io/pypi/v/distilabel.svg?style=flat-round&logo=pypi&logoColor=white) ](https://pypi.org/project/distilabel/) [ ![CI](https://static.pepy.tech/personalized-badge/distilabel?period=month&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads/month) ](https://pepy.tech/project/distilabel) 

 [ ![](https://img.shields.io/badge/twitter-black?logo=x) ](https://twitter.com/argilla_io) [ ![](https://img.shields.io/badge/linkedin-blue?logo=linkedin) ](https://www.linkedin.com/company/argilla-io) [ ![](https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white) ](http://hf.co/join/discord) 

Distilabel is the framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers.

- **Get started in 5 minutes!**
    ---
    Install distilabel with `pip` and run your first `Pipeline` to generate and evaluate synthetic data.
    [ Quickstart](https://distilabel.argilla.io/latest/sections/getting_started/quickstart/)
- **How-to guides**
    ---
    Get familiar with the basics of distilabel. Learn how to define `steps`, `tasks` and `llms` and run your `Pipeline`.

## Why use distilabel? {#why-use-distilabel}

Distilabel can be used for generating synthetic data and AI feedback for a wide variety of projects including traditional predictive NLP (classification, extraction, etc.), or generative and large language model scenarios (instruction following, dialogue generation, judging etc.). Distilabel's programmatic approach allows you to build scalable pipelines for data generation and AI feedback. The goal of distilabel is to accelerate your AI development by quickly generating high-quality, diverse datasets based on verified research methodologies for generating and judging with AI feedback.

Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time **achieving and keeping high-quality standards for your synthetic data**.

Take control of your data and models

**Ownership of data for fine-tuning your own LLMs** is not easy but distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

Improve efficiency by quickly iterating on the right data and models

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## What do people build with distilabel? {#what-do-people-build-with-distilabel}

The Argilla community uses distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel).

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of \~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data on an immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B), show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.
