---
description: The path to recent advanced AI systems has been more about building larger systems than making scientific breakthroughs.
title: Scaling up: how increasing inputs has made artificial intelligence more capable
image: https://ourworldindata.org/cdn-cgi/imagedelivery/qLq-8BTgXU8yG0N6HnOy8g/ded84c3a-78c9-40c4-ace2-6bf8dd7c4f00/public
---

[Home](https://ourworldindata.org/)[Artificial Intelligence](https://ourworldindata.org/artificial-intelligence)

For most of Artificial Intelligence’s (AI’s) history, many researchers expected that building truly capable systems would need a long series of scientific breakthroughs: revolutionary algorithms, deep insights into human cognition, or fundamental advances in our understanding of the brain. While scientific advances have played a role, recent AI progress has revealed an unexpected insight: a lot of the recent improvement in AI capabilities has come simply from scaling up existing AI systems.[1](#note-1)

Here, scaling means deploying more computational power, using larger datasets, and building bigger models. This approach has worked surprisingly well so far.[2](#note-2) Just a few years ago, state-of-the-art AI systems struggled with basic tasks like counting.[3](#note-3)[4](#note-4) Today, they can [solve complex math problems,](https://ourworldindata.org/grapher/test-scores-ai-capabilities-relative-human-performance) write software, create extremely realistic images and videos, and discuss academic topics.

This article will provide a brief overview of scaling in AI over the past years. The data comes from [Epoch AI](https://epochai.org/), an organization that analyzes trends in computing, data, and investments to understand where AI might be headed.[5](#note-5) Epoch AI maintains the most extensive dataset on AI models and regularly publishes [key figures](https://epochai.org/trends) on AI growth and change.

# What is scaling in AI models?[](#what-is-scaling-in-ai-models)

Let’s briefly break down what scaling means in AI. Scaling is about increasing three main things during training, which typically need to grow together:

* The amount of data used for training the AI;
* The model’s size, measured in “parameters”;
* Computational resources, often called "compute" in AI.

The idea is simple but powerful: bigger AI systems, trained on more data and using more computational resources, tend to perform better. Even without substantial changes to the algorithms, this approach often leads to better performance across many tasks.[6](#note-6)

Here is another reason why this is important: as researchers scale up these AI systems, they not only [improve](https://theaidigest.org/progress-and-dangers) in the tasks they were trained on but can sometimes lead them to develop new abilities that they did not have on a smaller scale.[7](#note-7) For example, language models initially struggled with simple arithmetic tests like three-digit addition, but larger models could handle these easily once they reached a certain size.[8](#note-8) The transition wasn't a smooth, incremental improvement but a more abrupt leap in capabilities.

This abrupt jump in capability, rather than steady improvement, can be concerning. If, for example, models suddenly develop unexpected and potentially harmful behaviors simply as a result of getting bigger, it would be harder to anticipate and control.

This makes tracking these metrics important.

# What are the three components of scaling up AI models?[](#what-are-the-three-components-of-scaling-up-ai-models)

## Data: scaling up the training data[](#data-scaling-up-the-training-data)

One way to view today's AI models is by looking at them as very sophisticated pattern recognition systems. They work by identifying and learning from statistical regularities in the text, images, or other data on which they are trained. The more data the model has access to, the more it can learn about the nuances and complexities of the knowledge domain in which it’s designed to operate.[9](#note-9)

In 1950, Claude Shannon built one of the earliest examples of “AI”: a robotic mouse named Theseus that could "remember" its path through a maze using simple relay circuits. Each wall Theseus bumped into became a data point, allowing it to learn the correct route. The total number of walls or data points was 40\. You can find this data point in the chart; it is the first one.

While Theseus stored simple binary states in relay circuits, modern AI systems utilize vast neural networks, which can learn much more complex patterns and relationships and thus process billions of data points.

All recent notable AI models — especially large, state-of-the-art ones — rely on vast amounts of training data. With the y-axis displayed on a logarithmic scale, the chart shows that the data used to train AI models has grown exponentially. From 40 data points for Theseus to trillions of data points for the largest modern systems in a little more than seven decades.

Since 2010, the training data has doubled approximately every nine to ten months. You can see this rapid growth in the chart, shown by the purple line extending from the start of 2010 to October 2024, the latest data point as I write this article.[10](#note-10)

Datasets used for training large language models, in particular, have experienced an even faster growth rate, [tripling in size each year since 2010](https://epochai.org/trends). Large language models process text by breaking it into tokens — basic units the model can encode and understand. A token doesn't directly correspond to one word, but on average, three English words correspond to about four tokens.

GPT-2, released in 2019, is estimated to have been trained on 4 billion tokens, roughly equivalent to 3 billion words. To put this in perspective, as of September 2024, the English Wikipedia contained around 4.6 billion words.[11](#note-11) In comparison, GPT-4, released in 2023, was trained on almost 13 trillion tokens, or about 9.75 trillion words. This means that GPT-4’s training data was equivalent to over 2000 times the amount of text of the entire English Wikipedia.

As we use more data to train AI systems, we might eventually run out of high-quality human-generated materials like books, articles, and research papers. Some researchers predict we could exhaust useful training materials within the next few decades[12](#note-12). While AI models themselves can generate vast amounts of data, training AI on machine-generated materials could create problems, making the models less accurate and more repetitive.[13](#note-13)

![](https://ourworldindata.org/grapher/exponential-growth-of-datapoints-used-to-train-notable-ai-systems.png)

## Parameters: scaling up the model size[](#parameters-scaling-up-the-model-size)

Increasing the amount of training data lets AI models learn from much more information than ever before. However, to pick up on the patterns in this data and learn effectively, models need what are called "parameters". Parameters are a bit like knobs that can be tweaked to improve how the model processes information and makes predictions. As the amount of training data grows, models need more capacity to capture all the details in the training data. This means larger datasets typically require the models to have more parameters to learn effectively.

Early neural networks had hundreds or thousands of parameters. With its simple maze-learning circuitry, Theseus was a model with just 40 parameters — equivalent to the number of walls it encountered. Recent large models, such as GPT-3, boast up to 175 billion parameters.[14](#note-14) While the raw number may seem large, this roughly translates into 700 GB if stored on a disc, which is easily manageable by today’s computers.

The chart shows how the number of parameters in AI models has skyrocketed over time. Since 2010, the number of AI model parameters has approximately doubled every year. The highest estimated number of parameters recorded by Epoch AI is 1.6 trillion in the QMoE model.

While bigger AI models can do more, they also face some problems. One major issue is called "overfitting". This happens when an AI becomes “too optimized” for processing the particular data it was trained on but struggles with new data. To combat this, researchers employ two strategies: implementing specialized techniques for more generalized learning and expanding the volume and diversity of training data.

![](https://ourworldindata.org/grapher/exponential-growth-of-parameters-in-notable-ai-systems.png)

## Compute: scaling up computational resources[](#compute-scaling-up-computational-resources)

As AI models grow in data and parameters, they require exponentially more computational resources. These resources, commonly referred to as “compute” in AI research, are typically measured in total floating-point operations (“FLOP”), where each FLOP represents a single arithmetic calculation like addition or multiplication.

The computational needs for AI training have changed dramatically over time. With their modest data and parameter counts, early models could be trained in hours on simple hardware. Today’s most advanced models require [hundreds of days](https://epochai.org/data/notable-ai-models) of continuous computations, even with tens of thousands of special-purpose computers.

The chart shows that the computation used to train each AI model — shown on the vertical axis — has consistently and exponentially increased over the last few decades. From 1950 to 2010, compute doubled roughly every two years. However, since 2010, this growth has accelerated dramatically, now doubling approximately every six months, with the most compute-intensive model reaching 50 billion petaFLOP as I write this article.[15](#note-15)

To put this scale in perspective, a single high-end graphics card like the NVIDIA GeForce RTX 3090 — widely used in AI research — running at full capacity for an entire year would complete just [1.1 million petaFLOP computations](https://epochai.org/data/notable-ai-models-documentation#estimation). 50 billion petaFLOP is approximately 45,455 times more than that.

![](https://ourworldindata.org/grapher/exponential-growth-of-computation-in-the-training-of-notable-ai-systems.png)

# Compute, data, and parameters tend to scale at the same time[](#compute-data-and-parameters-tend-to-scale-at-the-same-time)

Compute, data, and parameters are closely interconnected when it comes to scaling AI models. When AI models are trained on more data, there are more things to learn. To deal with the increasing complexity of the data, AI models, therefore, require more parameters to learn from the various features of the data. Adding more parameters to the model means that it needs more computational resources during training.

This interdependence means that data, parameters, and compute need to grow simultaneously. Today’s [largest public datasets](https://epochai.org/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data) are about ten times bigger than what most AI models currently use, some containing hundreds of trillions of words. But without enough compute and parameters, AI models can’t yet use these for training.

##### Subscribe to our newsletters

We send two regular newsletters so you can stay up to date on our work and receive curated highlights from across Our World in Data.

[Subscribe](https://ourworldindata.org/subscribe)

# What can we learn from these trends for the future of AI?[](#what-can-we-learn-from-these-trends-for-the-future-of-ai)

Companies are seeking large financial investments to develop and scale their AI models, with [a growing focus](https://ourworldindata.org/grapher/global-investment-in-generative-ai) on generative AI technologies. At the same time, the key hardware that is used for training — GPUs — is getting much cheaper and more powerful, with its computing speed doubling roughly every 2.5 years per dollar spent.[16](#note-16) Some organizations are also now leveraging more computational resources not just in training AI models but also during inference — the phase when models generate responses — as illustrated by [OpenAI's latest o1 model](https://openai.com/index/learning-to-reason-with-llms/).

These developments could help create more sophisticated AI technologies faster and cheaper. As companies invest more money and the necessary hardware improves, we might see significant improvements in what AI can do, including potentially unexpected new capabilities.

Because these changes could have major effects on our society, it's important that we track and understand these developments early on. To support this, we will update key metrics — such as the growth in computational resources, training data volumes, and model parameters — on a monthly basis. These updates will help monitor the rapid evolution of AI technologies and provide valuable insights into their trajectory.

#### Acknowledgments

I’d like to thank Max Roser, Daniel Bachler, Charlie Giattino, and Edouard Mathieu for their helpful comments and ideas for this article and visualizations.

### Endnotes

1. Vaswani et al. (2017). Attention is all you need. _Advances in neural information processing systems_, _30._
2. Hestness et al. (2017). Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409.
3. According to some accounts, GPT-2, a state-of-the-art language model by OpenAI at the time, was unable to reliably count to ten.
4. Bengio et al. (2023). Managing AI risks in an era of rapid progress._arXiv preprint arXiv:2310.17688._
5. Epoch AI (2023), "Key Trends and Figures in Machine Learning". Published online at epochai.org. Retrieved from: 'https://epochai.org/trends' \[online resource\].
6. Hoffmann et al. (2022). Training compute-optimal large language models. _arXiv preprint arXiv:2203.15556_.; Kaplan et al. (2020). Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_.
7. Wei et al. (2022). Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_.
8. Some [researchers](https://arxiv.org/abs/2304.15004) argue that identifying new skills in AI largely hinges on the metrics used for evaluation. As a result, unless the model shows outstanding performance in a specific task, its developing abilities may remain unrecognized before they are “perfect”, giving the impression that these skills suddenly emerged.
9. For instance, language models like GPT (Generative Pre-trained Transformer) are trained on datasets consisting of billions of words, enabling them to understand and generate human-like text.
10. The regression line for 2010 onward highlights the rapid growth driven largely by the success of deep learning methods—an approach where artificial neural networks learn and improve by analyzing vast amounts of data to identify patterns and make predictions.
11. see the [Wikipedia page of Wikipedia’s size](https://en.wikipedia.org/wiki/Wikipedia:Size%5Fof%5FWikipedia)
12. Villalobos et al. (2024). ‘Will we run out of data? Limits of LLM scaling based on human-generated data’. ArXiv \[cs.LG\]. arXiv. https://arxiv.org/abs/2211.04325
13. Shumailov et al. (2024). AI models collapse when trained on recursively generated data. Nature 631, 755–759\. https://doi.org/10.1038/s41586-024-07566-y/
14. Brown et al. (2020). Language models are few-shot learners. _Advances in neural information processing systems_, 33, 1877-1901.
15. One petaFLOP is equal to 1,000,000,000,000,000 (one quadrillion) FLOP.
16. Hobbhahn and Besiroglu (2022). Trends in GPU Price-Performance. _Published online at epochai.org_. Retrieved from: '<https://epochai.org/blog/trends-in-gpu-price-performance>' \[online resource\]

### Cite this work

Our articles and data visualizations rely on work from many different people and organizations. When citing this article, please also cite the underlying data sources. This article can be cited as:

```
Veronika Samborska (2025) - “Scaling up: how increasing inputs has made artificial intelligence more capable” Published online at OurWorldinData.org. Retrieved from: 'https://archive.ourworldindata.org/20260828-100009/scaling-up-ai.html' [Online Resource] (archived on August 28, 2026).
```

BibTeX citation

```
@article{owid-scaling-up-ai,
    author = {Veronika Samborska},
    title = {Scaling up: how increasing inputs has made artificial intelligence more capable},
    journal = {Our World in Data},
    year = {2025},
    note = {https://archive.ourworldindata.org/20260828-100009/scaling-up-ai.html}
}
```

![Our World in Data logo](https://ourworldindata.org/owid-logo.svg)

### Reuse this work freely

All visualizations, data, and articles produced by Our World in Data are completely open access under the [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You have the permission to use, distribute, and reproduce these in any medium, provided the source and authors are credited.

The data produced by third parties and made available by Our World in Data is subject to the license terms from the original third-party authors. We will always indicate the original source of the data in our documentation, so you should always check the license of any such third-party data before use and redistribution.

All of [our charts can be embedded](https://ourworldindata.org/faqs#how-can-i-embed-one-of-your-interactive-charts-in-my-website) in any site.

#### Our World in Data is free and accessible for everyone.

Help us do this work by making a donation.

[Donate now](https://ourworldindata.org/donate)

```json
{"@context":"https://schema.org","@type":"Article","headline":"Scaling up: how increasing inputs has made artificial intelligence more capable","description":"The path to recent advanced AI systems has been more about building larger systems than making scientific breakthroughs.","image":["https://ourworldindata.org/cdn-cgi/imagedelivery/qLq-8BTgXU8yG0N6HnOy8g/ded84c3a-78c9-40c4-ace2-6bf8dd7c4f00/public"],"mainEntityOfPage":{"@type":"WebPage","@id":"https://ourworldindata.org/scaling-up-ai"},"datePublished":"2025-01-20T04:00:00.000Z","author":[{"@type":"Person","name":"Veronika Samborska","url":"https://ourworldindata.org/team/veronika-samborska"}],"publisher":{"@type":"Organization","name":"Our World in Data","url":"https://ourworldindata.org","logo":"https://ourworldindata.org/owid-logo.png"}}
```
