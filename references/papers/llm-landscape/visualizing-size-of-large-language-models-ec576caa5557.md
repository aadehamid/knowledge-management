title: Visualizing the size of Large Language Models 💻 | by Anil George | Medium
description: Visualizing the size of Large Language Models 💻 The 3 Important factors that determine the size of a Language model are: Model Size Training Size Compute Size 🎛 Visualizing Model Size The Model …
author: Anil George

# Visualizing the size of Large Language Models 💻 {#45e4}

The 3 Important factors that determine the size of a Language model are:

1. Model Size
2. Training Size
3. Compute Size

## 🎛 Visualizing Model Size {#4a5d}

The Model Size depends on the **number of learnable parameters** in the Model.

- These parameters comprise the **weights** (and biases) linked to the individual neurons in the Model’s neural network.
- Before training, these parameters are set to random values. As the training process proceeds, they get updated to optimize the Model’s performance on a specific task.
- In the analogy of “**Dials**” and “**Knobs**” — this can be likened to adjusting the various dials in a device to tune it correctly.

Once training is complete, the final parameter values can be envisioned as cell values stuffed into a “**Giant Excel sheet**”.

### **Model Size in terms of Football Fields** ⚽ {#06e5}

- If we assume each Excel cell is (1cm x 1cm) in size.
- A single Football field-sized Excel sheet (100m x 60m) will contain approx. 60 million parameters. This is roughly equal to the number of parameters in the original [**Transformer** model](https://arxiv.org/abs/1706.03762) released in 2017.

- [**GPT-1**](https://openai.com/research/language-unsupervised), released in 2018, contained about 117 million parameters. Equivalent to an Excel sheet the size of 2 Football Fields (2FFs).
- The recent [**PALM** ](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)1, 2 (340M-540M) models from Google can be visualized as a giant Excel sheet the size of 6000–7000 Football Fields!

```
List of LLMs by Model Size and Release Year-------------------------------------------2017 - Original Transformer - 65 Million Parameters ( or 1 Football field) 2018 - GPT 1 - 117 Million Parameters ( or 2 FFs) 2019 - GPT 2 - 1500 Million Parameters ( or 20 FFs)2020 - GPT 3 - 175,000 Million Parameters ( or 2500 FFs)2021 - Gopher - 280,000 Million Parameters ( or 4000 FFs)2022 - PALM - 540,000 Million Parameters ( or 7700 FFs)
```

## 📚 Visualizing Training Size {#36df}

The Training Size depends on the number of **Tokens** in the Training dataset.

- A Token can be a word, sub-word, or character — depending on how the training text is divided into tokens (Tokenization).
- The Training dataset is split into **Batches,** tokens within each Batch are processed together before updating the Model’s parameters.
- A single pass of the entire Training dataset through the Model is called an **Epoch**.
- Most recent language models, have [Epoch \= 1](https://arxiv.org/pdf/1906.06669.pdf). Consequently, such Models “see” a Token in the Training dataset only once.

### **Training Size in terms of Library Shelves 🗄️** {#3f47}

- The Training process can be visualized as the Model “reading” words from a large Collection of Books.
- If we assume a typical Book contains \~100,000 Tokens and a typical Library shelf holds \~100 books. Each Library shelf would contain about 10 million Tokens.

- The original **Transformer** model for English to German translation used the WMT dataset with 4.5 million sentence pairs (\~100 million tokens or **10 Library shelves**).
- **GPT-1** was trained on 7000 books from the [Book Corpus](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zhu_Aligning_Books_and_ICCV_2015_paper.html) dataset (\~600 million tokens or **60 Library shelves**).
- The recent **PALM** models from Google were trained on 780 billion tokens, equivalent to **78,000 Library shelves!**

```
List of LLMs by Training Size and Release Year ---------------------------------------------2017 - Original Transformer  - 100 Million Tokens ( or 10 Library shelves) 2018 - GPT 1 - 600 Million Tokens ( or 60 shelves) 2019 - GPT 2 - 28000 Million Tokens ( or 2800 shelves)2020 - GPT 3 - 300,000 Million Tokens ( or 30,000 shelves)2021 - Gopher - 300,000 Million Tokens ( or 30,000 shelves)2022 - PALM - 780,000 Million Tokens ( or 78,000 shelves)
```

## 🧮 Visualizing Compute Size {#0113}

The Compute size depends on the number of **floating-point operations** (FLOPs), or computations required during different stages of Model Training.

```
Typical FLOPs capacity for Different Devices at FP32 precision (1GFLOP = 1 Billion FLOPs = 1E+9 FLOPs)1. 💻 A Modern mid-size laptop ~ 100 GFLOPs 2. 📱 Apple iPhone 14 Pro ~ 2000 GFLOPs 3. 🎮 Sony PlayStation 5 ~ 10000 GFLOPs 4. 🖥️ Nvidia H100 NVL GPU ~ 134,000 GFLOPs
```

Stages during Training include:

1. **Forward Pass —** Model takes a sequence of Training tokens as **Input** and makes a Prediction. (e.g., the next word in the sequence).
2. **Loss Computation —** The difference between the Predicted value and Actual value is computed through a **Loss Function**.
3. **Backpropagation and Parameter Update** — The gradient of the Loss Function is computed (via Backpropagation) and is used to update the Model parameters to minimize the Loss.
4. **Multiple Epochs** — The process of forward pass, loss computation, backpropagation, and parameter update are repeated for all batches in the training dataset across multiple ‘runs’ or **Epochs**.
5. In most Modern LLMs, Epoch is equal to 1, which means the Model processes the entire training dataset just once.

The approximate number of Computations required for the entire Training process is given by the [below thumb rule:](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)

```
Ct ~ 6.N.D where Ct = Number of Training Computations       N = Number of Model Parameters           D = Number of Training Tokens
```

Using the **6ND** formula for Training computations:

1. The original [**Transformer**](https://arxiv.org/pdf/1706.03762.pdf) model (for English to German task) would have consumed 3.9 E\+16 FLOPs for 1 Epoch, assuming 10 Epochs we get 3.9 E\+17 FLOPs. *(Equivalent to*  ***45 days***  *training time on a Midsize Laptop \= 100GFLOPs)*
2. **GPT-1** would have consumed 4.2 E\+17 FLOPs for 1 Epoch, assuming [100 Epochs](https://arxiv.org/pdf/1906.06669.pdf) we get 4.2 E\+19 FLOPs. *(Equivalent to*  ***13 years***  *training time on a Midsize Laptop)*
3. **GPT-2** would have consumed 2.5 E\+20 FLOPs for 1 Epoch , assuming [20 Epochs](https://arxiv.org/pdf/1906.06669.pdf) we get 5 E\+21 FLOPs. *(Equivalent to*  ***1600 years***  *training time on a Midsize Laptop)*
4. The more recent **PALM** model would have consumed 2.53 E\+24 FLOPs assuming Epoch \= 1. *(Equivalent to*  ***800,000 years***  *training time on a Midsize Laptop!)*

**Inference Computations:** The approx. number of computations required at Inference time is given by the below [thumb rule:](https://a16z.com/2023/04/27/navigating-the-high-cost-of-ai-compute/)

```
Ci ~ 2.N.l where Ci = Number of Inference Computations       N = Number of Model Parameters           l = Input\Output length
```

```
List of LLMs by Training Compute Size and Release Year ---------------------------------------------2017 - Original Transformer  - 3.9E+17 FLOPs   ( ~ 45 Days avg. Laptop)2018 - GPT 1 - 4.21E+19 FLOPs ( ~ 13 Years on avg. Laptop)2019 - GPT 2 - 5.04E+21 FLOPs ( ~ 1600 Years on avg. Laptop) 2020 - GPT 3 - 3.15E+23 FLOPs ( ~ 99900 Years on avg. Laptop)2021 - Gopher - 5.04E+23 FLOPs ( ~ 160000 Years on avg. Laptop)2022 - PALM - 2.53E+24 FLOPs ( ~ 800000 Years on avg. Laptop)
```

## Bonus: Visualizing GPT4 Model Size {#26af}

Below information on GPT4 architecture is based on a leaked report by [SemiAnalysis](https://www.semianalysis.com/p/gpt-4-architecture-infrastructure):

**Model Size** 🎛

- GPT4 employs a **Mixture of Experts** (MoE) model with 16 experts (111 billion parameters per expert) for a total of **\~1.8 trillion Parameters.**
- To fit GPT4 parameters in a giant Excel Sheet, it will need to be the size of **30,000 Football Fields** or **180 square kms** (more than the size of Mumbai City!)

**Training Size** 📚

- GPT4 was trained on **\~13 trillion tokens** (across Multiple Epochs).
- This is equivalent to reading all books contained within **1.3 million Library Shelves** or **650 Km** long line of Library Shelves!

**Compute Size** 🧮

- [Estimated Training](https://www.linkedin.com/pulse/supposed-leak-gpt4-architecture-alvaro-duran-tovar/) FLOPs for GPT 4 **\~ 2.15 E25 FLOPs.**
- To train GPT4 on mid-size Laptop (100GFLOPs) it will take **7 million years!**

**Estimated Training Cost** \~ $ 64 Million

- [*A100 GPU has peak FLOP*](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)  *\=*  ***312 TFLOPS***  *(for TF32, sparsity enabled)*
- [*Azure cost for A100 GPU (ND96asrA100 v4) On-Demand*](https://azure.microsoft.com/en-in/pricing/details/machine-learning/#pricing)  *\=* ***3.40$/hr*** *.*
- Estimated Minimum Training Cost (for 2.15 E25 FLOPs) \= **$64 Million.**
- This is close to [Sam Altman’s estimates on GPT4 Training Cost](https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/) \= **$100 Million.**

**Estimated Inference Cost** \~ 0.3 cents for 1000 input & output tokens

- Assumption: Prompt and Response \= 1024 tokens length
- Estimated Inference FLOPs \= [3 \* GPT3 Inference FLOPs](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) \= [3\* 350 TFLOPs](https://a16z.com/2023/04/27/navigating-the-high-cost-of-ai-compute/) \= 1000 TFLOPs (for 1024 Input & Output tokens)
- [Azure cost for A100 GPU (ND96asrA100 v4) On-Demand](https://azure.microsoft.com/en-in/pricing/details/machine-learning/#pricing) \=**3.40$/hr**.
- Estimated Inference Cost (for 1024 Input & Output tokens) \= **0.003$** or every 330 input/output pairs, will cost 1$.
