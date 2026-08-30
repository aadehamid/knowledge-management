title: Logics-MLLM/Logics-Parsing · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

# Logics-MLLM/Logics-Parsing · Hugging Face

![](https://huggingface.co/Logics-MLLM/Logics-Parsing/resolve/main/imgs/logo.jpg){width=80%}

 🤗 [GitHub](https://github.com/alibaba/Logics-Parsing)   |   🤖 [Demo](https://www.modelscope.cn/studios/Alibaba-DT/Logics-Parsing/summary)   |   📑 [Technical Report](https://arxiv.org/abs/2509.19760) 

##  Introduction 

![LogicsDocBench 概览](https://huggingface.co/Logics-MLLM/Logics-Parsing/resolve/main/imgs/overview.png)

Logics-Parsing is a powerful, end-to-end document parsing model built upon a general Vision-Language Model (VLM) through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). It excels at accurately analyzing and structuring highly complex documents.

##  Key Features 

- **Effortless End-to-End Processing**
    - Our single-model architecture eliminates the need for complex, multi-stage pipelines. Deployment and inference are straightforward, going directly from a document image to structured output.
    - It demonstrates exceptional performance on documents with challenging layouts.
- **Advanced Content Recognition**
    - It accurately recognizes and structures difficult content, including intricate scientific formulas.
    - Chemical structures are intelligently identified and can be represented in the standard **SMILES** format.
- **Rich, Structured HTML Output**
    - The model generates a clean HTML representation of the document, preserving its logical structure.
    - Each content block (e.g., paragraph, table, figure, formula) is tagged with its **category**, **bounding box coordinates**, and **OCR text**.
    - It automatically identifies and filters out irrelevant elements like headers and footers, focusing only on the core content.
- **State-of-the-Art Performance**
    - Logics-Parsing achieves the best performance on our in-house benchmark, which is specifically designed to comprehensively evaluate a model’s parsing capability on complex-layout documents and STEM content.

##  Benchmark 

Existing document-parsing benchmarks often provide limited coverage of complex layouts and STEM content. To address this, we constructed an in-house benchmark comprising 1,078 page-level images across nine major categories and over twenty sub-categories. Our model achieves the best performance on this benchmark.

![](https://huggingface.co/Logics-MLLM/Logics-Parsing/resolve/main/imgs/BenchCls.png)

Edit

Edit

Edit

TEDS

Edit

Edit

Edit

Edit

0.148

0.115

86.3

0.113

0.118

0.104

\*

0.128

0.146

0.152

0.06

0.142

86.2

86.6

0.120

0.115

0.255

Logics-Parsing

0.124

0.145

0.089

0.139

0.106

0.165

0.136

0.113

0.519

0.252

0.115

82.6

0.535

\*

##  Quick Start 

###  1. Installation 

```shell
conda create -n logis-parsing python=3.10
conda activate logis-parsing

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

###  2. Download Model Weights 

```
# Download our model from Modelscope.
pip install modelscope
python download_model.py -t modelscope

# Download our model from huggingface.
pip install huggingface_hub
python download_model.py -t huggingface
```

###  3. Inference 

```shell
python3 inference.py --image_path PATH_TO_INPUT_IMG --output_path PATH_TO_OUTPUT --model_path PATH_TO_MODEL
```

##  Acknowledgments 

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Mathpix](https://mathpix.com/)
