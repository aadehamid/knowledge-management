title: LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation
description: LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation
keywords: LLM, CLIP, Multimodal Learning, Vision-Language

![LLM2CLIP Logo](https://microsoft.github.io/LLM2CLIP/static/images/logo.png)

#  LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation 

##  Weiquan Huang^1\*^, Aoqi Wu^1\*^, Yifan Yang^2†‡^, Xufang Luo^2^, Yuqing Yang^2^, Usman Naseem^3^, Chunyu Wang^2^, Qi Dai^2^, Xiyang Dai^2^, Dongdong Chen^2^, Chong Luo^2^, Lili Qiu^2^, Liang Hu^1†^ 

^1^Tongji University, ^2^Microsoft Corporation, ^3^Macquarie University

^\*^Equal Contribution (work done during internship at Microsoft Research Asia)  \
^†^Corresponding authors: [yifanyang@microsoft.com](mailto:yifanyang@microsoft.com), [lianghu@tongji.edu.cn](mailto:lianghu@tongji.edu.cn)  \
^‡^Project Lead 

##  News  

-  **\[2026-01-23\]** 🎉 **LLM2CLIP received the AAAI 2026 Outstanding Paper Award!** Our work was recognized for advancing multimodal representation learning by leveraging large language models as powerful textual teachers. [ (AAAI 2026 Conference Paper Awards and Recognition) ](https://aaai.org/about-aaai/aaai-awards/aaai-conference-paper-awards-and-recognition/) 
-  **\[2025-03-25\]** 🔥 **SigLIP2 models updated with LLM2CLIP training.** The new checkpoints bring substantial improvements in short- and long-text image retrieval, as well as multilingual text–image retrieval. 
-  **\[2024-11-18\]** Our Caption-Contrastive finetuned Llama3-8B-CC released on [ HuggingFace ](https://huggingface.co/microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned), with more versions coming soon. 
-  **\[2024-11-08\]** We are training a scaled-up version with ten times the dataset. Updates include EVA ViT-E, InternVL-300M, SigCLIP-SO-400M, and more VLLM results. Stay tuned for the most powerful CLIP models. Thanks for your star! 
-  **\[2024-11-06\]** OpenAI's CLIP and EVA02's ViT models are now available on [ HuggingFace ](https://huggingface.co/collections/microsoft/llm2clip-672323a266173cfa40b32d4c). 
-  **\[2024-11-01\]** Our paper was accepted at the NeurIPS 2024 SSL Workshop! 

## Abstract

CLIP is one of the most important multimodal foundational models today, aligning visual and textual signals into a shared feature space using a simple contrastive learning loss on large-scale image-text pairs. What powers CLIP’s capabilities? The rich supervision signals provided by natural language — the carrier of human knowledge — shape a powerful cross-modal representation space. As a result, CLIP supports a variety of tasks, including zero-shot classification, detection, segmentation, and cross-modal retrieval, significantly influencing the entire multimodal domain.

However, with the rapid advancements in large language models (LLMs) like GPT-4 and LLaMA, the boundaries of language comprehension and generation are continually being pushed. This raises an intriguing question: can the capabilities of LLMs be harnessed to further improve multimodal representation learning? The potential benefits of incorporating LLMs into CLIP are clear. LLMs’ strong textual understanding can fundamentally improve CLIP’s ability to handle image captions, drastically enhancing its ability to process long and complex texts — a well-known limitation of vanilla CLIP. Moreover, LLMs are trained on a vast corpus of text, possessing open-world knowledge. This allows them to expand on caption information during training, increasing the efficiency of the learning process.

In this paper, we propose LLM2CLIP, a novel approach that embraces the power of LLMs to unlock CLIP’s potential. By fine-tuning the LLM in the caption space with contrastive learning, we extract its textual capabilities into the output embeddings, significantly improving the output layer’s textual discriminability. We then design an efficient training process where the fine-tuned LLM acts as a powerful teacher for CLIP’s visual encoder. Thanks to the LLM’s presence, we can now incorporate longer and more complex captions without being restricted by vanilla CLIP text encoder’s context window and ability limitations. Our experiments demonstrate that this approach brings substantial improvements in cross-modal tasks.

**LLM2CLIP Overview:** After applying caption contrastive fine-tuning to the LLM, the increased textual discriminability enables more effective CLIP training. We leverage the open-world knowledge and general capabilities of the LLM to better process dense captions, addressing the previous limitations of the pretrained CLIP visual encoder and providing richer, higher-dimensional textual supervision.

![Radar Chart](https://microsoft.github.io/LLM2CLIP/static/images/radar_paper.png)

**LLM2CLIP: Demonstrating excellence across multiple benchmarks.**

## Experimental Results

 To validate our hypothesis, we designed a caption-to-caption retrieval experiment (CRA). Each image in MS-COCO has five captions. We treat two captions of the same image as positives and retrieve across the whole validation split. CRA measures how well a text encoder can distinguish fine-grained caption semantics—an essential prerequisite for serving as a reliable teacher. We found that vanilla LLMs can be weak at this task, while our Caption-Contrastive (CC) fine-tuning substantially improves CRA, enabling effective supervision for CLIP-style training. 

[Top-1 Caption Retrieval Accuracy (CRA) on MS-COCO val.]
| **Text Encoder** | **CRA** |
|:--:|:--:|
| CLIP-L/14 | 25.2 |
| EVA02-L/14 | 27.11 |
| Llama3-8B | 5.2 |
| Llama3.2-1B | 5.6 |
| **Llama3-8B-CC** | **29.5** |
| Llama3.2-1B-CC | 29.4 |

##  LLM2CLIP pushes state-of-the-art CLIP models even further. 

[
                  Text-encoder ablation on image-text retrieval. Community LLM-derived text encoders can be strong, while our CC-tuned LLM encoders
                  (suffix “-CC”) provide consistent gains and strong averages across diverse benchmarks.
]
| Method | Flickr | ^^ | COCO | ^^ | ShareGPT4V | ^^ | Urban-1k | ^^ | DOCCI | ^^ | Avg | ^^ |
|:--:|:--:|----|:--:|----|:--:|----|:--:|----|:--:|----|:--:|----|
| ^^ | I2T | T2I | I2T | T2I | I2T | T2I | I2T | T2I | I2T | T2I | I2T | T2I |
| CLIP | 89.6 | 77.8 | 59.4 | 48.6 | 88.1 | 87.7 | 68.0 | 74.8 | 67.0 | 71.3 | 74.4 | 72.0 |
| Directly Finetune (50%) | 89.3 | 77.8 | 59.3 | 48.5 | 88.3 | 88.2 | 68.5 | 76.0 | 67.2 | 71.2 | 74.5 | 72.3 |
| bge-en-icl | 89.1 | 78.5 | 58.8 | 49.7 | 95.0 | 95.7 | 77.9 | 87.7 | 73.5 | 79.4 | 78.9 | 78.2 |
| LLM2Vec-Llama-3-8B | 91.1 | 81.4 | 61.5 | 51.9 | 94.5 | **96.4** | 82.1 | 88.6 | 77.7 | 82.6 | 81.4 | 80.2 |
| NV-Embed-v2 | 90.4 | 80.0 | 60.5 | 51.6 | 94.5 | 95.7 | 83.3 | **90.0** | 78.3 | 82.1 | 81.4 | 79.9 |
| bge-m3-XLM-R | 80.7 | 70.3 | 51.4 | 42.0 | 84.5 | 86.5 | 56.8 | 63.4 | 51.7 | 55.9 | 65.0 | 63.6 |
| jina-v3-XLM-R | 84.4 | 73.7 | 56.3 | 45.6 | 90.0 | 90.9 | 70.8 | 74.1 | 66.7 | 70.6 | 73.6 | 71.0 |
| e5 (XLM-R) | 86.4 | 75.3 | 56.4 | 46.2 | 88.4 | 88.5 | 71.1 | 77.3 | 67.5 | 71.2 | 74.0 | 71.7 |
| VLM2VEC | 91.6 | 79.8 | 61.3 | 51.5 | 93.9 | 91.0 | **90.9** | **92.0** | 80.5 | 86.0 | 83.6 | 80.1 |
| VLM2VEC (finetune) | 90.2 | 79.3 | 60.0 | 50.1 | 89.8 | 91.4 | 76.9 | 85.8 | 74.1 | 78.7 | 78.2 | 77.1 |
| Qwen2.5-0.5B-CC | 86.2 | 74.9 | 56.0 | 45.1 | 92.6 | 93.2 | 73.7 | 79.0 | 69.5 | 73.0 | 75.6 | 73.0 |
| Llama-3.2-1B-CC | 88.9 | 78.8 | 59.8 | 49.1 | 96.3 | 95.6 | 80.1 | 85.1 | 77.0 | 80.8 | 80.4 | 77.9 |
| Llama-3-8B-CC | 90.4 | 80.7 | 62.7 | 51.9 | 96.5 | 96.2 | 84.2 | 89.5 | 83.3 | **86.4** | 83.4 | 80.9 |
| DeepSeek-R1-Distill-Llama-8B-CC | 91.7 | 80.9 | 62.2 | 51.9 | 96.7 | 96.3 | 84.3 | 88.3 | 82.5 | 85.2 | 83.5 | 80.5 |
| **Llama-3.1-8B-CC** | **92.2** | **81.5** | **63.5** | **52.3** | **97.1** | 96.2 | **86.5** | 89.3 | **84.7** | 85.9 | **84.8** | **81.0** |

 **Note:** “-CC” indicates encoders that underwent our Caption-Contrastive fine-tuning on top of the original LLM. 

##  Multi-lingual Cross-Modal Retrieval 

[
                  Multi-lingual retrieval results. LLM2CLIP significantly improves cross-lingual image-text retrieval
                  (Flickr-CN, COCO-CN, XM3600) with strong gains on both I2T and T2I.
]
| Models | Flickr-CN | ^^ | COCO-CN | ^^ | XM3600 | ^^ |
|:--:|:--:|----|:--:|----|:--:|----|
| ^^ | I2T | T2I | I2T | T2I | I2T | T2I |
| CN-CLIP | 80.2 | 68.0 | 63.4 | 64.0 | -- | -- |
| EVA-L-224 | 4.4 | 0.9 | 2.6 | 1.0 | 14.0 | 8.0 |
|  **\+ LLM2CLIP** | **90.6** | 75.6 | **72.0** | 70.1 | 68.3 | 56.0 |
| SigLIP2 | 79.2 | 56.9 | 55.3 | 51.7 | 59.7 | 48.2 |
|  **\+ LLM2CLIP** | 90.0 | **76.1** | 70.8 | **70.2** | **69.1** | **56.3** |

##  Vision Transfer: Stronger Visual Representations 

 Beyond retrieval, LLM2CLIP also improves the **visual encoder itself**. We observe consistent gains on **zero-shot segmentation**, **open-vocabulary detection**, and even **supervised COCO** finetuning, showing that fine-grained textual supervision can transfer to stronger visual understanding. 

[
                  Zero-shot segmentation (mIOU), open-vocabulary detection, and supervised COCO val2017 results.
]
| Method | Zero-shot Seg. mIOU | ^^ | ^^ | ^^ | OV-COCO Det. | ^^ | ^^ | COCO val2017 |
|:--:|:--:|----|----|----|:--:|----|----|:--:|
| ^^ | COCO-S | ADE | VOC | City | Novel | Base | All | AP^bb^/AP^seg^ |
| EVA02 | 12.9 | 11.5 | 21.0 | 13.5 | 24.7 | 53.6 | 46.0 | 45.0/38.2 |
| **\+ LLM2CLIP** | **15.3** | **15.8** | **29.1** | **20.1** | **28.9** | **54.7** | **48.0** | **45.6**/**38.7** |

##  LLM2CLIP boosts VLMs like Llava for stronger multimodal understanding 

 We also replace Llava 1.5’s CLIP ViT-L/14 with our LLM2CLIP-enhanced visual encoder. LLM2CLIP improves Llava across most benchmarks (and strengthens both VQA and multi-modal evaluation suites). 

[
                  Performance of Llava 1.5 benchmarks. For + LLM2CLIP, we replace Llava’s CLIP ViT-L/14 with our finetuned version.
]
| **MODEL** | **VQA** | ^^ | ^^ | ^^ | ^^ | **Pope** | ^^ | ^^ | **MM** | ^^ | ^^ | ^^ | ^^ | **Seed** | ^^ | ^^ |
|:--:|:--:|----|----|----|----|:--:|----|----|:--:|----|----|----|----|:--:|----|----|
| ^^ | **V2** | **GQA** | **Vz** | **SQA** | **TV** | **R** | **A** | **P** | **MME** | **MB** | **MC** | **LB** | **MV** | **All** | **I** | **V** |
| **Llava (Paper)** | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 87.3 | 86.1 | 84.2 | 1510.7 | 64.3 | 58.3 | 65.4 | 31.1 | 58.6 | 66.1 | 37.3 |
| **Llava (Rep.)** | 79.04 | 62.86 | 50.57 | 67.97 | 57.48 | 87.7 | **84.85** | 86.3 | 1476.69 | 66.66 | 60.39 | 58.0 | 34.3 | 59.86 | 66.95 | **39.71** |
|    **\+ LLM2CLIP** | **79.80** | **63.15** | **52.37** | **69.92** | **58.35** | **88.55** | 82.76 | **87.75** | **1505.82** | **68.29** | **60.40** | **62.7** | **34.8** | **60.96** | **68.80** | 38.96 |

## BibTeX

```
@misc{huang2024llm2clip,
  title     = {LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation},
  author    = {Weiquan Huang and
               Aoqi Wu and
               Yifan Yang and
               Xufang Luo and
               Yuqing Yang and
               Usman Naseem and
               Chunyu Wang and
               Qi Dai and
               Xiyang Dai and
               Dongdong Chen and
               Chong Luo and
               Lili Qiu and
               Liang Hu},
  year      = {2024},
  eprint    = {2411.04997},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2411.04997}
}
```
