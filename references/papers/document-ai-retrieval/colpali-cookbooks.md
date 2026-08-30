title: GitHub - tonywu71/colpali-cookbooks: Recipes for learning, fine-tuning, and adapting ColPali to your multimodal RAG use cases. 👨🏻‍🍳
description: Recipes for learning, fine-tuning, and adapting ColPali to your multimodal RAG use cases. 👨🏻‍🍳 - tonywu71/colpali-cookbooks

# GitHub - tonywu71/colpali-cookbooks: Recipes for learning, fine-tuning, and adapting ColPali to your multimodal RAG use cases. 👨🏻‍🍳

[![arXiv](https://camo.githubusercontent.com/b8b72935b70a9d6789827b35a0b5fe955d368c98d5f44bc87c6307eaa3af05c4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d323430372e30313434392d6233316231622e7376673f7374796c653d666f722d7468652d6261646765)](https://arxiv.org/abs/2407.01449) [![Hugging Face](https://camo.githubusercontent.com/89af6d4fde5b0e90ceb44feadad824f73a80bddbeb6544ed83c67d0349ae7214/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5669646f72652d4646443231453f7374796c653d666f722d7468652d6261646765266c6f676f3d68756767696e6766616365266c6f676f436f6c6f723d303030)](https://huggingface.co/vidore) [![X](https://camo.githubusercontent.com/72e16e6cdaf0be41cd4a7d85bfdbff9410fb5d66a2b1e0dc7684b8e5a88dab09/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5468726561642d2532333030303030303f7374796c653d666f722d7468652d6261646765266c6f676f3d58266c6f676f436f6c6f723d7768697465)](https://x.com/tonywu_71/status/1809183824464560138)

[\[ColPali Engine\]](https://github.com/illuin-tech/colpali) [\[ViDoRe Benchmark\]](https://github.com/illuin-tech/vidore-benchmark)

[ColPali](https://huggingface.co/papers/2407.01449) is a model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColPali treats each page as an image. It uses [Paligemma-3B](https://github.com/tonywu71/colpali-cookbooks/blob/main/paligemma) to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

This repository contains notebooks for learning about the ColVision family of models, fine-tuning them for your specific use case, creating similarity maps to interpret their predictions, and more! 😍

You can find the cookbooks in the [`examples`](https://github.com/tonywu71/colpali-cookbooks/tree/main/examples) directory. In the table below, they are listed from most recent to oldest.

| Task | Notebook | Description |
|----|----|----|
| Inference, interpretability | [Use the 🤗 transformers-native ColQwen2](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/use_transformers_native_colqwen2.ipynb) | Use the 🤗 transformers-native implementation of ColQwen2 for inference, scoring, and interpretability. |
| Inference, interpretability | [Use the 🤗 transformers-native ColPali](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/use_transformers_native_colpali.ipynb) | Use the 🤗 transformers-native implementation of ColPali for inference, scoring, and interpretability. |
| RAG | [ColQwen2: One model for your whole RAG pipeline with adapter hot-swapping 🔥](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/run_e2e_rag_colqwen2_with_adapter_hot_swapping.ipynb) | Save VRAM by using a unique VLM for your entire RAG pipeline. Works even on Colab's free T4 GPU! |
| Interpretability | [ColQwen2: Generate your own similarity maps 👀](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/gen_colqwen2_similarity_maps.ipynb) | Generate your own similarity maps to interpret ColQwen2's predictions. |
| Interpretability | [ColPali: Generate your own similarity maps 👀](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/gen_colpali_similarity_maps.ipynb) | Generate your own similarity maps to interpret ColPali's predictions. |
| Fine-tuning | [Fine-tune ColPali 🛠️](https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/finetune_colpali.ipynb) | Fine-tune ColPali using LoRA and optional 4bit/8bit quantization. |

The easiest way to use the notebooks is to open them from the `examples` directory and click on the Colab button below:

[![Colab](https://camo.githubusercontent.com/fad3a607a854a6142decae428f1ab034c763c8d66328c20e4ec3de42e5234bbb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4f70656e5f696e5f436f6c61622d4639414230303f6c6f676f3d676f6f676c65636f6c6162266c6f676f436f6c6f723d666666267374796c653d666f722d7468652d6261646765)](https://camo.githubusercontent.com/fad3a607a854a6142decae428f1ab034c763c8d66328c20e4ec3de42e5234bbb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4f70656e5f696e5f436f6c61622d4639414230303f6c6f676f3d676f6f676c65636f6c6162266c6f676f436f6c6f723d666666267374796c653d666f722d7468652d6261646765)

This will open the notebook in Google Colab, where you can run the code and experiment with the models.

If you prefer to run the notebooks locally, you can clone the repository and open the notebooks in Jupyter Notebook or in your IDE.

**ColPali: Efficient Document Retrieval with Vision Language Models**

Authors: **Manuel Faysse**\*, **Hugues Sibille**\*, **Tony Wu**\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\* denotes equal contribution)

```
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}
```
