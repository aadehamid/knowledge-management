title: vidore/colqwen2-v0.1 · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

#  ColQwen2: Visual Retriever based on Qwen2-VL-2B-Instruct with ColBERT strategy 

ColQwen is a model based on a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents from their visual features. It is a [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) extension that generates [ColBERT](https://arxiv.org/abs/2004.12832)- style multi-vector representations of text and images. It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) and first released in [this repository](https://github.com/ManuelFay/colpali)

This version is the untrained base version to guarantee deterministic projection layer initialization.

![](https://github.com/illuin-tech/colpali/blob/main/assets/colpali_architecture.webp?raw=true){width=800}

##  Version specificity 

This model takes dynamic image resolutions in input and does not resize them, changing their aspect ratio as in ColPali. Maximal resolution is set so that 768 image patches are created at most. Experiments show clear improvements with larger amounts of image patches, at the cost of memory requirements.

This version is trained with `colpali-engine==0.3.1`.

Data is the same as the ColPali data described in the paper.

##  Model Training 

###  Dataset 

Our training dataset of 127,460 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%). Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both [*ViDoRe*](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) and in the train set to prevent evaluation contamination. A validation set is created with 2% of the samples to tune hyperparameters.

*Note: Multilingual data is present in the pretraining corpus of the language model and most probably in the multimodal training.*

###  Parameters 

All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in `bfloat16` format, use low-rank adapters ([LoRA](https://arxiv.org/abs/2106.09685)) with `alpha=32` and `r=32` on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a `paged_adamw_8bit` optimizer. We train on an 8 GPU setup with data parallelism, a learning rate of 5e-5 with linear decay with 2.5% warmup steps, and a batch size of 32.

##  Usage 

###  Sentence Transformers 

ColQwen2 can be used as a multi-vector (ColBERT-style late interaction) retriever with Sentence Transformers via the `MultiVectorEncoder`:

```bash
pip install "sentence-transformers[image]>=6.0.0"
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2-v0.1")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
images = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(images)
print(f"Query 0 shape:    {tuple(query_embeddings[0].shape)}")
print(f"Document 0 shape: {tuple(document_embeddings[0].shape)}")
# Query 0 shape:    (25, 128)
# Document 0 shape: (755, 128)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[14.4976, 12.0572,  8.9111,  7.2152],
#         [ 6.9040, 15.0062,  7.6410,  7.9966]])
```

To score only the image-patch tokens (which roughly halves the stored document index), enable the mask allowlist after loading:

```python
model[-1].keep_only_token_ids = [151655]  # the <|image_pad|> token id
```

###  ColPali Engine 

> Note: `colpali-engine` 0.3.13 and later no longer send the `"Query: "` query prefix that this checkpoint was trained with (illuin-tech/colpali#280 dropped it from `ColQwen2Processor`, and illuin-tech/colpali#339 then changed the base class default it fell back on to `""`). The Sentence Transformers configuration in this repository reproduces the original training-time format, so its embeddings differ slightly from current `colpali-engine` output.

Make sure `colpali-engine` is installed from source or with a version superior to 0.3.1. `transformers` version must be > 4.45.0.

```bash
pip install git+https://github.com/illuin-tech/colpali
```

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

##  Limitations 

- **Focus**: The model primarily focuses on PDF-type documents and high-ressources languages, potentially limiting its generalization to other document types or less represented languages.
- **Support**: The model relies on multi-vector retreiving derived from the ColBERT late interaction mechanism, which may require engineering efforts to adapt to widely used vector retrieval frameworks that lack native multi-vector support.

##  License 

ColQwen2's vision language backbone model (Qwen2-VL) is under `apache2.0` license. The adapters attached to the model are under MIT license.

##  Contact 

- Manuel Faysse: [manuel.faysse@illuin.tech](mailto:manuel.faysse@illuin.tech)
- Hugues Sibille: [hugues.sibille@illuin.tech](mailto:hugues.sibille@illuin.tech)
- Tony Wu: [tony.wu@illuin.tech](mailto:tony.wu@illuin.tech)

##  Citation 

If you use any datasets or models from this organization in your research, please cite the original dataset as follows:

```bibtex
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
