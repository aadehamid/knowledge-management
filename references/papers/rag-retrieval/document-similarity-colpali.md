title: Document Similarity Search with ColPali
description: A Blog post by Frank Sommers on Hugging Face

This post was inspired by Merve Noyan's posts [here](https://huggingface.co/posts/merve/247019069617685) and [here](https://huggingface.co/posts/merve/995511131459162) on OCR-free, vision-based document AI, using the recent ColPali model. I wanted to write a more fleshed-out version, focusing on a different angle of using vision language models for document AI: similarity-based retrieval. 

Given a document page, you can use ColPali to find all document pages that are similar to that example page. This type of search gives you an entry point into large, unlabeled (or mislabeled) document archives. Similarity-based retrieval may be an initial step in 'taming' legacy corporate document stores, for example, to label a subset of business documents for subsequent fine-tuning for classification or information extraction tasks.

A Colab notebook, listed below, demonstrates this on a small synthetic business document dataset.

#  Multimodal Retrieval 

Document retrieval addresses the needle-in-the-haystack problem: You have a large document corpus of possibly millions, or more, documents, some containing many pages, and you want to find the documents, or even the pages, most relevant to your query. 

On the Web, that's what search engines do. Search engines excel at searching not only textual HTML documents, but images, videos, and audio content. 

Similarly, business documents contain not only text, but important visual and layout cues. Consider a document with a title in bold, large text on top of the first page, bold section headers, images, diagrams, charts, and tables. Successful document retrieval must take those visual modalities into account, in addition to the document text.

Document retrieval initially focused on the textual content of documents: A document's text has traditionally been extracted with OCR, and then that text is indexed for lexical search. 

OCR-based text extraction, followed by layout and bounding box analysis, is still at the heart of important document AI models, such as LayoutLM. For example, [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) encodes the document text, including the order of the text token sequence, the OCR bounding box coordinates of tokens or line segments, and the document itself. That results in state-of-the-art results in key document AI tasks, but only if the first step, the OCR text extraction, works well. 

It often doesn't.

In my recent experience, that OCR bottleneck resulted in a close to 50% performance degradation on a named-entity recognition (NER) task on a real-world, production document archive. 

This illustrates a problem: Many document AI models are evaluated on clean, academic datasets. But real world data is messy, and the same models that do well on neat, academic benchmarks may not perform as well on messy, irregular, real-world document data.

#  VLMs for Document AI 

More recently, vision-language models (VLMs) excel at capturing both visual and textual context within images, and that includes document images. Indeed, VLMs, such as [LlaVA-NeXT](https://huggingface.co/lmms-lab/llama3-llava-next-8b), [PaliGemma](https://huggingface.co/google/paligemma-3b-mix-224), [Idefics3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3), [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), exhibit great document AI capabilities out of the box, such as zero-shot information extraction, without the need for OCR. [Donut](https://github.com/clovaai/donut) and Idefics3, are specifically pretrained on large document corpora, making them ideal foundations for further fine-tuning. 

Excellent as these VLM-based, OCR-free models are for document question answering, information extraction, or classification, their main focus is not scaleable document retrieval.

A new model, [ColPali](https://huggingface.co/vidore/colpali-v1.2), represents a step change in document retrieval. ColPali fuses the multi-vector retrieval technique pioneered in [ColBERT](https://arxiv.org/abs/2004.12832) with a vision language model, [PaliGemma](https://huggingface.co/blog/paligemma), and applies this potent fusion to multimodal document retrieval. The ColPali authors' [blog post](https://huggingface.co/blog/manu/colpali) describes this model in detail.

#  Embedding Choices 

In the text-only modality, neural network-based retrieval has supplanted lexical search: Given a suitable representation of a document, such as a high-dimensional vector in the latent space of an embedding model, and a similar representation of a query, you can produce a list of documents matching the query by computing the similarity of the query vector to the document vectors. 

If you wanted to extend that technique to account for the visual modality in documents, a reasonable approach would be to use a document's representation from a VLM, such as the output tensor of the model's last hidden state, as your document embedding for retrieval. The idea is that that last hidden state vector contains a rich encoding of the document by the VLM, thereby making it a good candidate to use in vector similarity search.

It turns out, however, that having a single vector represent your document is not the most efficient option for retrieval. The authors of [ColBERT](https://arxiv.org/abs/2004.12832) (2020) found that using a *bag of vectors*, instead of a single vector, not only improves retrieval accuracy, but also reduces retrieval latency as well as computation cost. 

#  Multi-Vectors and Late Interaction 

In ColBERT, each vector represents a rich embedding of a *document token* in a latent vector space, such as BERT's. Instead of a single vector, that bag of token vectors represents the document. That allows the token embeddings to capture the rich context of the tokens within the document. The ability to capture rich context for each document token is the *Co* in ColBERT.

ColBERT document embeddings are created offline, in an indexing step. At retrieval time, a ColBERT-style retriever encodes the query into a bag of token vectors in the latent embedding space. The retriever then computes the similarity between each of the document's and the query's tokens. 

For each query token / document token pair, only the match with maximum similarity, or $MaxSimMaxSim$, is considered. The highest similarities are then summed up for the whole document, to arrive at the query / document match score. That method allows ColBERT to perform inexpensive vector comparisons and additions to identify documents matching the query. 

That also means that the document and the query interact only at retrieval time, what the ColBERT authors termed *late interaction.* The "late" is the "l" in ColBERT.

You can find a step-by-step walkthrough of $MaxSimMaxSim$ and late interaction below.

By contrast, early interaction is when the document and query are paired and processed together from the beginning of the model's architecture. While that allows the model to capture rich interactions between query and document, that semantic richness comes at increased computational cost. The ColBERT paper describes this in detail.

As the BERT part of the name suggests, ColBERT focuses on textual document retrieval. In that, it has set a new standard for retrieval accuracy and performance. The multi-vector and late interaction-based paradigm also proved eminently scaleable, as explained in the ColBERT2 paper.

ColPali's main contribution is to extend multi-vector and late interaction-based search to a vision language model, in this case to PaliGemma. The technique is generalizable and can be applied to other VLMs; there is already ColIdefics, for example.

#  Document Similarity 

You would typically use ColPali to retrieve documents matching a textual query, as in a RAG pipeline. However, I wanted to highlight ColPali's versatility by showing that the query need not just be text, but can be another richly formatted document as well. Given a document corpus $CC$ and an example document $DD$, 

*Select all documents from corpus $CC$ that are similar to the example document $DD$.*

I found similarity-based document retrieval especially useful when browsing a large, unlabeled corpus, such as an enterprise document archive: 

Similarity-based retrieval can be a first step in "taming" an enterprise archive for downstream document AI tasks, such as document classification and key information extraction. Similarity search can also give you a tool to discover mislabeled or misclassified documents. 

To see how this works with ColPali, I put together a Colab notebook:

#   ColBERT-Style $MaxSimMaxSim$  

It may be hard to grasp at first how ColBERT-style multi-vector retrieval differs from the more traditional single vector-based search. I often find that a simplified, step-by-step example is helpful, so I walk through a highly simplified ColBERT-style $MaxSimMaxSim$ by hand. Hope someone else may find this useful, too.

Consider the following 2 documents, $D1D1$ and $D2D2$, and a query $QQ$, focusing only on the document and query texts.

|  |  |
|----|----|
| Document $D1D1$ | `the apple is sweet and crisp` |
| Document $D2D2$ | `the banana is ripe and yellow` |
| Query $QQ$ | `sweet apple` |

For simplicity, assume the following 2 dimensional embeddings for $D1D1$, $D2D2$, and $QQ$. (BERT's embeddings would be 768 dimensions, and more than one token would be assigned to a word in BERT. In ColPali, these would be tokens from the VLM's embedding space.)

I assigned $[0.0,0.0][0.0, 0.0]$ to filling words in this made-up embedding space. 

The documents embeddings are created in an offline indexing phase, and the query embedding is defined at query time. The bag of embeddings represent $D1D1$, $D2D2$, and $QQ$, respectively:

**Document 1:**

| token | embedding |
|----|----|
| "the" | $[0.0,0.0][0.0, 0.0]$ |
| "apple" | $[0.9,0.1][0.9, 0.1]$ |
| "is" | $[0.0,0.0][0.0, 0.0]$ |
| "sweet" | $[0.1,0.9][0.1, 0.9]$ |
| "and" | $[0.0,0.0][0.0, 0.0]$ |
| "crisp" | $[0.7,0.7][0.7, 0.7]$ |

**Document 2:**

| token | embedding |
|----|----|
| "the" | $[0.0,0.0][0.0, 0.0]$ |
| "banana" | $[0.8,0.2][0.8, 0.2]$ |
| "is" | $[0.0,0.0][0.0, 0.0]$ |
| "ripe" | $[0.2,0.8][0.2, 0.8]$ |
| "and" | $[0.0,0.0][0.0, 0.0]$ |
| "yellow" | $[0.3,0.7][0.3, 0.7]$ |

**Query:**

| token | embedding |
|----|----|
| "sweet" | $[0.1,0.9][0.1, 0.9]$ |
| "apple" | $[0.9,0.1][0.9, 0.1]$ |

At query time, a ColBERT-style retriever computes the similarity between the first query token, **sweet**, and the **Document 1** tokens, as expressed in their dot products (other similarity measures can be used, too). It then chooses the highest similarity score, to indicate the maximum similarity between the tokens:

| tokens | dot product |
|----|----|
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{the})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{the})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{apple})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{apple})$ | $[0.1,0.9]\cdot[0.9,0.1]=0.09+0.09=0.18[0.1, 0.9] \cdot [0.9, 0.1] = 0.09 + 0.09 = 0.18$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{is})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{is})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{sweet})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{sweet})$ | $[0.1,0.9]\cdot[0.1,0.9]=(0.1*0.1)+(0.9*0.9)=0.82[0.1, 0.9] \cdot [0.1, 0.9] = (0.1 * 0.1) + (0.9 * 0.9) = \mathbf{0.82}$ (highest) |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{and})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{and})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}1_{crisp})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d1}_{crisp})$ | $[0.1,0.9]\cdot[0.7,0.7]=0.07+0.63=0.70[0.1, 0.9] \cdot [0.7, 0.7] = 0.07 + 0.63 = 0.70$ |
| $MaxSim(\mathbf{q}_{sweet},D1)=max(0.0,0.18,0.0,0.82,0.0,0.70)=0.82MaxSim(\mathbf{q}_{sweet}, D1) = max(0.0, 0.18, 0.0, \mathbf{0.82}, 0.0, 0.70) = 0.82$ |  |

The process repeats for the next query token, **apple**, and the **Document 1** tokens:

| tokens | dot product |
|----|----|
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{the})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{the})$ | $[0.9,0.1]\cdot[0.0,0.0]=0.0[0.9, 0.1] \cdot [0.0, 0.0] = 0.0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{apple})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{apple})$ | $[0.9,0.1]\cdot[0.9,0.1]=(0.9*0.9)+(0.1*0.1)=0.82[0.9, 0.1] \cdot [0.9, 0.1] = (0.9 * 0.9) + (0.1 * 0.1) = \mathbf{0.82}$ (highest) |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{is})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{is})$ | $[0.9,0.1]\cdot[0.0,0.0]=0.0[0.9, 0.1] \cdot [0.0, 0.0] = 0.0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{sweet})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{sweet})$ | $[0.9,0.1]\cdot[0.1,0.9]=(0.9*0.1)+(0.1*0.9)=0.18[0.9, 0.1] \cdot [0.1, 0.9] = (0.9 * 0.1) + (0.1 * 0.9) = 0.18$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{and})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{and})$ | $[0.9,0.1]\cdot[0.0,0.0]=0.0[0.9, 0.1] \cdot [0.0, 0.0] = 0.0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}1_{crisp})\text{sim}(\mathbf{q}_{apple}, \mathbf{d1}_{crisp})$ | $[0.9,0.1]\cdot[0.7,0.7]=(0.9*0.7)+(0.1*0.7)=0.63+0.07=0.70[0.9, 0.1] \cdot [0.7, 0.7] = (0.9 * 0.7) + (0.1 * 0.7) = 0.63 + 0.07 = 0.70$ |
| $MaxSim(\mathbf{q}_{apple},D1)=max(0.0,0.82,0.0,0.18,0.0,0.70)=0.82MaxSim(\mathbf{q}_{apple}, D1) = max(0.0, \mathbf{0.82}, 0.0, 0.18, 0.0, 0.70) = 0.82$ |  |

Finally, ColBERT aggregates the MaxSim scores to arrive at the score for $QQ$ and $D1D1$: $Score(Q,D1)=MaxSim(\mathbf{q}_{sweet},D1)+MaxSim(\mathbf{q}_{apple},D1)=0.82+0.82=1.64Score(Q, D1) = MaxSim(\mathbf{q}_{sweet}, D1) + MaxSim(\mathbf{q}_{apple}, D1) = 0.82 + 0.82 = \mathbf{1.64}$

A key benefit of this style of matching is that it allows the individual vector embeddings to benefit from the rich semantics that an embedding model, such as BERT, provides. Further, since the document's token vectors are precomputed prior to the query, at query time only $QQ$'s embeddings would need to be computed, followed by fast dot product and summation operations.

To rank $D1D1$ and $D2D2$ with regard to $QQ$, we perform the $MaxSimMaxSim$ operation for $D2D2$, and then compare $D1D1$ and $D2D2$'s scores:

| tokens | dot product |
|----|----|
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{the})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{the})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{banana})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{banana})$ | $[0.1,0.9]\cdot[0.8,0.2]=0.08+0.18=0.26[0.1, 0.9] \cdot [0.8, 0.2] = 0.08 + 0.18 = 0.26$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{is})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{is})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{ripe})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{ripe})$ | $[0.1,0.9]\cdot[0.2,0.8]=0.02+0.72=0.74[0.1, 0.9] \cdot [0.2, 0.8] = 0.02 + 0.72 = \mathbf{0.74}$ (highest) |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{and})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{and})$ | $[0.1,0.9]\cdot[0.0,0.0]=0[0.1, 0.9] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{sweet},\mathbf{d}2_{yellow})\text{sim}(\mathbf{q}_{sweet}, \mathbf{d2}_{yellow})$ | $[0.1,0.9]\cdot[0.3,0.7]=0.03+0.63=0.66[0.1, 0.9] \cdot [0.3, 0.7] = 0.03 + 0.63 = 0.66$ |
| $MaxSim(\mathbf{q}_{sweet},D2)=max0.0,0.26,0.0,0.74,0.0,0.66=0.74MaxSim(\mathbf{q}_{sweet}, D2) = max{0.0, 0.26, 0.0, \mathbf{0.74}, 0.0, 0.66} = 0.74$ |  |

| tokens | dot product |
|----|----|
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{the})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{the})$ | $[0.9,0.1]\cdot[0.0,0.0]=0[0.9, 0.1] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{banana})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{banana})$ | $[0.9,0.1]\cdot[0.8,0.2]=0.72+0.02=0.74[0.9, 0.1] \cdot [0.8, 0.2] = 0.72 + 0.02 = \mathbf{0.74}$ (highest) |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{is})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{is})$ | $[0.9,0.1]\cdot[0.0,0.0]=0[0.9, 0.1] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{ripe})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{ripe})$ | $[0.9,0.1]\cdot[0.2,0.8]=0.18+0.08=0.26[0.9, 0.1] \cdot [0.2, 0.8] = 0.18 + 0.08 = 0.26$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{and})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{and})$ | $[0.9,0.1]\cdot[0.0,0.0]=0[0.9, 0.1] \cdot [0.0, 0.0] = 0$ |
| $\text{sim}(\mathbf{q}_{apple},\mathbf{d}2_{yellow})\text{sim}(\mathbf{q}_{apple}, \mathbf{d2}_{yellow})$ | $[0.9,0.1]\cdot[0.3,0.7]=0.27+0.07=0.34[0.9, 0.1] \cdot [0.3, 0.7] = 0.27 + 0.07 = 0.34$ |
| $MaxSim(\mathbf{q}_{apple},D2)=max0.0,0.74,0.0,0.26,0.0,0.34=0.74MaxSim(\mathbf{q}_{apple}, D2) = max{0.0, \mathbf{0.74}, 0.0, 0.26, 0.0, 0.34} = 0.74$ |  |
| $Score(Q,D2)=MaxSim(\mathbf{q}_{sweet},D2)+MaxSim(\mathbf{q}_{apple},D2)=0.74+0.74=1.48Score(Q, D2) = MaxSim(\mathbf{q}_{sweet}, D2) + MaxSim(\mathbf{q}_{apple}, D2) = 0.74 + 0.74 = \mathbf{1.48}$ |  |

Comparison with regard to $QQ$:

| document | score |
|----|----|
| Document 1 | 1.64 |
| Document 2 | 1.48 |

Therefore, Document 1 is more relevant for the query $QQ$.

The main takeaway is that the bags of vectors must be stored in the document indices, and that a simple vector comparison is not enough for this type of retrieval. 
