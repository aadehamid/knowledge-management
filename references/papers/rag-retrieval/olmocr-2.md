title: olmOCR 2: Unit test rewards for document OCR | Ai2
description: olmOCR 2, our latest document OCR model, achieves state-of-the-art performance for English-language digitized print documents.

# olmOCR 2: Unit test rewards for document OCR | Ai2

PDFs are everywhere—research articles, government filings, textbooks, financial statements, historical archives. Turning them into trustworthy, structured text is foundational for search, analytics, accessibility, and downstream AI. Yet even strong OCR systems struggle with multi-column layouts, dense tables, math notation, and degraded scans. Teams often can't adapt models to their own document formats without brittle heuristics.

olmOCR is our end-to-end vision-language approach to reading complex documents in a single pass. With each release, we've pushed reliability on these challenging cases. Getting there required training directly on what correctness looks like—not just scaling data or model size.

Today we're releasing **olmOCR 2** (olmOCR-2-7B-1025), achieving state-of-the-art performance for real-world OCR of English-language digitized print documents. olmOCR 2 scores **82.4 points** on our olmOCR-Bench, delivering substantial improvements where OCR often fails.

Just as important, we’re shipping a practical pipeline – including training code – so you can specialize the model to your specific documents. With a modest sample of your own pages, you can fine-tune and adapt olmOCR 2—no complicated post-processing steps required.

## **How it works**

olmOCR 2 is built on Qwen2.5-VL-7B and fine-tuned on olmOCR-mix-1025, a dataset of 270,000 PDF pages with diverse properties—academic papers, historical scans, legal documents, brochures, and more. We’ve updated our dataset since our previous release to include 20,000 additional pages of difficult handwritten and typewritten documents.

olmOCR 2 reads complex pages in a single pass. A vision encoder processes the page image, and a decoder generates structured text: Markdown for headings and document structure, HTML for tables, and LaTeX for math equations. This structure is produced directly rather than stitched together by post-hoc rules, so olmOCR 2 avoids many failure modes of multi-stage pipelines and adapts better to varied document types.

For a deeper dive into olmOCR 2's design and architecture, see our [initial olmOCR blog](https://allenai.org/blog/olmocr).

## **Training with unit tests as verifiable rewards**

The core innovation in olmOCR 2 is training directly against verifiable correctness. We introduced *evaluation as unit tests* in olmOCR-Bench—deterministic verifiers that assert properties like "table structure preserved," "math faithfully transcribed," or "reading order consistent." The key insight: if a page can be checked programmatically for correctness, that same check can both supervise training and score evaluation.

For olmOCR 2, we developed a synthetic document pipeline that generates training data with these verifiable properties built in:

1. Sample real-world PDF pages with challenging content
2. Use Claude Sonnet 4 to analyze layout and re-render pages as clean, semantic HTML
3. Refine the HTML iteratively for visual fidelity to the original
4. Derive exact markdown targets and automatically generate programmatic test cases from the HTML source

This pipeline produced olmOCR-synthmix-1025: 2,186 PDF pages with 30,381 verifiable test cases at just $0.12 per page. Because we control the HTML source, transcription errors don't corrupt our training signal—we test against what we actually rendered, not what the model thought it saw.

We apply Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm, using pass/fail rewards from these unit tests. For each document, the model generates 28 completions during training. Completions that pass more unit tests receive higher rewards, teaching the model to produce faithful structured outputs rather than approximate guesses.

The rewards combine:

- Primary: Pass/fail signals from unit tests (table structure, equation accuracy, reading order, etc.)
- Auxiliary: Clean termination (proper EOS tokens) and consistent metadata extraction

Crucially, olmOCR-Bench uses the exact same unit test framework for evaluation. This means our training objective directly aligns with our benchmark. To be clear, we keep train and test splits clean and never train on olmOCR-Bench documents or test cases. The synthetic data simply helps us generate training data of a similar format to our benchmark evaluation.

## **State-of-the-art performance**

On olmOCR-Bench, olmOCR 2 achieves 82.4 points—a nearly \+4-point improvement over our previous release and one of the highest scores to date. olmOCR 2 outperforms both specialized tools like Marker (76.1) and MinerU (75.8) as well as general-purpose VLMs.

olmOCR 2 delivers substantial gains where OCR traditionally struggles:

- **82.3%** on old math scans (up from 79.9%)
- **84.9%** on tables (up from 72.9%)
- **83.7%** on multi-column (up from 77.3%)

Historical texts also show marked improvement. Earlier olmOCR versions struggled with the date in Abraham Lincoln's January 10, 1864 letter to Major General Hitchcock, consistently misreading Lincoln's handwriting. olmOCR 2 interprets it correctly.

These gains come without sacrificing deployment flexibility. We ship an FP8 quantized model that achieves 3,400 output tokens/sec on a single H100 GPU—fast enough to process 10,000 pages for less than $2.

## **Release details**

We're releasing the model weights for olmOCR 2 on Hugging Face – both FP8 and full precision – along with the datasets and code we used to train and fine-tune them. For easier experimentation and deployment, the model is also available through APIs on [DeepInfra,](https://deepinfra.com/allenai/olmOCR-2-7B-1025) [Parasail](https://www.parasail.io/) (allenai/olmOCR-2-7B-1025 in the serverless catalog), and [Cirrascale](https://ai2endpoints.cirrascale.ai/home).

olmOCR 2 represents a step toward OCR that is not only highly accurate, but fully reproducible and adaptable. Our [olmOCR toolkit](https://github.com/allenai/olmocr) provides everything needed to process documents at scale, including production inference pipelines, utilities for automatically extracting and integrating PDF metadata to improve accuracy, batch processing tools, and fine-tuning scripts for adapting olmOCR to your document types.

If your work depends on understanding documents – whether for accessibility, research, compliance, or discovery – this release will make your tech stack simpler and your results more trustworthy.

**Resources:**

- Model: [BF16](https://huggingface.co/allenai/olmOCR-2-7B-1025) & [FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [Demo](https://olmocr.allenai.org/)
- [Dataset](https://huggingface.co/datasets/allenai/olmOCR-mix-1025)
- [Paper](https://arxiv.org/abs/2510.19817)
