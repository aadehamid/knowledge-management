title: Introducing Gemma 3: The Developer Guide- Google Developers Blog Introducing Gemma 3: The Developer Guide
description: Gemma 3, the newest version of Google's open model family, introduces multimodality, enhanced reasoning, and support for 140+ languages.

# Introducing Gemma 3: The Developer Guide

[Omar Sanseviero](https://developers.googleblog.com/en/search/?author=Omar+Sanseviero) Member of the Technical Staff

[Philipp Schmid](https://developers.googleblog.com/en/search/?author=Philipp+Schmid) Developer Relations Engineer

![Gemma 3](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/gemma-3_2.original.png)

Since its first launch, Gemma models have been downloaded over 100 million times, with the community creating over 60,000 variations for all kinds of use cases. We are excited to introduce Gemma 3, our most capable and advanced version of the Gemma open-model family, building upon the success of previous Gemma releases. We listened to community feedback and added the most requested features, such as longer context, multimodality, and more!

## \
What’s new in Gemma?

Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 is available in four sizes (1B, 4B, 12B, and 27B) as both pre-trained models, which can be fine-tuned for your own use cases and domains, and general-purpose instruction-tuned versions.

## How was Gemma built?

Gemma's pre-training and post-training processes were optimized using a combination of distillation, reinforcement learning, and model merging. This approach results in enhanced performance in math, coding, and instruction following. Gemma 3 uses a new tokenizer for better multilingual support for over 140\+ languages and was trained on 2T tokens for 1B, 4T for 4B, 12T for 12B, and 14T tokens for 27B, on Google TPUs using the JAX Framework.

For post-training, Gemma 3 uses 4 components:

- Distillation from a larger instruct model into the Gemma 3 pre-trained checkpoints.

- Reinforcement Learning from Human Feedback (RLHF) to align model predictions with human preferences.

- Reinforcement Learning from Machine Feedback (RLMF) to enhance mathematical reasoning.

- Reinforcement Learning from Execution Feedback (RLEF) to improve coding capabilities.

These updates significantly improved the model math, coding, and instruction following capabilities, making it the top open compact model in LMArena, with a score of 1338.

The instruct versions of Gemma 3 use the same dialog format as Gemma 2, so you don’t need to update your tooling to update to the latest version for text-only input. For image input, Gemma 3 allows specifying images interleaved with text.

### **\
Multi-turn text example**

```markdown
<bos><start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
Gemma<end_of_turn>
<start_of_turn>model
Gemma who?<end_of_turn>
```

**Interleaved image example**

```markdown
<bos><start_of_turn>user
Image A: <start_of_image>
Image B: <start_of_image>

Label A: water lily
Label B:<end_of_turn>
<start_of_turn>model
Desert rote<end_of_turn>
```

## Multimodality

Gemma 3 has an integrated vision encoder based on [SigLIP](https://arxiv.org/abs/2303.15343). The Gemma 3 vision model, which was kept frozen during training, is the same across its different sizes (4B, 12B and 27B). Thanks to this, Gemma can use images and videos as inputs, allowing it to analyze images, answer questions about an image, compare images, identify objects, and even reply about text within an image. Although the model was originally created to work with images of 896x896 pixels, a new adaptive window algorithm is used to segment input images, allowing Gemma 3 to work with high resolution and non-square images.

![Gemma 3 Multimodality example](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/gemma-3-multimodality-example.original.png)

![Gemma 3 multimodality - output example](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/gemma-3-multimodality--output-example.original.png)

## ShieldGemma 2

ShieldGemma 2 is a 4B image safety classifier built on Gemma 3. It outputs labels across key safety categories, enabling safety moderation of synthetic images (from image generation models) and natural images (which could be the input filter of a Vision-Language Model such as Gemma 3). Learn more about [ShieldGemma 2](https://developers.googleblog.com/en/safer-and-multimodal-responsible-ai-with-gemma/).

## \
What are you building?

We're continually astounded by the ingenuity of the Gemma community and the explosive growth of the [Gemmaverse](https://ai.google.dev/gemma/gemmaverse). From research labs pioneering novel fine-tuning techniques – such as the [SimPO method](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO) developed by Princeton NLP, which directly optimizes for human preferences without a reference model; INSAIT training [state-of-the-art LLMs for Bulgarian](https://ai.google.dev/gemma/gemmaverse/insait) – to developers training Gemma on entirely new modalities like [Nexa AI did with OmniAudio](https://ai.google.dev/gemma/gemmaverse/omniaudio). We can't wait to see what breakthroughs you achieve next.

## \
Get started with Gemma 3 today

Ready to explore the potential of Gemma 3 today? Here's how:

- **Experiment directly:** Use [Google AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemma-3-27b-it) to try Gemma 3 in just a couple of clicks.

- **Download the models**: Find the model weights on [Hugging Face](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) and [Kaggle](https://www.kaggle.com/models/google/gemma-3).

- **Learn & integrate:** Dive into [our technical report](https://goo.gle/Gemma3Report) and [comprehensive documentation](https://ai.google.dev/gemma/docs) to quickly integrate Gemma into your projects or start with our inference guide or try fine-tuning with a custom dataset.

- **Use your favorite development tools:** Leverage your preferred tools and frameworks, including [Hugging Face Transformers](https://huggingface.co/blog/gemma3), [Ollama](https://ollama.com/library/gemma3), our new [Gemma JAX library](https://gemma-llm.readthedocs.io/en/latest/), [MaxText](https://github.com/AI-Hypercomputer/maxtext), [LiteRT](https://developers.googleblog.com/en/gemma-3-on-mobile-and-web-with-google-ai-edge), [Gemma.cpp](https://github.com/google/gemma.cpp), llama.cpp, and [Unsloth](https://unsloth.ai/blog/gemma3).

- **Deploy your way**: Gemma 3 offers multiple deployment options, including [Google GenAI API](https://github.com/googleapis/python-genai), [Vertex AI](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3), [Cloud Run](https://cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama), [Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu), and [Cloud GPU](https://cloud.google.com/gpu) and integrations across platforms, giving you the flexibility to choose the best fit for your use case.

 posted in: 

-  [Gemma](https://developers.googleblog.com/en/search/?product_categories=Gemma) 
-  [Announcements](https://developers.googleblog.com/en/search/?content_type_categories=Announcements) 
-  [generative AI models](https://developers.googleblog.com/en/search/?tag=generative%20AI%20models) 
-  [Generative AI](https://developers.googleblog.com/en/search/?tag=Generative%20AI) 
-  [Explore](https://developers.googleblog.com/en/search/?tag=Explore) 
-  [ShieldGemma](https://developers.googleblog.com/en/search/?tag=ShieldGemma) 
-  [Gemma 3](https://developers.googleblog.com/en/search/?tag=Gemma%203) 

Related Posts
