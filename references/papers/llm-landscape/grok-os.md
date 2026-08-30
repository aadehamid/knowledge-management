title: Open Release of Grok-1
description: We are releasing the weights and architecture of our 314 billion parameter Mixture-of-Experts model Grok-1.

# Open Release of Grok-1

We are releasing the weights and architecture of our 314 billion parameter Mixture-of-Experts model Grok-1.

---

**We are releasing the base model weights and network architecture of [Grok-1](https://x.ai/blog/grok), our large language model. Grok-1 is a 314 billion parameter Mixture-of-Experts model trained from scratch by xAI.**

This is the raw base model checkpoint from the Grok-1 pre-training phase, which concluded in October 2023. This means that the model is not fine-tuned for any specific application, such as dialogue.

We are releasing the weights and the architecture under the Apache 2.0 license.

To get started with using the model, follow the instructions at [github.com/xai-org/grok](https://github.com/xai-org/grok).

## [Model Details](#model-details) {#model-details}

- Base model trained on a large amount of text data, not fine-tuned for any particular task.
- **314B** parameter Mixture-of-Experts model with 25% of the weights active on a given token.
- Trained from scratch by xAI using a custom training stack on top of JAX and Rust in October 2023.
