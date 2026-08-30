Docs & API

Search docs

⌘K

[Vibe](/vibe)[Studio](/studio)[Inference & Models](/inference)[Admin](/admin)[Resources](/resources)[API Reference](/api)

Search docs

⌘K

Toggle theme[Reach out](https://mistral.ai/contact?utm_source=docs&utm_medium=header_cta&utm_campaign=studio_trial)[Try Studio](https://console.mistral.ai?utm_source=docs&utm_medium=header_cta&utm_campaign=studio_trial)

[Home](/)

[Resources](/resources)

* Build
* [API Reference](/api)
* [SDKs](/resources/sdks)
* [Supported languages](/resources/languages)

* [Cookbooks](/resources/cookbooks)

* [Migration guides](/resources/migration-guides)
* Updates
* [Release notes](/resources/release-notes)
* [Changelogs](/resources/changelogs)

* [Security advisories](/resources/security-advisories)

* Knowledge base
* [Glossary](/resources/glossary)
* [Error glossary](/resources/error-glossary)
* [Known limitations](/resources/known-limitations)
* [Observability integrations](/resources/observability-integrations)

* Deprecated features

  + [Fine-tuning API (legacy)](/resources/deprecated/customization)
  + [Native reasoning (deprecated)](/resources/deprecated/native-reasoning)

  + [Fine-Tuning](/resources/deprecated/finetuning)

    - [Text & Vision Fine-tuning](/resources/deprecated/finetuning/text_vision_finetuning)
    - [Classifier Factory](/resources/deprecated/finetuning/classifier_factory)

  + Moderation & Guardrailing

* Community
* [Ambassadors](/resources/ambassadors)
* [Mistral Events ↗](https://luma.com/mistral.ai)

3. [Resources](/resources)
5. Deprecated features
7. Fine-Tuning

# Fine-tuning

Warning

Deprecated

This feature is deprecated and is no longer actively supported.

Warning

Every fine-tuning job comes with a minimum fee of $4, and there's a monthly storage fee of $2 for each model. For more detailed pricing information, please visit our [pricing page](https://mistral.ai/technology/#pricing).

Fine-tuning vs. Prompting

Copy section link

## Fine-tuning vs. Prompting

When deciding whether to use prompt engineering or fine-tuning for an AI model, it can be difficult to determine which method is best. It's generally recommended to start with prompt engineering, as it's faster and less resource-intensive. To help you choose the right approach, here are the key benefits of prompting and fine-tuning:

Benefits of Prompting

Copy section link

### Benefits of Prompting

* A generic model can work out of the box (the task can be described in a zero shot fashion)
* Does not require any fine-tuning data or training to work
* Can easily be updated for new workflows and prototyping

See the [prompt engineering guide](/inference/prompting) to explore prompting methods for Mistral models.

Benefits of Fine-tuning

Copy section link

### Benefits of Fine-tuning

* Works significantly better than prompting
* Typically works better than a larger model (faster and cheaper because it doesn't require a very long prompt)
* Provides a better alignment with the task of interest because it has been specifically trained on these tasks
* Can be used to teach new facts and information to the model (such as advanced tools or complicated workflows)

Common use cases

Copy section link

### Common use cases

Fine-tuning has a wide range of use cases, some of which include:

* Customizing the model to generate responses in a specific format and tone
* Specializing the model for a specific topic or domain to improve its performance on domain-specific tasks
* Improving the model through distillation from a stronger model by training it to mimic the behavior of the larger model
* Enhancing the model’s performance by mimicking the behavior of a model with a complex prompt, but without the need for the actual prompt, thereby saving tokens, and reducing associated costs
* Reducing cost and latency by using a small yet efficient fine-tuned model

Fine-tuning Services

Copy section link

## Fine-tuning Services

* [Text and vision general fine-tuning](/resources/deprecated/finetuning/text_vision_finetuning) via SFT: supervised fine-tuning, the most common fine-tuning method to teach the model knowledge and how to follow instructions.
* [Classifier Factory](/resources/deprecated/finetuning/classifier_factory): A tool to finetune and create classifier specific models from a dataset of text.

### WHY MISTRAL

[About us](https://mistral.ai/about)[Our customers](https://mistral.ai/customers)[Careers](https://mistral.ai/careers)[Contact us](https://mistral.ai/contact)

### EXPLORE

[AI Solutions](https://mistral.ai/solutions)[Partners](https://mistral.ai/partners)[Research](https://mistral.ai/news?category=Research)

### DOCUMENTATION

[Documentation](/)[Ambassadors](/resources/ambassadors)[Cookbooks](/resources/cookbooks)

### BUILD

[Studio](https://console.mistral.ai)[Vibe](https://mistral.ai/products/vibe)[Mistral Code](https://mistral.ai/products/mistral-code)[Mistral Compute](https://mistral.ai/products/mistral-compute)[Try the API](https://docs.mistral.ai/api)

### LEGAL

[Terms of service](https://mistral.ai/terms)[Privacy policy](https://mistral.ai/terms#privacy-policy)[Legal notice](https://mistral.ai/legal)Privacy Choices[Brand](https://mistral.ai/brand)

### COMMUNITY

[Discord↗](https://discord.gg/mistralai)[X↗](https://x.com/mistralai)[Github↗](https://github.com/mistralai)[LinkedIn↗](https://linkedin.com/company/mistralai)[Ambassadors](/resources/ambassadors)

Mistral AI © 2026

Toggle theme

![Sun](/assets/sprites/sun.gif)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)![Grass](/_next/image?url=%2Fassets%2Fsprites%2Fgrass_tile.png&w=640&q=75)

![Cat](/assets/sprites/cat-walking-white.gif)

[Native reasoning (deprecated)](/resources/deprecated/native-reasoning)[Text & Vision Fine-tuning](/resources/deprecated/finetuning/text_vision_finetuning)