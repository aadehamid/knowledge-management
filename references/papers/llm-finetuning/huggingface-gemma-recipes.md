title: GitHub - huggingface/huggingface-gemma-recipes: Inference, Fine Tuning and many more recipes with Gemma family of models
description: Inference, Fine Tuning and many more recipes with Gemma family of models - huggingface/huggingface-gemma-recipes

# GitHub - huggingface/huggingface-gemma-recipes: Inference, Fine Tuning and many more recipes with Gemma family of models

[![repository thumbnail](https://github.com/huggingface/huggingface-gemma-recipes/raw/main/assets/thumbnail.png)](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/assets/thumbnail.png)

🤗💎 Welcome! This repository contains *minimal* recipes to get started quickly with the Gemma family of models.

> [!NOTE]
> Gemma 4 Multimodal inference (vision, video, audio, function calling, object detection): [![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma4_%28E2B%29-Multimodal.ipynb)

To quickly run a Gemma 💎 model on your machine, install the latest version of `timm` (for the vision encoder) and 🤗 `transformers` to run inference, or if you want to fine tune it.

```
$ pip install -U -q transformers timm
```

The easiest way to start using Gemma 3n is by using the pipeline abstraction in transformers:

```
import torch
from transformers import pipeline

pipe = pipeline(
   "image-text-to-text",
   model="google/gemma-3n-E4B-it", # "google/gemma-3n-E4B-it"
   device="cuda",
   torch_dtype=torch.bfloat16
)

messages = [
   {
       "role": "user",
       "content": [
           {"type": "image", "url": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
           {"type": "text", "text": "Describe this image"}
       ]
   }
]

output = pipe(text=messages, max_new_tokens=32)
print(output[0]["generated_text"][-1]["content"])
```

Initialize the model and the processor from the Hub, and write the `model_generation` function that takes care of processing the prompts and running the inference on the model.

```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "google/gemma-3n-e4b-it" # google/gemma-3n-e2b-it
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

def model_generation(model, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = inputs["input_ids"].shape[-1]

    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=32, disable_compile=False)
        generation = generation[:, input_len:]

    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    print(decoded[0])
```

And then using calling it with our specific modality:

```
# Text Only

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]
model_generation(model, messages)
```

```
# Interleaved with Audio

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English:"},
            {"type": "audio", "audio": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/speech.wav"},
        ]
    }
]
model_generation(model, messages)
```

```
# Interleaved with Image

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
model_generation(model, messages)
```

- [Multimodal inference with Gemma 4 (vision, video, audio, function calling, object detection)](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma4_%28E2B%29-Multimodal.ipynb) [![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma4_%28E2B%29-Multimodal.ipynb)

- [Multimodal inference using Gemma 3n via pipeline](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/notebooks/gemma3n_inference_via_pipeline.ipynb) [![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/gemma3n_inference_via_pipeline.ipynb)

- [Function Calling with Gemma 3n: Local File Reader](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma_3n_Function_Calling_document_summarizer.ipynb) [![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma_3n_Function_Calling_document_summarizer.ipynb)

We include a series of notebook\+scripts for fine tuning the models.

- [Fine tuning Gemma 3n on images using TRL](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/ft_gemma3n_image_trl.py)
- [Fine tuning Gemma 3n on images (script)](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/ft_gemma3n_image_vt.py)
- [Fine tuning Gemma 3n on audio (script)](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/ft_gemma3n_audio_vt.py)
- [Fine tuning Gemma3n on video\+audio using FineVideo (all modalities)](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/gemma3n_fine_tuning_on_all_modalities.py)

- [Retrieval-Augmented Generation with Gemma 3n](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma_RAG.ipynb) [![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma_RAG.ipynb)

Before fine-tuning the model, ensure all dependencies are installed:

```
$ pip install -U -q -r requirements.txt
```

✨ **Bonus:** We've also experimented with adding **object detection** 🔍 capabilities to Gemma 3. You can explore that work in [this dedicated repo](https://github.com/ariG23498/gemma3-object-detection).
