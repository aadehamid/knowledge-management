title: Introducing Gemma 3n: The developer guide
description: Extremely consequential new open weights model release from Google today: Multimodal by design: Gemma 3n natively supports image, audio, video, and text inputs and text outputs. Optimized for on-device: Engineered …
author: Simon Willison

# Introducing Gemma 3n: The developer guide

26th June 2025 - Link Blog

**[Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)**. Extremely consequential new open weights model release from Google today:

> - **Multimodal by design:** Gemma 3n natively supports image, audio, video, and text inputs and text outputs.
> - **Optimized for on-device:** Engineered with a focus on efficiency, Gemma 3n models are available in two sizes based on [**effective**](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/#per-layer-embeddings-%28ple%29:-unlocking-more-memory-efficiency) parameters: E2B and E4B. While their raw parameter count is 5B and 8B respectively, architectural innovations allow them to run with a memory footprint comparable to traditional 2B and 4B models, operating with as little as 2GB (E2B) and 3GB (E4B) of memory.

This is **very** exciting: a 2B and 4B model optimized for end-user devices which accepts text, images *and* audio as inputs!

Gemma 3n is also the most comprehensive day one launch I've seen for any model: Google partnered with "AMD, Axolotl, Docker, Hugging Face, llama.cpp, LMStudio, MLX, NVIDIA, Ollama, RedHat, SGLang, Unsloth, and vLLM" so there are dozens of ways to try this out right now.

So far I've run two variants on my Mac laptop. Ollama offer [a 7.5GB version](https://ollama.com/library/gemma3n) (full tag `gemma3n:e4b-it-q4_K_M0`) of the 4B model, which I ran like this:

```
ollama pull gemma3n
llm install llm-ollama
llm -m gemma3n:latest "Generate an SVG of a pelican riding a bicycle"
```

It drew me this:

![The pelican looks a bit like a grey pig. It is floating above a bicycle that looks more like a rail cart.](https://static.simonwillison.net/static/2025/gemma3n-ollama.jpg)

The Ollama version doesn't appear to support image or audio input yet.

... but the [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) version does!

First I tried that on [this WAV file](https://static.simonwillison.net/static/2025/pelican-joke-request.wav) like so (using a recipe adapted from [Prince Canuma's video](https://www.youtube.com/watch?v=8-8R2UvUBrc)):

```
uv run --with mlx-vlm mlx_vlm.generate \
  --model gg-hf-gm/gemma-3n-E4B-it \
  --max-tokens 100 \
  --temperature 0.7 \
  --prompt "Transcribe the following speech segment in English:" \
  --audio pelican-joke-request.wav
```

That downloaded a 15.74 GB bfloat16 version of the model and output the following correct transcription:

> Tell me a joke about a pelican.

Then I had it draw me a pelican for good measure:

```
uv run --with mlx-vlm mlx_vlm.generate \
  --model gg-hf-gm/gemma-3n-E4B-it \
  --max-tokens 100 \
  --temperature 0.7 \
  --prompt "Generate an SVG of a pelican riding a bicycle"
```

I quite like this one:

![The bicycle is a sin wave, but you can make out which bits are the frame and the wheels. The pelican is white with a black triangular bill. It's a much better attempt than the Ollama one.](https://static.simonwillison.net/static/2025/gemma3n-mlx-vlm.jpg)

It's interesting to see such a striking visual difference between those 7.5GB and 15GB model quantizations.

Finally, I had it describe the image it had just created:

```
uv run --with mlx-vlm mlx_vlm.generate \
  --model gg-hf-gm/gemma-3n-E4B-it \
  --max-tokens 1000 \
  --prompt "Describe image" \
  --image gemma3n-mlx-vlm.jpg
```

Here's what I got back - it thought this was a chemical diagram!

> The image is a cartoon-style illustration depicting a molecular structure against a light blue background. The structure is composed of several differently colored and shaped elements connected by curved black lines. 
> Here's a breakdown of the elements:
> - **Large Black Circles:** There are two large, solid black circles, likely representing larger molecules or components.
> - **Cream-Colored Oval:** A prominent cream-colored oval shape is centrally located and connected to other elements. It has a smaller, darker cream-colored sphere attached to its left side.
> - **Red Oval:** A smaller, solid red oval is positioned near one of the black circles and connected to the cream-colored oval.
> - **Black Lines:** These lines act as bonds, connecting the various molecular components. They are curved and dynamic, suggesting movement or interaction.
> - **Triangular Shape:** A small black triangle is attached to the smaller cream-colored sphere.
> - **Letter "I":** The letter "I" appears twice, likely labeling specific parts of the molecule. 
> The overall impression is of a simplified representation of a biological molecule, possibly a protein or a complex organic compound. The use of different colors helps to distinguish the various components within the structure.
