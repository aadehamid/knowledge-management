title: How to Run Local LLMs with Docker: Step-by-Step Guide | Unsloth Documentation
description: Learn how to run Large Language Models \(LLMs\) with Docker & Unsloth on your local device.

# How to Run Local LLMs with Docker: Step-by-Step Guide | Unsloth Documentation

You can now run any model, including Unsloth [Dynamic GGUFs](https://docs.unsloth.ai/docs/basics/dynamic-3.0-ggufs), on Mac, Windows or Linux with a single line of code or **no code** at all. We collabed with Docker to simplify model deployment, and Unsloth now powers most GGUF models on Docker.

Before you start, make sure to look over [hardware requirements](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#hardware-info--performance) and [our tips](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#hardware-info--performance) for optimizing performance when running LLMs on your device.

[Docker Terminal Tutorial](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#method-1-docker-terminal) [Docker no-code Tutorial](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#method-2-docker-desktop-no-code)

To get started, run OpenAI [gpt-oss](https://docs.unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) with a single command:

```
docker model run ai/gpt-oss:20B
```

Or to run a specific [Unsloth model](https://docs.unsloth.ai/docs/get-started/unsloth-model-catalog) / quant from Hugging Face:

```
docker model run hf.co/unsloth/gpt-oss-20b-GGUF:F16
```

You don’t need Docker Desktop, Docker CE is enough to run models.

#### **Why Unsloth \+ Docker?** {#why-unsloth--docker}

We collab with model labs like Google Gemma to fix model bugs and boost accuracy. Our Dynamic GGUFs consistently outperform other quant methods, giving you high-accuracy, efficient inference.

If you use Docker, you can run models instantly with zero setup. Docker uses [Docker Model Runner](https://github.com/docker/model-runner) (DMR), which lets you run LLMs as easily as containers with no dependency issues. DMR uses Unsloth models and `llama.cpp` under the hood for fast, efficient, up-to-date inference.

## ⚙️ Hardware Info \+ Performance {#hardware-info--performance}

For the best performance, aim for your VRAM \+ RAM combined to be at least equal to the size of the quantized model you're downloading. If you have less, the model will still run, but significantly slower.

Make sure your device also has enough disk space to store the model. If your model only barely fits in memory, you can expect around \~5 tokens/s, depending on model size.

Having extra RAM/VRAM available will improve inference speed, and additional VRAM will enable the biggest performance boost (provided the entire model fits)

**Quantization recommendations:**

- For models under 30B parameters, use at least 4-bit (Q4).
- For models 70B parameters or larger, use a minimum of 2-bit quantization (e.g., UD\_Q2\_K\_XL).

## ⚡ Step-by-Step Tutorials {#step-by-step-tutorials}

Below are **two ways** to run models with Docker: one using the [terminal](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#method-1-docker-terminal), and the other using [Docker Desktop](https://docs.unsloth.ai/docs/models/tutorials/how-to-run-llms-with-docker#method-2-docker-desktop-no-code) with no code:

### Method #1: Docker Terminal {#method-1-docker-terminal}

#### Run the model {#run-the-model}

Decide on a model to run, then run the command via terminal.

- Go to Terminal to run the commands. To verify if you have `docker` installed, you can type 'docker' and enter.
- Docker Hub defaults to running Unsloth Dynamic 4-bit, however you can select your own quantization level (see step #3).

For example, to run OpenAI `gpt-oss-20b` in a single command:

Or to run a specific [Unsloth](https://docs.unsloth.ai/docs/get-started/unsloth-model-catalog) gpt-oss quant from Hugging Face:

**This is how running gpt-oss-20b should look via CLI:**

#### To run a specific quantization level: {#to-run-a-specific-quantization-level}

If you want to run a specific quantization of a model, append `:` and the quantization name to the model (e.g., `Q4` for Docker or `UD-Q4_K_XL`). You can view all available quantizations on each model’s Docker Hub page. e.g. see the listed quantizations for gpt-oss [here](https://hub.docker.com/r/ai/gpt-oss#gptoss).

The same applies to Unsloth quants on Hugging Face: visit the [model’s HF page](https://huggingface.co/unsloth/gpt-oss-20b-GGUF?show_file_info=gpt-oss-20b-Q2_K_L.gguf), choose a quantization, then run something like: `docker model run hf.co/unsloth/gpt-oss-20b-GGUF:Q2_K_L`

### Method #2: Docker Desktop (no code) {#method-2-docker-desktop-no-code}

#### Install Docker Desktop {#install-docker-desktop}

Docker Model Runner is already available in [Docker Desktop](https://docs.docker.com/ai/model-runner/get-started/#docker-desktop).

1. Decide on a model to run, open Docker Desktop, then click on the models tab.
2. Click 'Add models \+' or Docker Hub. Search for the model.

Browse the verified model catalog available on [Docker Hub](https://hub.docker.com/r/ai).

#### Pull the model {#pull-the-model}

Click the model you want to run to see available quantizations.

- Quantizations range from 1–16 bits. For models under 30B parameters, use at least 4-bit (`Q4`).
- Choose a size that fits your hardware: ideally, your combined unified memory, RAM, or VRAM should be equal to or greater than the model size. For example, an 11GB model runs well on 12GB unified memory.

#### Run the model {#run-the-model-1}

Type any prompt in the 'Ask a question' box and use the LLM like you would use ChatGPT.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F9nVjWcVsYK9CeT8gk3nQ%252FScreenshot%25202025-11-16%2520at%25206.54.50%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3Dd7e5b63d-9c3e-42b0-882c-de046bbcfc9a&width=768&dpr=3&quality=100&sign=1f784c50d7821b005ab4c2e2ecc06688&sv=3){width=2702 height=1680}

#### **To run the latest models:** {#to-run-the-latest-models}

You can run any new model on Docker as long as it’s supported by `llama.cpp` or `vllm` and available on Docker Hub.

### What Is the Docker Model Runner? {#what-is-the-docker-model-runner}

The Docker Model Runner (DMR) is an open-source tool that lets you pull and run AI models as easily as you run containers. GitHub: [https://github.com/docker/model-runner](https://github.com/docker/model-runner)

It provides a consistent runtime for models, similar to how Docker standardized app deployment. Under the hood, it uses optimized backends (like `llama.cpp`) for smooth, hardware-efficient inference on your machine.

Whether you’re a researcher, developer, or hobbyist, you can now:

- Run open models locally in seconds.
- Avoid dependency hell, everything is handled in Docker.
- Share and reproduce model setups effortlessly.
