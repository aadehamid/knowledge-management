title: Introducing Unsloth Studio | Unsloth Documentation
description: Run and train AI models locally with Unsloth Studio.

# Introducing Unsloth Studio | Unsloth Documentation

We’re launching **Unsloth Studio** (Beta): an open-source, no-code web UI for training, running and exporting open models in one unified **local** interface.

[Quickstart](https://unsloth.ai/docs/new/studio#quickstart) [Features](https://unsloth.ai/docs/new/studio#features) [Github](https://github.com/unslothai/unsloth)

- **Run GGUF**, **MLX** and diffusion image/video models locally on **Mac**, Windows, Linux.
- Train 500\+ models 2x faster with 70% less VRAM (no accuracy loss)
- Run and train text, diffusion vision, TTS audio, embedding models

![Cover](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FJDIIEQ8fFZy2g5snC2kp%252Fcloudflare%2520url.png%3Falt%3Dmedia%26token%3Daadb42b5-16f0-42d1-8059-c2611e7cd931&width=490&dpr=3&quality=100&sign=94f4694749bcbbe3060251a4ae6ae449&sv=3){width=1920 height=1080}

### **Deploy anywhere** {#deploy-anywhere}

Securely deploy models remotely and access Unsloth anywhere via Cloudflare HTTPS

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FHTlw8oO7wDIYyGqilty0%252Funsloth%2520qwen3.8%2520final.png%3Falt%3Dmedia%26token%3Db8b32558-d0c8-4de8-bedf-7ad10e2c4c4a&width=768&dpr=3&quality=100&sign=80b3076adde63d5456e99e039132df16&sv=3){width=2560 height=1600}

- **MacOS:** Training, MLX and GGUF inference all work inside of Unsloth.

## ⭐ Features {#features}

### Execute code \+ heal Tool calling {#execute-code--heal-tool-calling}

Unsloth Studio lets LLMs run Bash and Python, not just JavaScript with bypass permissions. It also sandboxes programs like Claude Artifacts so models can test code, generate files, and verify answers with real computation.

E.g. Unsloth creates a sandbox to allow GLM-5.2 to execute code which shows as HTML preview in Canvas.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FLL8RSnAuc7XdutjAVD4n%252Fcode%2520exec.png%3Falt%3Dmedia%26token%3D566a7385-69e0-46db-a022-bef8e2cad677&width=768&dpr=3&quality=100&sign=d6e2f3e06c67d8e47bdcec00b2e4ada6&sv=3){width=1920 height=1080} Accurate tool calls with sandboxed code execution

### Web search upgraded {#web-search-upgraded}

Unsloth's private, unlimited and secure web search actually visits pages directly to collect relevant information and data and doesn't just scan through website summaries. This provides outputs much more accurate / in-depth info and context.

E.g. Qwen3.5-4B searched 20\+ websites and cited sources, with web search happening inside its thinking trace.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FY3nfB43TEk7k11zcE4wm%252Fthe%2520big%2520one.gif%3Falt%3Dmedia%26token%3D335be087-7375-4f89-9039-71195ee44ab8&width=768&dpr=3&quality=100&sign=00c1a05af4c2ffa537298ad6a6ae4869&sv=3){width=956 height=720}

### Unsloth as an API endpoint {#unsloth-as-an-api-endpoint}

You can now use local LLMs via tools like [Claude Code](https://unsloth.ai/docs/basics/claude-code) and [Codex](https://unsloth.ai/docs/basics/codex) by connecting it to [Unsloth's API endpoint](https://unsloth.ai/docs/basics/api). This means you'll be able to directly run Qwen and Gemma models in those tools with Unsloth's inference which includes features like self-healing tool-calling, websearch etc.

You can also [connect a provider](https://unsloth.ai/docs/integrations/connections) like OpenAI, Anthropic or vLLM to Unsloth.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F1s98Id9xclzwMfxjXw2O%252Funsloth%2520api%2520cropped.png%3Falt%3Dmedia%26token%3D64fac263-ca5b-4447-a740-41f58ec94904&width=768&dpr=3&quality=100&sign=515c07ea0b0a02007f8955affa4d41db&sv=3){width=1328 height=904}

### **No-code training** {#no-code-training}

[Upload PDF, CSV, JSON](https://unsloth.ai/docs/new/studio#data-recipes) docs, or YAML configs and start training instantly on NVIDIA. Unsloth’s kernels optimize LoRA, FP8, FFT, PT across 500\+ text, vision, TTS/audio and embedding models.

Fine-tune the latest LLMs like [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune) and NVIDIA [Nemotron 3](https://unsloth.ai/docs/models/nemotron-3). [Multi-GPU](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) works automatically, with a new version coming.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRjAfHShyL7MfHfq6BStl%252Fonboarding%2520updated.png%3Falt%3Dmedia%26token%3D7cdde1a0-8f8c-4d25-9414-e28f35f211cd&width=768&dpr=3&quality=100&sign=8ed3b30141978f29c36d9bab2011dcc4&sv=3){width=1515 height=1120}

### Data Recipes {#data-recipes}

[**Data Recipes**](https://unsloth.ai/docs/new/studio/data-recipe) transforms your docs into useable / synthetic datasets via graph-node workflow. Upload unstructured or structured files like PDFs, CSV and JSON. Unsloth Data Recipes, powered by NVIDIA Nemo [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner), auto turns documents into your desired formats.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fcc9T0V8WsyjcuOE2sIVV%252Fdata%2520recipes%2520longer.png%3Falt%3Dmedia%26token%3D5ae33e8d-09b1-45e0-8f5c-40dca8bbcf0c&width=768&dpr=3&quality=100&sign=9d7943b9c3e8ba406d41f8895ec241d3&sv=3){width=2416 height=1618}

### Observability {#observability}

Gain [complete visibility](https://unsloth.ai/docs/new/studio/start#training-progress) into and control over your training runs. Track training loss, gradient norms, and GPU utilization in real time, and customize to your liking.

You can even view the training progress on other devices like your phone.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FCIrWHN1JzfaFNOoavmZS%252Fobserve%2520new.png%3Falt%3Dmedia%26token%3D21fdbc5b-a073-437a-b487-b5bdff4716f6&width=768&dpr=3&quality=100&sign=078cd212970e4265094eb1e627d2d7ce&sv=3){width=1760 height=1256}

### Export / Save models {#export-save-models}

[**Export any model**](https://unsloth.ai/docs/new/studio/export), including your fine-tuned models, to safetensors, or GGUF for use with llama.cpp, vLLM, Ollama, LM Studio, and more.

Stores your training history, so you can revisit runs, export again and experiment.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F8UHzGTHF9q6LWrJy8Y4r%252FScreenshot%25202026-03-15%2520at%25203.02.02%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3Dcb5e78f8-481a-4c9f-9361-db53e6e0ec37&width=768&dpr=3&quality=100&sign=ce593de0aa3155259907257c2f1e0f49&sv=3){width=988 height=626}

### Privacy first \+ Secure {#privacy-first--secure}

Unsloth Studio can be used 100% offline and locally on your computer. Its token-based authentication, including encrypted password and JWT access / refresh flows keeps your data secure.

You can use pre-exisiting / old models or GGUFs that previously downloaded from HF etc. Read [instructions here](https://unsloth.ai/docs/new/studio/chat#using-old-existing-gguf-models).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F15gRLbMDX1ReKdHBBl1G%252FScreenshot%25202026-03-15%2520at%25203.54.51%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3Dca096807-54c2-4d8c-bdc1-c1bb0055469b&width=768&dpr=3&quality=100&sign=33168929fecae0e758cb993b9562b4c0&sv=3){width=1772 height=1062}

## ⚡ Quickstart {#quickstart}

### **Install with Unsloth Desktop** {#install-with-unsloth-desktop}

The easiest way to install Unsloth Studio is with the native Desktop app. Download it for your operating system:

For a manual Unsloth Studio installation, use the commands below. Run the same command again to update:

### **MacOS, Linux, WSL:** {#macos-linux-wsl}

### **Windows PowerShell:** {#windows-powershell}

#### Launch Unsloth {#launch-unsloth}

**Launch Unsloth securely with HTTPS and Cloudflare**

**NEW!** Unsloth now provides a secure way to launch Unsloth over HTTPS through a free Cloudflare tunnel. Use the below (works in Windows, Mac & Linux):

**For more details about install and uninstallation please visit the**  [**Unsloth Studio Install**](https://unsloth.ai/docs/new/studio/install)  **section.**

[Installation](https://unsloth.ai/docs/new/studio/install)

####  Google Colab notebook {#google-colab-notebook}

We’ve created a [free Google Colab notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb) so you can explore all of Unsloth’s features on Colab’s T4 GPUs. You can train and run most models up to 22B parameters, and switch to a larger GPU for bigger models. Just Click 'Run all' and the UI should pop up after installation.

Once installation is complete, scroll to **Start Unsloth Studio** and click **Open Unsloth Studio** in the white box shown on the left:

**Scroll further down, to see the actual UI.**

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FkYitMrK55Ic6eIGqiKEJ%252FScreenshot%25202026-03-16%2520at%252011.21.16%25E2%2580%25AFPM.png%3Falt%3Dmedia%26token%3D4388c309-a598-41f3-9301-e434c334ac1c&width=768&dpr=3&quality=100&sign=a12e3a8ee456757fc38f62118fa8e6b3&sv=3){width=884 height=404}

Sometimes the Unsloth link may return an error. This happens because you might have disabled cookies or you're using an adblocker or Mozilla. You can still access the UI by scrolling below the button.

### 👾 Unsloth Start {#unsloth-start}

[Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start) lets you connect [Claude Code](https://unsloth.ai/docs/basics/claude-code), [Codex](https://unsloth.ai/docs/basics/codex) and other agents to local models via the `unsloth start` command.

Start Unsloth, load a model, open your project folder, and then run:

Replace `claude` with any agent below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FpXE6kCHjh8qOEaggf94M%252FScreenshot_20260718_122426.png%3Falt%3Dmedia%26token%3Da59e4c8c-efdb-451b-b1f8-621955564f6d&width=768&dpr=3&quality=100&sign=efb41f556577e0908b3e23d3c0d1d037&sv=3){width=1058 height=739} Claude Code running with Qwen3.5 locally.

##  Workflow {#workflow}

Here is a usual workflow of Unsloth Studio to get you started:

1. Load a model from local files or a supported integration.
2. Import training data from PDFs, CSVs, or JSONL files, or build a dataset from scratch.
3. Start training with recommended presets or customize the config yourself.
4. Chat with the trained model and compare its outputs against the base model.

You can read our individual deep dives into each section of Unsloth Studio:

##  FAQ {#faq}

**Does Unsloth collect or store data?** Unsloth does not collect usage telemetry. Unsloth only collects the minimal hardware information required for compatibility, such as GPU type and device (e.g. Mac). Unsloth Studio runs 100% offline and locally.

**How do I use an old / exisiting model that I downloaded previously from Hugging Face?** Yes, you can use pre-exisiting/old models or GGUFs that you previously downloaded from Hugging Face etc. They should be now be automatically detected by Unsloth otherwise read our [instructions here](https://unsloth.ai/docs/new/studio/chat#using-old-existing-gguf-models).

**Why is inference sometimes slower in Unsloth?** Unsloth, like other local inference apps, are powered by llama.cpp, so speeds should be mostly the same. Sometimes Unsloth might be because you turned on web-search, code execution, self-healing tool-calling on. All these features may make your inference slower. If the speed difference is still slower with all features turned off, please make a GitHub issue!

**Does Unsloth Studio support OpenAI-compatible APIs?** Yes, see our [API endpoint guide here](https://unsloth.ai/docs/basics/api).

**Is Unsloth now licensed under AGPL-3.0?** Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains licensed under [**Apache 2.0**](https://github.com/unslothai/unsloth?tab=Apache-2.0-1-ov-file), while certain optional components, such as the Unsloth Studio UI are licensed [**AGPL-3.0**](https://github.com/unslothai/unsloth?tab=AGPL-3.0-2-ov-file).

This structure helps support ongoing Unsloth development while keeping the project open source and enabling the broader ecosystem to continue growing.

**Does Unsloth only support LLMs?** No. Unsloth supports a range of supported `transformers` compatible model families, including text, multimodal models, [text-to-speech](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), audio, [embeddings](https://unsloth.ai/docs/basics/embedding-finetuning), and BERT-style models.

**Can I use my own training config?** Yes. Import a YAML config and Unsloth will pre-fill the relevant settings.

**Do you need to train models to use the UI?** No, you can just download any GGUF or model without fine-tuning any model.

#### Acknowledgements {#acknowledgements}

A huge thank you to NVIDIA and Hugging Face for being part of our launch. Also thanks to all of our early beta testers for Unsloth Studio, we truly appreciate your time and feedback. We’d also like to thank llama.cpp, PyTorch and open model labs for providing the infrastructure that made Unsloth Studio possible.
