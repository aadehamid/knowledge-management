title: GitHub - mozilla-ai/llamafile: Distribute and run LLMs with a single file.
description: Distribute and run LLMs with a single file. Contribute to mozilla-ai/llamafile development by creating an account on GitHub.

# GitHub - mozilla-ai/llamafile: Distribute and run LLMs with a single file.

[![[line drawing of llama animal head in front of slightly open manilla folder filled with files]](https://github.com/mozilla-ai/llamafile/raw/main/docs/images/llamafile-640x640.png){width=320 height=320}](https://github.com/mozilla-ai/llamafile/blob/main/docs/images/llamafile-640x640.png)

[![License](https://camo.githubusercontent.com/b29de0acdfd19013f1f02689b15c933e4a6c145be9efa718288f88ba3280b1c5/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d417061636865253230322e302d626c75652e737667)](https://github.com/mozilla-ai/llamafile/blob/main/LICENSE) [![Based on llama.cpp](https://camo.githubusercontent.com/fa7ddec88d9f077a0533f628171ee0c465cdaa9c57b9d19043612789400df519/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6c616d612e6370702d376635656535342d6f72616e67652e737667)](https://github.com/ggml-org/llama.cpp/commit/7f5ee54) [![Based on whisper.cpp](https://camo.githubusercontent.com/a875d676511bc5b937ae74ce8fbdcaa6fc0ab5a9a49187159d8106e6af516913/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f776869737065722e6370702d326565656261352d677265656e2e737667)](https://github.com/ggml-org/whisper.cpp/commit/2eeeba5) [![Discord](https://camo.githubusercontent.com/b9e066c996f447dd8f42bbea676850fdf5bfbd958e299370c07e9a3a75455a5b/68747470733a2f2f646362616467652e6c696d65732e70696e6b2f6170692f7365727665722f59754d4e65754b5374723f7374796c653d666c6174)](https://discord.gg/YuMNeuKStr) [![Mozilla Builders](https://camo.githubusercontent.com/3bbdc36a1c1b8318dc952bd064bf3936794399d66a406e0ca54c2fc8cf927097/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4275696c646572732d3645364536453f6c6f676f3d6d6f7a696c6c61266c6f676f436f6c6f723d7768697465266c6162656c436f6c6f723d344134413441)](https://builders.mozilla.org/)

**llamafile lets you distribute and run LLMs with a single file.**

llamafile is a [Mozilla Builders](https://builders.mozilla.org/) project (see its [announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/)), now revamped by [Mozilla.ai](https://www.mozilla.ai/open-tools/llamafile).

Our goal is to make open LLMs much more accessible to both developers and end users. We're doing that by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most operating systems and CPU architectures, with no installation.

llamafile also includes **[whisperfile](https://docs.mozilla.ai/llamafile/whisperfile)**, a single-file speech-to-text tool built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and the same Cosmopolitan packaging. It supports transcription and translation of audio files across all the same platforms, with no installation required.

**llamafile versions starting from 0.10.0 use a new build system**, aimed at keeping our code more easily aligned with the latest versions of llama.cpp. This means they support more recent models and functionalities, but at the same time they might be missing some of the features you were accustomed to (check out [this doc](https://github.com/mozilla-ai/llamafile/blob/main/README_0.10.0.md) for a high-level description of what has been done). If you liked the "classic experience" more, you will always be able to access the previous versions from our [releases](https://github.com/mozilla-ai/llamafile/releases) page. Our pre-built llamafiles always show which version of the server they have been bundled with ([0.9.\* example](https://huggingface.co/mozilla-ai/llava-v1.5-7b-llamafile), [0.10.\* example](https://huggingface.co/mozilla-ai/llamafile_0.10)), so you will always know which version of the software you are downloading.

> **We want to hear from you!** Whether you are a new user or a long-time fan, please share what you find most valuable about llamafile and what would make it more useful for you. [Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).

Download and run your first llamafile in minutes:

```
# Download an example model (Qwen3.5 0.8B)
curl -LO https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile

# Make it executable (macOS/Linux/BSD)
chmod +x Qwen3.5-0.8B-Q8_0.llamafile

# Run it
./Qwen3.5-0.8B-Q8_0.llamafile
```

We chose this model because that's the smallest one we have built a llamafile for, so most likely to work out-of-the-box for you. If you have powerful hardware and/or GPUs, [feel free to choose](https://docs.mozilla.ai/llamafile/getting-started/pre-built-llamafiles) larger and more expressive models which should provide more accurate responses.

**Windows users:** Rename the file to add `.exe` extension before running.

**Note - Only executables under 4GB can run on Windows, so any llamafile above 4GB won't work. Download the [llamafile](https://github.com/mozilla-ai/llamafile/releases) binary and run it with any [external weights/models(GGUF)](https://docs.mozilla.ai/llamafile/getting-started/quickstart#using-llamafile-with-external-weights).**

Check the full documentation at [docs.mozilla.ai/llamafile](https://docs.mozilla.ai/llamafile), or directly jump into one of the following subsections:

- [Quickstart](https://docs.mozilla.ai/llamafile/getting-started/quickstart)
- [Pre-built llamafiles](https://docs.mozilla.ai/llamafile/getting-started/pre-built-llamafiles)
- [Running a llamafile](https://docs.mozilla.ai/llamafile/using-llamafile/running_llamafile)
- [Creating llamafiles](https://docs.mozilla.ai/llamafile/using-llamafile/creating_llamafiles)
- [Source installation](https://docs.mozilla.ai/llamafile/using-llamafile/source_installation)
- [Technical details](https://docs.mozilla.ai/llamafile/reference/technical_details)
- [Supported Systems](https://docs.mozilla.ai/llamafile/reference/support)
- [Troubleshooting](https://docs.mozilla.ai/llamafile/reference/troubleshooting)
- [Whisperfile](https://docs.mozilla.ai/llamafile/whisperfile)

While the llamafile project is Apache 2.0-licensed, our changes to llama.cpp and whisper.cpp are licensed under MIT (just like the projects themselves) so as to remain compatible and upstreamable in the future, should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.

[![Star History Chart](https://camo.githubusercontent.com/66214396b53c9e2a8b544516668d2e7b7ef934b2fc54229f728f1e51641907b5/68747470733a2f2f6170692e737461722d686973746f72792e636f6d2f7376673f7265706f733d4d6f7a696c6c612d4f63686f2f6c6c616d6166696c6526747970653d44617465)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
