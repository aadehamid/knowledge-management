title: GitHub - AlexsJones/llmfit: Hundreds of models & providers. One command to find what runs on your hardware.
description: Hundreds of models & providers. One command to find what runs on your hardware. - AlexsJones/llmfit

# GitHub - AlexsJones/llmfit: Hundreds of models & providers. One command to find what runs on your hardware.

 **English** · [中文](https://github.com/AlexsJones/llmfit/blob/main/README.zh.md) · [日本語](https://github.com/AlexsJones/llmfit/blob/main/README.ja.md) 

 [![Crates.io](https://camo.githubusercontent.com/ab39a4333e4d2c81eaf720bed302129c5bbd017f6002e0275fe746fdc1641e81/68747470733a2f2f696d672e736869656c64732e696f2f6372617465732f762f6c6c6d6669742e737667)](https://crates.io/crates/llmfit) [![License](https://camo.githubusercontent.com/7013272bd27ece47364536a221edb554cd69683b68a46fc0ee96881174c4214c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](https://github.com/AlexsJones/llmfit/blob/main/LICENSE) [![Signed with SignPath](https://camo.githubusercontent.com/02ef6144eb10f56f511557d80f0d4e69239be6f0dba5f48cdb1654632b935edd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5369676e506174682d7369676e65642d627269676874677265656e3f6c6f676f3d646174613a696d6167652f7376672b786d6c3b6261736536342c50484e325a79423462577875637a30696148523063446f764c336433647935334d793576636d63764d6a41774d43397a646d6369494864705a48526f505349784e694967614756705a326830505349784e6949675a6d6c7362443069643268706447556949485a705a58644362336739496a41674d4341784e6941784e69492b50484268644767675a443069545445774c6a41324e7941304c6a55324e3277744e4334334d7a51674e4334334d7a4d744d5334304c5445754e474578494445674d434177494441744d5334304d5451674d5334304d5452734d693478494449754d574578494445674d434177494441674d5334304d5451674d4777314c6a51304c5455754e4452684d534178494441674d4341774c5445754e4445304c5445754e44453065694976506a777663335a6e50673d3d)](https://about.signpath.io) 

> **📊 New: benchmark & share — real numbers from your machine, better estimates for everyone.** Download a model, serve it, and measure real tok/s on your hardware — then contribute the results back to the project as a PR, straight from the TUI. No `gh` CLI, no third-party account. Every run is saved locally first, your own measurements replace estimates in the fit table, and each merged submission ships in the next release: anyone on identical hardware gets measured `✓` numbers before they ever run a benchmark. [Follow the step-by-step benchmarking guide →](https://github.com/AlexsJones/llmfit/blob/main/docs/benchmarking.md)
> *Previously: [llmfit 1.0 — the release where the numbers became verifiable →](https://github.com/AlexsJones/llmfit/discussions/708)*

**Hundreds of models & providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU. Detects your hardware, scores each model across quality, speed, fit, and context dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default) and a classic CLI mode. Supports multi-GPU setups, MoE architectures, dynamic quantization selection, speed estimation, and local runtime providers (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio).

> **Sister projects:**
> - [sympozium](https://github.com/sympozium-ai/sympozium/) — managing agents in Kubernetes.
> - [llmserve](https://github.com/AlexsJones/llmserve) — a simple TUI for serving local LLM models. Pick a model, pick a backend, serve it.
> - [llama-panel](https://github.com/AlexsJones/llama-panel) — a native macOS app for managing local llama-server instances.

[![demo](https://github.com/AlexsJones/llmfit/raw/main/assets/demo.gif)](https://github.com/AlexsJones/llmfit/blob/main/assets/demo.gif)

|  |  |
|----|----|
| **Get started** | [Install](#install) · [Usage](#usage) · [How it works](#how-it-works) |
| **Guides** | [TUI guide](https://github.com/AlexsJones/llmfit/blob/main/docs/tui.md) · [Benchmarking step-by-step](https://github.com/AlexsJones/llmfit/blob/main/docs/benchmarking.md) · [CLI & automation](https://github.com/AlexsJones/llmfit/blob/main/docs/cli.md) · [Runtime providers](https://github.com/AlexsJones/llmfit/blob/main/docs/providers.md) · [OpenClaw integration](https://github.com/AlexsJones/llmfit/blob/main/docs/openclaw.md) |
| **Reference** | [How it works (full)](https://github.com/AlexsJones/llmfit/blob/main/docs/how-it-works.md) · [Platform & GPU support](https://github.com/AlexsJones/llmfit/blob/main/docs/platform-support.md) · [Custom models](https://github.com/AlexsJones/llmfit/blob/main/docs/custom-models.md) · [Development](https://github.com/AlexsJones/llmfit/blob/main/docs/development.md) |
| **Project** | [Contributing](#contributing) · [Alternatives](#alternatives) · [Code signing](#code-signing) · [License](#license) |

---

```
scoop install llmfit
```

If Scoop is not installed, follow the [Scoop installation guide](https://scoop.sh/).

Prebuilt binary (recommended, works on all macOS/Linux versions):

```
brew install AlexsJones/llmfit/llmfit
```

Or from the homebrew-core formula, which builds from source on macOS versions without a bottle:

```
brew install llmfit
```

```
port install llmfit
```

```
curl -fsSL https://llmfit.axjns.dev/install.sh | sh
```

Downloads the latest release binary from GitHub and installs it to `/usr/local/bin` (or `~/.local/bin` if no sudo).

**Install to `~/.local/bin` without sudo:**

```
curl -fsSL https://llmfit.axjns.dev/install.sh | sh -s -- --local
```

To install or update llmfit:

```
uv tool install -U llmfit
```

To run without installing:

```
uvx llmfit
```

You can also install llmfit as a Python package in the normal way with tools such as pip or uv.

```
docker run ghcr.io/alexsjones/llmfit
```

This prints JSON from `llmfit recommend` command. The JSON could be further queried with `jq`.

```
podman run ghcr.io/alexsjones/llmfit recommend --use-case coding | jq '.models[].name'
```

To launch the interactive TUI instead, pass the global `--tui` flag:

```
docker run --rm -it ghcr.io/alexsjones/llmfit --tui
```

```
git clone https://github.com/AlexsJones/llmfit.git
cd llmfit
cargo build --release
# binary is at target/release/llmfit
```

---

```
llmfit          # interactive TUI: your hardware, every model, ranked
```

The TUI shows your detected specs at the top and every model scored for fit, speed, quality, and context. See the [TUI guide](https://github.com/AlexsJones/llmfit/blob/main/docs/tui.md) for navigation, planning, simulation, downloads, the community leaderboard, and benchmarking.

For scripts, agents, and classic terminal output:

```
llmfit fit                    # table of all models ranked by fit
llmfit recommend --json       # top picks as JSON (agent/script consumption)
llmfit info "<model>"         # one model: fit analysis, estimate basis, verify commands
llmfit bench                  # measure real tok/s/TTFT against your running provider
llmfit doctor                 # hardware detection report for bug reports
```

Full reference: [CLI & automation](https://github.com/AlexsJones/llmfit/blob/main/docs/cli.md).

---

llmfit detects your hardware (RAM, CPU, GPU/VRAM, backend), then scores every model in its catalog across four dimensions: memory fit, estimated speed, quality, and context. Speed estimates come from a memory-bandwidth model grounded in runtime sampling and real community measurements — and every estimate ships its inputs, so `llmfit info` shows exactly what a number assumes and how to verify it on your machine.

Full detail, including the estimation formulas and the model database: [How llmfit works](https://github.com/AlexsJones/llmfit/blob/main/docs/how-it-works.md).

---

Contributions are welcome, especially new models.

Please run `cargo fmt` before pushing your changes. Most CI check failures are caused by unformatted code:

```
cargo fmt
```

Guides for adding models — locally (no rebuild) or to the built-in catalog: [Custom models](https://github.com/AlexsJones/llmfit/blob/main/docs/custom-models.md).

---

If you're looking for a different approach, check out [llm-checker](https://github.com/Pavelevich/llm-checker) -- a Node.js CLI tool with Ollama integration that can pull and benchmark models directly. It takes a more hands-on approach by actually running models on your hardware via Ollama, rather than estimating from specs. Good if you already have Ollama installed and want to test real-world performance. Note that it doesn't support MoE (Mixture-of-Experts) architectures -- all models are treated as dense, so memory estimates for models like Mixtral or DeepSeek-V3 will reflect total parameter count rather than the smaller active subset.

---

llmfit's Windows release binaries are digitally signed (Authenticode) via [SignPath.io](https://about.signpath.io/), with a free code signing certificate provided by the [SignPath Foundation](https://signpath.org/).

Signing happens automatically in the [release pipeline](https://github.com/AlexsJones/llmfit/blob/main/.github/workflows/release.yml): only artifacts built by GitHub Actions from this repository are submitted for signing, and signing requests are approved by the project maintainer ([@AlexsJones](https://github.com/AlexsJones)).

**Code signing policy:** see the [SignPath Foundation code signing policy and terms](https://signpath.org/terms).

**Privacy:** this program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it. llmfit only contacts external services when you explicitly use the corresponding feature (e.g. model downloads, runtime provider queries, or the community leaderboard).

---

MIT
