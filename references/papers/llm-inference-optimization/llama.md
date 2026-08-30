title: GitHub - ggml-org/llama.cpp: LLM inference in C/C++
description: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.

# GitHub - ggml-org/llama.cpp: LLM inference in C/C\+\+

A few options to get `llama.cpp` installed on your machine:

- Visit [https://llama.app](https://llama.app) and follow the instructions
- Run with Docker - see our [Docker documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Once installed:

```
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

- Plain C/C\+\+ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU\+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

| Backend | Target devices |
|----|----|
| [BLAS](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#blas-build) | All |
| [BLIS](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/BLIS.md) | All |
| [CANN](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cann) | Ascend NPU |
| [CUDA](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda) | Nvidia GPU |
| [HIP](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip) | AMD GPU |
| [Hexagon \[In Progress\]](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#musa) | Moore Threads GPU |
| [Metal](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO \[In Progress\]](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan) | GPU |
| [WebGPU](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#webgpu) | All |
| [ZenDNN](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#zendnn) | AMD CPU |

- [cli](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- [completion](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md)
- [server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [GBNF grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

- [How to build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Running on Docker](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- [Build on Android](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [Multi-GPU usage](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- [Performance troubleshooting](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](https://github.com/ggml-org/llama.cpp/blob/master/docs/xcframework.md)
- [Completions](https://github.com/ggml-org/llama.cpp/blob/master/docs/completions.md)
- [Models](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)
- [Release process](https://github.com/ggml-org/llama.cpp/blob/master/docs/release.md)

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) for more information

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C\+\+ - Public domain
