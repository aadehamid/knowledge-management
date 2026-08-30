title: GitHub - zml/zml: Any model. Any hardware. Zero compromise. Built with @ziglang / @openxla / MLIR / @bazelbuild
description: Any model. Any hardware. Zero compromise. Built with @ziglang / @openxla / MLIR / @bazelbuild - zml/zml

# GitHub - zml/zml: Any model. Any hardware. Zero compromise. Built with @ziglang / @openxla / MLIR / @bazelbuild

ZML is a production inference stack, purpose-built to decouple AI workloads from proprietary hardware.

Any model, many hardwares, one codebase, peak performance.

Compiled directly to NVIDIA, AMD, INTEL, TPU, Trainium for peak hardware performance on any accelerator. No rewriting.

It is built using the [Zig](https://ziglang.org) language, [MLIR](https://mlir.llvm.org), and [Bazel](https://bazel.build).

We use `bazel` to build ZML and its dependencies. The only prerequisite is `bazel`, which we recommend installing through `bazelisk`.

```
brew install bazelisk
```

```
curl -L -o /usr/local/bin/bazel 'https://github.com/bazelbuild/bazelisk/releases/download/v1.28.0/bazelisk-linux-amd64'
chmod +x /usr/local/bin/bazel
```

Run the MNIST example:

```
bazel run //examples/mnist
```

This downloads a small pretrained MNIST model, compiles it, loads the weights, and classifies a random handwritten digit.

The main LLM example is [`//examples/llm`](https://github.com/zml/zml/blob/master/examples/llm). It currently supports:

- Llama 3.1 / 3.2
- Qwen 3.5
- LFM 2.5

Authenticate with Hugging Face if you want to load gated repos such as Meta Llama:

```
bazel run //tools/hf -- auth login
```

Alternatively, set the `HF_TOKEN` environment variable.

Then run a prompt directly:

```
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="What is the capital of France?"
```

Open the interactive chat loop by omitting `--prompt`:

```
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.2-1B-Instruct
```

You can also load from:

- a local directory: `--model=/var/models/meta-llama/Llama-3.2-1B-Instruct`
- S3: `--model=s3://bucket/path/to/model`

Append one or more platform flags when compiling or running:

- NVIDIA CUDA: `--@zml//platforms:cuda=true`
- AMD RoCM: `--@zml//platforms:rocm=true`
- Intel OneAPI: `--@zml//platforms:oneapi=true`
- Google TPU: `--@zml//platforms:tpu=true`
- AWS Trainium / Inferentia 2: `--@zml//platforms:neuron=true`
- Disable CPU compilation: `--@zml//platforms:cpu=false`

Example on CUDA:

```
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="Write a haiku about Zig"
```

Example on ROCm:

```
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="Write a haiku about Zig"
```

Example on Intel OneAPI:

```
bazel run //examples/llm --@zml//platforms:cpu=false --@zml//platforms:oneapi=true -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="Write a haiku about Zig"
```

```
bazel test //zml:test
```

- [`examples/llm`](https://github.com/zml/zml/blob/master/examples/llm): unified LLM CLI for Llama, Qwen, and LFM
- [`examples/mnist`](https://github.com/zml/zml/blob/master/examples/mnist): smallest end-to-end model run
- [`examples/sharding`](https://github.com/zml/zml/blob/master/examples/sharding): logical mesh, partitioners, shard-local execution, profiler output
- [`examples/io`](https://github.com/zml/zml/blob/master/examples/io): inspect and load local, `hf://`, `https://`, and `s3://` repositories through the VFS layer
- [`examples/benchmark`](https://github.com/zml/zml/blob/master/examples/benchmark): measure loading and execution performance

```
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) Layer {
            return .{
                .weight = store.createTensor("weight", .{ .d_out, .d }, null),
                .bias = store.createTensor("bias", .{.d_out}, null),
            };
        }

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        }
    };

    pub fn init(store: zml.io.TensorStore.View) Mnist {
        return .{
            .fc1 = .init(store.withPrefix("fc1")),
            .fc2 = .init(store.withPrefix("fc2")),
        };
    }

    pub fn load(
        self: *const Mnist,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.Sharding,
    ) !zml.Bufferized(Mnist) {
        return zml.io.load(Mnist, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mnist)) void {
        self.fc1.weight.deinit();
        self.fc1.bias.deinit();
        self.fc2.weight.deinit();
        self.fc2.bias.deinit();
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        var x = input.flatten().convert(.f32).withTags(.{.d});
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};
```

For a full walkthrough, see:

- [Getting Started](https://github.com/zml/zml/blob/master/docs/tutorials/getting_started.md)
- [Writing your first model](https://github.com/zml/zml/blob/master/docs/tutorials/write_first_model.md)
- [ZML Concepts](https://github.com/zml/zml/blob/master/docs/learn/concepts.md)
- [Deploying on a server](https://github.com/zml/zml/blob/master/docs/howtos/deploy_on_server.md)

- Run more examples in [`./examples`](https://github.com/zml/zml/blob/master/examples)
- Read the example-specific notes in [`examples/llm/README.md`](https://github.com/zml/zml/blob/master/examples/llm/README.md)
- Learn tagged dimensions in [`working_with_tensors.md`](https://github.com/zml/zml/blob/master/docs/tutorials/working_with_tensors.md)
- Start building a model with [`write_first_model.md`](https://github.com/zml/zml/blob/master/docs/tutorials/write_first_model.md)
- Explore deployment in [`deploy_on_server.md`](https://github.com/zml/zml/blob/master/docs/howtos/deploy_on_server.md)

See [here](https://github.com/zml/zml/blob/master/CONTRIBUTING.md).

ZML is licensed under the [Apache 2.0 license](https://github.com/zml/zml/blob/master/LICENSE).

[ ![](https://camo.githubusercontent.com/2a3d3717fb0efb49dc721fbedc96f100365d5b8198aeca6c6ce3b45397ae0a13/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d7a6d6c2f7a6d6c) ](https://github.com/zml/zml/graphs/contributors)
