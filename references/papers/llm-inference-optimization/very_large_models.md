title: Serve very large language models \(DeepSeek V3, Kimi-K2, GLM 4.7/5\)
description: This example demonstrates the basic patterns for serving language models on Modal whose weights consume hundreds of gigabytes of storage.

# Serve very large language models (DeepSeek V3, Kimi-K2, GLM 4.7/5) {#serve-very-large-language-models-deepseek-v3-kimi-k2-glm-475}

This example demonstrates the basic patterns for serving language models on Modal whose weights consume hundreds of gigabytes of storage.

In short:

- load weights into a Modal Volume ahead of server launch
- use random “dummy” weights when iteratively developing your server
- use two, four, or eight H200 or B200 GPUs
- use lower-precision weight formats (FP4 on Blackwell, FP8 on Hopper)
- default to using speculative decoding, especially if batches are in the few tens of sequences

For more tips on how to serve specific types of LLM inference at high performance, see [this guide](https://modal.com/docs/guide/high-performance-llm-inference). For a gentler introduction to LLM serving, see [this example](https://modal.com/docs/examples/llm_inference).

## Set up the container image  {#set-up-the-container-image}

We start by creating a Modal Image based on the Docker image provided by the SGLang team. This contains our Python and system dependencies. Add more by chaining `.apt_install` and `.uv_pip_install` or `.pip_install` method calls, as we do below with `.entrypoint`. See the [Modal Image guide](https://modal.com/docs/guide/images) for details.

### Load model weights  {#load-model-weights}

Large model weights take a long time to move around. Model weight servers like Hugging Face will send weights at a few hundred megabytes per second. For large models, with weight sizes in the hundreds of gigabytes, that means thousands of seconds (tens of minutes) of model loading time.

After loading them we can cache these weights in a Modal [Volume](https://modal.com/docs/guide/volumes) so that they are loaded about 10x faster — about one to three gigabytes per second.

That still means minutes of startup time. Both of these latencies kill productivity when you’re iterating on aspects besides model behavior, like server configuration.

For this reason, we recommend skipping model loading while you’re developing a server or configuration — even when benchmarking, if you can! You can still exercise the same code paths if you use the `dummy` model loading format. In this sample code, we add an `APP_USE_DUMMY_WEIGHTS` environment variable to control this behavior from the command line during iteration.

We download the model weights from Hugging Face by running a Python function as part of the Modal Image build. Note that command-line logging will be somewhat limited.

To run the function, we need to pick a specific model to download. We’ll use Z.ai’s GLM 4.7 in eight bit [floating point quantization](https://modal.com/llm-almanac/quant-formats).

This model takes about thirty minutes to an hour to download from Hugging Face.

### Configure the inference engine  {#configure-the-inference-engine}

Running large models efficiently requires specialized inference engines like SGLang. These engines are generally highly configurable.

For SGLang, there are three main sources of configuration values:

- *Environment variables* for the process running `sglang`.
- *Command-line arguments* for the command to launch the `sglang` process.
- *Configuration files* loaded by the `sglang` process.

For deployments, we prefer to put information in configuration files where possible. CLI arguments and configuration files can typically be interchanged. CLI arguments are convenient when iterating, but configuration files are easier to share. We use environment variables only as a last resort, typically to activate new or experimental features.

**Environment variables**

SGLang environment variables are prefixed with `SGL_` or `SGLANG_`. The `SGL_` prefix is deprecated.

The snippet below adds any such environment variables present during deployment to the Modal Image.

**YAML**

Configuration files can be passed in YAML format.

We include a default config in-line in the code here for ease of use. It’s designed to run GLM 4.7 FP8 at low to moderate concurrency. In particular, it uses that model’s built-in multi-token prediction speculative decoding to improve [time per output token](https://modal.com/llm-almanac/how-to-benchmark).

You’ll want to provide your own configuration file for other settings, in particular if you change the model.

We add an environment variable, `APP_LOCAL_CONFIG_PATH`, to change the loaded configuration.

**Command-line arguments**

We launch our server by kicking off a subprocess. The convenience function below encapsulates the command and its arguments.

We pass a few key bits of configuration that are consumed by other code here, rather than in a configuration file, so that values stay in sync.

That includes:

- Model information, which is also used during weight cacheing
- GPU count, which is also used below when defining our Modal deployment
- the port to serve on, which is also used to connect up Modal networking

We also pass the `HF_HUB_OFFLINE` environment variable here, so that our server will crash when trying to load the real model if those weights are not in cache. For smaller models, we can instead load weights dynamically on server start (and cache them so later starts are faster). But for large models, weight loading extends the first start latency so much that downstream timeouts are triggered — or need to be extended so much that they are no longer tight enough on the happy path.

Lastly, we import the `sglang` library as part of loading the Image on Modal. This is a minor optimization, but it can shave a few seconds off cold start latencies by providing better prefetching hints, and every second counts!

## Configure infrastructure  {#configure-infrastructure}

Now, we wrap our configured SGLang server for our large model in the infrastructure required to run and interact with it. Infrastrucure in Modal is generally attached to an App. Here, we’ll attach our Modal Image as the default for Modal Functions that run in the App.

Most importantly, we need to decide what hardware to run on. [H200 and B200 GPUs](https://modal.com/blog/introducing-b200-h200) have over 100 GB of [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram) — 141 GB and 180 GB, respectively. The model’s weights will be stored in this memory, and they consume several hundred gigabytes of space, so we will generally want several of these accelerators. We also need space for the model’s KV cache of activations on input sequences.

In eight-bit precision, GLM 4.7 consumes \~350 GB of space, so we use four H200s for 564 GB of RAM.

We’ll use a Modal Server to serve our model. This reduces client latencies and provides for regionalized deployment. You can read more about it in [this example](https://modal.com/docs/examples/sglang_low_latency). To configure it, we need to pass in region information for the GPU workers and for the load-balancing proxy.

Lastly, we need to configure autoscaling parameters. By default, Modal is fully serverless, and applications scale to zero when there is no load. But booting up inference engines for large models takes minutes, which is generally longer than clients can tolerate waiting.

So a production deployment of large models that has clients with per-request SLAs in the few or tens of seconds generally needs to keep one replica up at all times. In Modal, we achieve this with the `min_containers` parameter of `App.cls` or `App.function`.

This can trigger substantial costs, so we leave the value at `0` in this sample code.

Deployments of large models with a single node per replica can generally handle a few tens of requests without queueing. When a particular replica has more requests than it can handle, we want to scale it up. This behavior is configured by setting `target_concurrency` parameter to `target_inputs`.

### Define the server  {#define-the-server}

Now we’re ready to put all of our infrastructure configuration together into a Modal Server.

The Modal Server allows us to control [container lifecycle](https://modal.com/docs/guide/lifecycle-functions). In particular, it lets us define work that a replica should do before and after it handles requests in methods decorated with `modal.enter` and `modal.exit`, respectively.

We called a `wait_for_server_ready` function in our `modal.enter` method. That’s defined below. It pings the `/health` endpoint until the server responds.

## Test the server  {#test-the-server}

You can deploy a fresh replica and test it using the command

which will create an ephemeral Modal App and execute the `local_entrypoint` code below.

Because the weights are randomized, the outputs are also random. Remove the `APP_USE_DUMMY_WEIGHTS` flag to test the trained model.

The unique client logic for Modal deployments is in the `probe` function below. Specifically, when a Modal Server is spinning up, i.e. before the `modal.enter` finishes for at least one replica, clients will see a `503 Service Unavailable` status and so should retry.

## Deploy the server  {#deploy-the-server}

When you’re ready, you can create a persistent deployment with

And hit it with any OpenAI API-compatible client!

## Addenda  {#addenda}

The `probe` function above uses this helper function to stream response tokens as they become available.
