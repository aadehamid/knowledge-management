title: Volumes
description: Volumes are a high-performance distributed file system for Modal applications. They are optimized for write-once, read-many I/O workloads, like creating machine learning model weights and distributing them for inference.

# Volumes {#volumes}

Volumes are a high-performance distributed file system for Modal applications. They are optimized for write-once, read-many I/O workloads, like creating machine learning model weights and distributing them for inference.

Key benefits:

- Volumes are distributed by default, so you can use them alongside Modal’s global compute pool without needing to manage replicas across regions.
- Volumes have caching and chunking optimizations built-in to maximize throughput.
- Volumes come with a fully-featured filesystem interface for easy integration into your favorite ML tools and frameworks.
- Volumes are backed by multiple underlying cloud providers to guarantee high availability.

This page is a high-level guide to using Modal Volumes. For reference documentation on the `modal.Volume` object, see [this page](https://modal.com/docs/sdk/py/latest/Volume). For reference documentation on the `modal volume` CLI command, see [this page](https://modal.com/docs/cli/latest/volume).

## Pricing  {#pricing}

Please refer to our [pricing page](https://modal.com/pricing) for up-to-date prices. We calculate usage by snapshotting your total storage once a day. When you delete data, you may still be billed for that storage for up to four days, to reflect our underlying processing costs.

## Volumes v2  {#volumes-v2}

Read more about [Volumes v2](#volumes-v2-overview) below.

## Creating a Volume  {#creating-a-volume}

The easiest way to create a Volume and use it as a part of your App is to use the [`modal volume create`](https://modal.com/docs/cli/latest/volume#modal-volume-create) CLI command. This will create the Volume and output some sample code:

> 🌱 To create a v2 Volume, pass `--version=2` in the command above.

## Using a Volume on Modal  {#using-a-volume-on-modal}

To attach an existing Volume to a Modal Function, use [`Volume.from_name`](https://modal.com/docs/sdk/py/latest/Volume#from_name):

You can also browse and manipulate Volumes from an ad hoc Modal Shell:

Volumes will be mounted under `/mnt`.

Volumes are designed to provide up to 2.5 GB/s of bandwidth. Actual throughput is not guaranteed and may be lower depending on network conditions.

## Mount options  {#mount-options}

When attaching a Volume to a Function or Sandbox, you can configure mount options using [`Volume.with_mount_options`](https://modal.com/docs/sdk/py/latest/Volume#with_mount_options). These options are not stored on the Volume itself — they apply per container mount, so the same Volume can be mounted differently for distinct containers.

### Read-only mounts  {#read-only-mounts}

To prevent a container from writing to a Volume, mount it in read-only mode:

### Mounting a sub-path  {#mounting-a-sub-path}

You can mount a subdirectory of a Volume instead of the entire Volume using the `sub_path` mount option. If the subdirectory doesn’t exist yet, it will be created when the container starts.

Sub-path mounting is especially useful when you want a single Volume to serve multiple end user sessions, but don’t want a session to access or even see files for other sessions in the Volume.

**Note:** Sub-path is currently restricted to directories - you can not mount an individual file.

## Downloading a file from a Volume  {#downloading-a-file-from-a-volume}

While there’s no file size limit for individual files in a volume, the frontend only supports downloading files up to 16 MB. For larger files, please use the CLI:

### Creating Volumes lazily from code  {#creating-volumes-lazily-from-code}

You can also create Volumes lazily from code using:

> 🌱 To create a v2 Volume, pass `version=2` to the call to `from_name()` in the code above.

This will create the Volume if it doesn’t exist.

## Using a Volume from outside of Modal  {#using-a-volume-from-outside-of-modal}

Volumes can also be used outside Modal via the [Python SDK](https://modal.com/docs/sdk/py/latest/Volume) or our [CLI](https://modal.com/docs/cli/latest/volume).

### Using a Volume from local code  {#using-a-volume-from-local-code}

You can interact with Volumes from anywhere you like using the `modal` Python client library.

For more details, see the [reference documentation](https://modal.com/docs/sdk/py/latest/Volume).

### Using a Volume via the command line  {#using-a-volume-via-the-command-line}

You can also interact with Volumes using the command line interface. You can run `modal volume` to get a full list of its subcommands:

For more details, see the [reference documentation](https://modal.com/docs/cli/latest/volume).

## Volume commits and reloads  {#volume-commits-and-reloads}

Unlike a normal filesystem, you need to explicitly reload the Volume to see changes made since it was first mounted. This reload is handled by invoking the [`.reload()`](https://modal.com/docs/sdk/py/latest/Volume#reload) method on a Volume object. Similarly, any Volume changes made within a container need to be committed for those the changes to become visible outside the current container. This is handled periodically by [background commits](#background-commits) and directly by invoking the [`.commit()`](https://modal.com/docs/sdk/py/latest/Volume#commit) method on a `modal.Volume` object.

At container creation time the latest state of an attached Volume is mounted. If the Volume is then subsequently modified by a commit operation in another running container, that Volume modification won’t become available until the original container does a [`.reload()`](https://modal.com/docs/sdk/py/latest/Volume#reload).

Consider this example which demonstrates the effect of a reload:

The output for this example is this:

This code runs two containers, one for `f` and one for `g`. Only the last function invocation reads the file created and committed by `f` because it was configured to reload.

### Background commits  {#background-commits}

Modal Volumes run background commits: every few seconds while your Function or Sandbox executes, the contents of attached Volumes will be committed without your application code calling `.commit`. A final snapshot and commit is also automatically performed on container shutdown.

Being able to persist changes to Volumes without changing your application code is especially useful when [training or fine-tuning models using frameworks](#model-checkpointing).

## Model serving  {#model-serving}

A single ML model can be served by simply baking it into a `modal.Image` at build time using [`run_function`](https://modal.com/docs/sdk/py/latest/Image#run_function). But if you have dozens of models to serve, or otherwise need to decouple image builds from model storage and serving, use a `modal.Volume`.

Volumes can be used to save a large number of ML models and later serve any one of them at runtime with great performance. This snippet below shows the basic structure of the solution.

For more details, see our [guide to storing model weights on Modal](https://modal.com/docs/guide/model-weights).

## Model checkpointing  {#model-checkpointing}

Checkpoints are snapshots of an ML model and can be configured by the callback functions of ML frameworks. You can use saved checkpoints to restart a training job from the last saved checkpoint. This is particularly helpful in managing [preemption](https://modal.com/docs/guide/preemption).

For more, see our [example code for long-running training](https://modal.com/docs/examples/long-training).

### Hugging Face `transformers`  {#hugging-face-transformers}

To periodically checkpoint into a `modal.Volume`, just set the `Trainer`’s [`output_dir`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.output_dir) to a directory in the Volume.

## Volume performance  {#volume-performance}

Volumes work best when they contain less than 50,000 files and directories. The latency to attach or modify a Volume scales linearly with the number of files in the Volume, and past a few tens of thousands of files the linear component starts to dominate the fixed overhead.

There is currently a hard limit of 500,000 inodes (files, directories and symbolic links) per Volume. If you reach this limit, any further attempts to create new files or directories will error with [`ENOSPC` (No space left on device)](https://pubs.opengroup.org/onlinepubs/9799919799/).

If you need to work with a large number of files, consider using Volumes v2! It is currently in Beta. See below for more info.

## Filesystem consistency  {#filesystem-consistency}

### Concurrent modification  {#concurrent-modification}

Concurrent modification from multiple containers is supported, but concurrent modifications of the same files should be avoided. Last write wins in case of concurrent modification of the same file — any data the last writer didn’t have when committing changes will be lost!

The number of commits you can run concurrently is limited. If you run too many concurrent commits each commit will take longer due to contention. If you are committing small changes, avoid doing more than 5 concurrent commits (the number of concurrent commits you can make is proportional to the size of the changes being committed).

As a result, Volumes are typically not a good fit for use cases where you need to make concurrent modifications to the same file (nor is distributed file locking supported).

While a reload is in progress the Volume will appear empty to the container that initiated the reload. That means you cannot read from or write to a Volume in a container where a reload is ongoing (note that this only applies to the container where the reload was issued, other containers remain unaffected).

### Busy Volume errors  {#busy-volume-errors}

You can only reload a Volume when there no open files on the Volume. If you have open files on the Volume the [`.reload()`](https://modal.com/docs/sdk/py/latest/Volume#reload) operation will fail with “volume busy”. The following is a simple example of how a “volume busy” error can occur:

### Can’t find file on Volume errors  {#cant-find-file-on-volume-errors}

When accessing files in your Volume, don’t forget to pre-pend where your Volume is mounted in the container.

In the example below, where the Volume has been mounted at `/data`, “hello” is being written to `/data/xyz.txt`.

If you instead write to `/xyz.txt`, the file will be saved to the local disk of the Modal Function. When you dump the contents of the Volume, you will not see the `xyz.txt` file.

## Disk usage reporting  {#disk-usage-reporting}

Modal Volumes are not block devices and do not have a fixed capacity. Additionally, used space is not currently reported at the filesystem level. As a result, tools that query disk usage via the `statfs` syscall (e.g. `shutil.disk_usage()`, `os.statvfs()`, `df`) will return placeholder values.

If you need to check the size of a Volume, refer to the size shown in the Modal dashboard, or use `du` on the mounted Volume within a container.

## Volumes v2 overview  {#volumes-v2-overview}

Volumes v2 generally behave just like Volumes v1, and most of the existing APIs and CLI commands that you are used to will work the same between versions. Because the file system implementation is completely different, there will be some significant performance characteristics that can differ from version 1 Volumes. Below is an outline of the key differences you should be aware of.

### Volumes v2 are still in Beta  {#volumes-v2-are-still-in-beta}

You can still reap the benefits of v2 for data that isn’t precious, or that is easy to rebuild, such as log files, regularly updated training data and model weights, caches, and more.

### Volumes v2 are HIPAA compliant  {#volumes-v2-are-hipaa-compliant}

If you delete the volume, the data is guaranteed to be lost according to HIPAA requirements.

### Volumes v2 is more scaleable  {#volumes-v2-is-more-scaleable}

Volumes v2 support more files, higher throughput, and more irregular access patterns. Commits and reloads are also faster.

Additionally, Volumes v2 supports hard-linking of files, where multiple paths can point to the same inode.

### In v2, you can store as many files as you want  {#in-v2-you-can-store-as-many-files-as-you-want}

There is no limit on the number of files in Volumes v2.

By contrast, in Volumes v1, there is a limit on the number of files of 500,000, and we recommend keeping the count to 50,000 or less.

### In v2, you can write concurrently from hundreds of containers  {#in-v2-you-can-write-concurrently-from-hundreds-of-containers}

The file system should not experience any performance degradation as more containers write to distinct files simultaneously.

By contrast, in Volumes v1, we recommend no more than five writers access the Volume at once.

Note, however, that concurrent access to a particular *file* in a Volume still has last-write-wins semantics in many circumstances. These semantics are unacceptable for most applications, so any particular file should only be written to by a single container at a time.

### In v2, random accesses have improved performance  {#in-v2-random-accesses-have-improved-performance}

In v1, writes to locations inside a file would sometimes incur substantial overhead, like a rewrite of the entire file.

In v2, this overhead is removed, and only changes are written.

### In v2, you can commit using `sync`  {#in-v2-you-can-commit-using-sync}

For Volumes v2, you can trigger a commit from within a Sandbox or modal shell by running the `sync` command on the mountpoint:

This is useful when you don’t have access to the Python SDK’s [`.commit()`](https://modal.com/docs/sdk/py/latest/Volume#commit) method, such as when running shell commands in a Sandbox or during an interactive `modal shell` session.

Running `sync` on the mountpoint will flush any pending writes to the kernel and then persist all data and metadata changes to the Volume’s persistent storage.

For example, to commit changes in a modal shell session:

Or to commit from within a Sandbox:

> ⚠️ This feature is only available for Volumes v2.

### Volumes v2 has a few limits in place  {#volumes-v2-has-a-few-limits-in-place}

While we work out performance trade-offs and listen to user feedback, we have put some artificial limits in place.

- Files must be less than 1 TiB.
- At most 262,144 files can be stored in a single directory. Directory depth is unbounded, so the total file count is unbounded.
- Traversing the filesystem can be slower in v2 than in v1, due to demand loading of the filesystem tree.

### Upgrading v1 Volumes  {#upgrading-v1-volumes}

Currently, there is no automated tool for upgrading v1 Volumes to v2. v1 Volumes need to be manually migrated by creating a new v2 Volume and either copying files over from the v1 Volume or writing new files.

To reuse the name of an existing v1 Volume for a new v2 Volume, first stop all Apps that are utilizing the v1 Volume before deleting it. If this is not feasible, e.g. due to wanting to avoid downtime, use a new name for the v2 Volume.

**Warning:** When deleting an existing Volume, any deployed Apps or running Functions utilizing that Volume will cease to function, even if a new Volume is created with the same name. This is because Volumes are identified with opaque unique IDs that are resolved at application deployment or start time. A newly created Volume with the same name as a deleted Volume will have a new Volume ID and any deployed or running Apps will still be referring to the old ID until these Apps are re-deployed or restarted.

In order to create a new Volume and copy data over from the old Volume, you can use a tool like `cp` if you intend to copy all the data in one go, or `rsync` if you want to incrementally copy the data across a longer time span:

## Further examples  {#further-examples}

- [Character LoRA fine-tuning](https://modal.com/docs/examples/diffusers_lora_finetune) with model storage on a Volume
- [Protein folding](https://modal.com/docs/examples/chai1) with model weights and output files stored on Volumes
- [Dataset visualization with Datasette](https://modal.com/docs/examples/cron_datasette) using a SQLite database on a Volume
