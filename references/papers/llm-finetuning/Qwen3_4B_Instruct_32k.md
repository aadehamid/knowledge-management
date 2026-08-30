# Fine-Tuning Qwen3-4B with MLX-LM-LoRA on Apple Silicon

Welcome to this step-by-step tutorial on fine-tuning large language models using **MLX-LM-LoRA** on Apple Silicon! 

This notebook demonstrates how to:
- ✅ Fine-tune a 4B parameter model with **32K+ context length** under 16GB RAM
- ✅ Use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- ✅ Apply **4-bit quantization** to reduce memory usage
- ✅ Train on custom instruction datasets
- ✅ Push your fine-tuned adapter to Hugging Face Hub

**Requirements:**
- Apple Silicon Mac (M1/M2/M3/M4/M5 series)
- At least 16GB of unified memory
- Python 3.8+

Let's get started! 🚀

```python
%%capture
!pip install -U mlx-lm-lora
```

## Step 1: Import Required Libraries

Now that we have installed `mlx-lm-lora`, let's import all the necessary libraries:

- **`from_pretrained`**: Loads the base model and applies quantization
- **`SFTTrainingArgs` & `train_sft`**: Handles supervised fine-tuning configuration and execution
- **`TextDataset` & `CacheDataset`**: Efficient dataset handling for training
- **`TrainingCallback` & `WandBCallback`**: Logging and monitoring during training
- **`generate`**: Used to test the model before and after fine-tuning

```python
from mlx_lm_lora.utils import from_pretrained, save_pretrained_merged, calculate_iters, push_to_hub
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft, evaluate_sft
from mlx_lm_lora.trainer.datasets import CacheDataset, TextDataset

from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.tuner.callbacks import TrainingCallback, WandBCallback
from mlx_lm.generate import generate

import mlx.optimizers as optim

from datasets import load_dataset
```

## Step 2: Configure Training Parameters

Here we set up all the configuration parameters for our fine-tuning job:

### Model Configuration
- **`model_name`**: The base model from Hugging Face (Qwen3-4B-Instruct)
- **`new_model_name`**: Name for your fine-tuned model
- **`adapter_path`**: Local path where LoRA adapter weights will be saved

### LoRA Configuration
- **`rank`**: Controls the bottleneck size (higher = more capacity, but slower). Try 8, 16, 32, 64, or 128
- **`dropout`**: Regularization to prevent overfitting
- **`scale`**: Controls how strongly LoRA updates affect base weights
- **`num_layers`**: Number of layers to apply LoRA (-1 for all layers)

### Quantization Configuration
- **`bits`**: Quantization precision (4-bit saves the most memory)
- **`group_size`**: Smaller = better quality, larger = faster inference
- **`mode`**: "mxfp4" provides good balance of quality and speed

**Pro Tip:** With these settings, you can handle 32K+ token sequences on just 16GB RAM! 🎯

```python
model_name = "Qwen/Qwen3-4B-Instruct-2507" # The base model to fine-tune.
new_model_name = "Qwen3-4B-Instruct-2507-LoRA-Dolci-Instruct" # The name for the fine-tuned model with LoRA applied.
adapter_path = f"./{new_model_name}" # The path to save the LoRA adapter. This is a small file that contains the fine-tuned weights and can be merged with the base model for inference.
user_name = "mlx-community" # Hugging Face username, needed if you want to push the model to the Hugging Face Hub. You can create an account for free at https://huggingface.co/join

dataset_name = "mlx-community/Dolci-Instruct-SFT-No-Tools-100K" # The dataset to fine-tune on.

max_seq_length = 32768 # Yes this is possible under 16GB RAM :D

lora_config = { # LoRA adapter configuration
    "rank": 12,  # Low-rank bottleneck size (Larger rank = smarter, but slower). Suggested 8, 16, 32, 64, 128
    "dropout": 0.0,
    "scale": 10.0, # Multiplier for how hard the LoRA update hits the base weights
    "use_dora": False, # Use DoRA, which is a more efficient version of LoRA that uses a single matrix instead of two.
    "num_layers": 10 # Use -1 for all layers
}
quantized_config={
    "bits": 4, # Use 4 bit quantization. Suggested 4, 6, 8
    "group_size": 32, # Quantize in groups of 32 weights. Smaller group size means better performance but slower inference. Suggested 32, 64, 128
    "mode": "mxfp4", # Quantization mode. "mxfp4" is a good balance between performance and accuracy.
}
```

## Step 3: Load and Quantize the Model

This step loads the base model and applies quantization + LoRA configuration:

1. **Downloads** the model from Hugging Face (cached locally after first download)
2. **Quantizes** the model to 4-bit precision to reduce memory footprint
3. **Injects** LoRA adapters into specified layers
4. **Creates** the adapter file structure for saving fine-tuned weights

The `print_trainable_parameters()` function shows how many parameters will actually be trained. With LoRA, you'll see only a small fraction (typically <1%) of total parameters are trainable, which is why fine-tuning is so efficient! 💪

```python
# Load the model, tokenizer, and create/save the adapter path and file.
model, tokenizer, adapter_file = from_pretrained(
    model=model_name,
    new_adapter_path=adapter_path,
    lora_config=lora_config,
    quantized_load=quantized_config,
)
print_trainable_parameters(model)
```

```python
# The formatting function applies the chat template to the messages in the dataset, which is necessary for the model to understand the input correctly.
def format(sample):
    sample["text"] = tokenizer.apply_chat_template(
        conversation=sample["messages"],
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False
    )
    return sample
```

```python
# Load the dataset, apply the formatting function, and create the TextDataset objects for training and validation. We take a subset of the dataset for quick training, but you can remove the .take() calls to use the full dataset.
train_dataset = load_dataset(dataset_name, split="train").take(100).map(format)
valid_dataset = load_dataset(dataset_name, split="valid").take(10).map(format)

# The TextDataset class is a simple wrapper that tokenizes the text and prepares it for training.
train_set = TextDataset(train_dataset, tokenizer, text_key="text")
valid_set = TextDataset(valid_dataset, tokenizer, text_key="text")
```

## Step 6: Start Fine-Tuning! 🚀

Now for the main event - fine-tuning the model with LoRA!

### Optimizer
We use **Muon optimizer** with a low learning rate (2e-4) to make small, careful adjustments to the model.

### Training Arguments Explained:
- **`batch_size=1`**: Process one sample at a time to save RAM
- **`gradient_accumulation_steps=8`**: Accumulate gradients over 8 steps to simulate batch_size=8
- **`iters`**: Total training steps (calculated based on dataset size and epochs)
- **`max_seq_length=32768`**: Support for extremely long contexts!
- **`grad_checkpoint=True`**: Trade computation for memory (enables long context training)
- **`seq_step_size=512`**: Process sequences in chunks for efficiency

### Monitoring
We use **Weights & Biases (WandB)** to track:
- Training loss
- Validation loss
- Learning rate
- Training speed (tokens/sec)

The model will:
1. Train for the specified iterations
2. Save checkpoints every 10 steps
3. Evaluate on validation set every 20 steps
4. Log metrics every 10 steps

**Training time:** Approximately 3-5 minutes on my MacBook Pro M4 Pro 24GB.

Press Run and grab a coffee! ☕

```python
# Just to check everything is working, let's print the first formatted text from the training dataset.
print(train_dataset[0]["text"])
```

## Step 4: Prepare the Dataset

We load the dataset and apply a formatting function that:

1. **Applies the chat template**: Converts conversation format into the model's expected format
2. **Tokenizes**: Prepares text for the model
3. **Creates train/validation splits**: We use a subset (100 training samples) for quick demonstration

The `TextDataset` class handles:
- Efficient tokenization with caching
- Proper sequence packing for long contexts
- Memory-efficient data loading

**Note:** We're using `.take(100)` for quick training. Remove it to use the full 100K dataset for better results!

---

## Step 5: Test the Base Model (Before Fine-Tuning)

Before we start training, let's test the base model's performance on a sample prompt. This gives us a baseline to compare against after fine-tuning.

We'll ask it to solve a multi-step math problem involving:
- Geometric sequence summation
- Probability calculation

Let's see how the base model performs! 🧪

```python
prompt = "Find the sum of the first 6 terms of a geometric sequence with first term a=4 and common ratio r=3, then calculate the probability of getting exactly 2 heads in 5 coin tosses."

text = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False
)
```

```python
_ = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=text,
    verbose=True,
    max_tokens=512
)
```

```python
opt = optim.Muon(learning_rate=2e-4) # Use a low learning rate for LoRA fine-tuning, since we don't want to change the base model weights too much. Adjust if necessary.

# Training arguments. Adjust these based on your dataset size, GPU capacity, and how long you want to train.
args = SFTTrainingArgs(
    batch_size=1, # Use batch size of 1 to save RAM, increase if you have more GPU memory
    iters=calculate_iters(train_set, batch_size=1, epochs=1), # Only train for 1 epoch since the dataset is small, increase if you want to train longer
    gradient_accumulation_steps=8, # Accumulate gradients over 8 steps to simulate a larger batch size and save RAM. Adjust based on your GPU capacity.
    val_batches=1, # Only use 1 batch for validation to speed it up, since the dataset is small. Remove or increase for better evaluation.
    steps_per_report=10, # Log training progress every 10 steps
    steps_per_eval=20, # Evaluate every 20 steps
    steps_per_save=10, # Save the model every 10 steps
    max_seq_length=max_seq_length,
    adapter_file=adapter_file,
    grad_checkpoint=True, # Use gradient checkpointing to save RAM at the cost of slightly slower training
    seq_step_size=512, # This makes it efficient, increasing it makes th emodel learn better but uses more RAM. Suggested 512, 1024, 2048
)

# Start training
train_sft(
    model=model,
    args=args,
    optimizer=opt,
    train_dataset=CacheDataset(train_set),
    val_dataset=CacheDataset(valid_set),
    # training_callback=WandBCallback(
    #     project_name=f"{new_model_name}-finetuning",
    #     log_dir=adapter_path,
    #     wrapped_callback=TrainingCallback(),
    #     config=None
    # )
    training_callback=TrainingCallback(), # You can use the basic TrainingCallback to log training progress to the console instead of Weights & Biases. Just comment out the WandBCallback and uncomment this line if you prefer that.
)
```

## Step 7: Test the Fine-Tuned Model 🎯

Training complete! Now let's test the fine-tuned model on the same prompt to see the improvement.

The model now has the LoRA adapter weights applied, so its responses should be:
- More accurate and detailed
- Better aligned with the instruction-following pattern from the training data
- More structured and consistent

Compare this output with the base model output from Step 5. You should see noticeable improvements in:
- **Answer quality**: More precise calculations and explanations
- **Instruction following**: Better adherence to the prompt format
- **Reasoning**: Clearer step-by-step problem solving

Let's see the results! 🌟

```python
_ = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=text,
    verbose=True,
    max_tokens=1024
)
```

## Step 8: Merge and Save the Full Model (Optional)

The LoRA adapter is just a small file containing the fine-tuned weights. For deployment, you might want to merge it with the base model to create a standalone model.

This step:
1. **Merges** the LoRA adapter weights back into the base model
2. **De-quantizes** the model back to full precision (optional, set to `False` to keep quantized)
3. **Saves** the complete merged model locally

**Important Notes:**
- The merged model will be much larger (~8-15GB for 4B models)
- You don't need to merge if you're only sharing the adapter (more common and efficient)
- The adapter file alone is typically only 10-100MB!

**When to merge:**
- ✅ For deployment where you want a standalone model
- ✅ For maximum inference speed
- ❌ For sharing (just share the adapter instead!)
- ❌ For experimenting with multiple adapters

```python
save_pretrained_merged(
    model=model,
    tokenizer=tokenizer,
    save_path=adapter_path,
    de_quantize=True,
)
```

## Step 9: Share Your Model on Hugging Face Hub 🤗

Time to share your fine-tuned adapter with the community!

This step:
1. **Uploads** your LoRA adapter to Hugging Face Hub
2. **Creates** a model card with metadata
3. **Makes it discoverable** for others to use

**Before running this cell:**
1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Get your API token from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Replace `"HF_KEY"` with your actual token, or better yet, set it as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

**Parameters:**
- **`model_path`**: Path to your adapter+merged model
- **`hf_repo`**: Your Hugging Face username/model-name
- **`private`**: Set to `True` if you want a private model
- **`remove_adapters`**: Removes the adapter files before uploading

Once uploaded, anyone can use your model with [MLX-LM](https://github.com/ml-explore/mlx-lm) and [LM Studio](https://lmstudio.ai)

🎉 Congratulations on creating your first fine-tuned model!

```python
push_to_hub(
  model_path=adapter_file,
  hf_repo=f"{user_name}/{new_model_name}",
  api_key="HF_KEY",
  private=False,
  commit_message="Add fine-tuned LoRA adapter",
  remove_adapters=True
)
```

## 🎉 Congratulations! You've Fine-Tuned a 4B Model on Apple Silicon!

### What You've Accomplished:
✅ Fine-tuned a 4B parameter model with **32K+ context length**  
✅ Used only **~12-16GB of RAM** thanks to quantization and LoRA  
✅ Trained on custom instruction data efficiently  
✅ Saved and shared your model adapter  

### Key Takeaways:
- **LoRA** makes fine-tuning 100x more efficient than full fine-tuning
- **4-bit quantization** enables huge models on consumer hardware
- **MLX-LM-LoRA** optimizes Apple Silicon's unified memory architecture
- You can handle **extremely long contexts** (32K+ tokens) efficiently

### Next Steps:
1. **Experiment with different datasets** - try your own data!
2. **Adjust hyperparameters** - try different LoRA ranks, learning rates, and batch sizes
3. **Train for more epochs** - remove `.take()` and train on the full dataset
4. **Try larger models** - 7B, 8B, 14B models work great with 24GB+ RAM
5. **Combine multiple adapters** - load different adapters for different tasks

### Resources:
- 📚 [MLX-LM-LoRA Documentation](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- 🤗 [Hugging Face Model Hub](https://huggingface.co/models)

### Performance Tips:
- Increase `batch_size` if you have more RAM (M2/M3/M4/M5)
- Adjust `seq_step_size` for better speed/memory tradeoff
- Use `gradient_accumulation_steps` to simulate larger batches
- Enable `grad_checkpoint` for ultra-long contexts

Happy fine-tuning! 🚀✨

---

**Tip:** Save this notebook as a template and modify it for your own fine-tuning projects!
