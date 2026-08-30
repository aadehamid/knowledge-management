title: Datasets Guide | Unsloth Documentation
description: Learn how to create & prepare a dataset for fine-tuning.

# Datasets Guide | Unsloth Documentation

## What is a Dataset? {#what-is-a-dataset}

For LLMs, datasets are collections of data that can be used to train our models. In order to be useful for training, text data needs to be in a format that can be tokenized. You'll also learn how to [use datasets inside of Unsloth](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide#applying-chat-templates-with-unsloth).

One of the key parts of creating a dataset is your [chat template](https://docs.unsloth.ai/docs/basics/chat-templates) and how you are going to design it. Tokenization is also important as it breaks text into tokens, which can be words, sub-words, or characters so LLMs can process it effectively. These tokens are then turned into embeddings and are adjusted to help the model understand the meaning and context.

### Data Format {#data-format}

To enable the process of tokenization, datasets need to be in a format that can be read by a tokenizer.

## Getting Started {#getting-started}

Before we format our data, we want to identify the following:

Purpose of dataset

Knowing the purpose of the dataset will help us determine what data we need and format to use.

The purpose could be, adapting a model to a new task such as summarization or improving a model's ability to role-play a specific character. For example:

- Chat-based dialogues (Q&A, learn a new language, customer support, conversations).
- Domain-specific data (medical, finance, technical).

Style of output

The style of output will let us know what sources of data we will use to reach our desired output.

For example, the type of output you want to achieve could be JSON, HTML, text or code. Or perhaps you want it to be Spanish, English or German etc.

Data source

When we know the purpose and style of the data we need, we need to analyze the quality and [quantity](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide#how-big-should-my-dataset-be) of the data. Hugging Face and Wikipedia are great sources of datasets and Wikipedia is especially useful if you are looking to train a model to learn a language.

The Source of data can be a CSV file, PDF or even a website. You can also [synthetically generate](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide#synthetic-data-generation) data but extra care is required to make sure each example is high quality and relevant.

One of the best ways to create a better dataset is by combining it with a more generalized dataset from Hugging Face like ShareGPT to make your model smarter and diverse. You could also add [synthetically generated data](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide#synthetic-data-generation).

## 🦥 Unsloth Data Recipes {#unsloth-data-recipes}

[Unsloth Data Recipes](https://docs.unsloth.ai/docs/new/studio/data-recipe) lets you upload documents like PDFs or CSVs files and transforms them into useable datasets. Create and edit datasets visually via a graph-node workflow.

The recipes page is the main entry point. Recipes are stored locally in the browser, so you come back to saved work later. From here, you can create a blank recipe or open a guided learning recipe.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fc5m3JX1kUA3UwmdcJcxH%252FArea.gif%3Falt%3Dmedia%26token%3D33bbd908-7d6c-456a-bc58-ce495c0adca1&width=768&dpr=3&quality=100&sign=aeff17c67b50b3c8bde23664f0df50ad&sv=3){width=1832 height=1080}

Data Recipes follows the same basic path. You open the recipes page, create or pick a recipe, build the workflow in the editor, validate it run a preview, then run the full dataset once the output looks right. Add seed data and generation blocks, validate the workflow, preview sample output, then run a full dataset build.

At a glance a usual workflow should look like this:

1. Open the recipes page.
2. Create a new recipe or open an existing one.
3. Add blocks to define your dataset workflow.
4. Click **Validate** to catch configuration issues early.
5. Run a preview to inspect sample rows quickly.
6. Run a full dataset build when the recipe is ready.
7. Review progress and output live in graph or in **Executions** view for mode details.
8. Select the resulting dataset in Unsloth and fine tune a model. Read more:

[Data Recipes](https://docs.unsloth.ai/docs/new/studio/data-recipe)

## Formatting the Data {#formatting-the-data}

When we have identified the relevant criteria, and collected the necessary data, we can then format our data into a machine readable format that is ready for training.

### Common Data Formats for LLM Training {#common-data-formats-for-llm-training}

For [**continued pretraining**](https://docs.unsloth.ai/docs/basics/continued-pretraining), we use raw text format without specific structure:

This format preserves natural language flow and allows the model to learn from continuous text.

If we are adapting a model to a new task, and intend for the model to output text in a single turn based on a specific set of instructions, we can use **Instruction** format in [Alpaca style](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset)

When we want multiple turns of conversation we can use the ShareGPT format:

The template format uses the "from"/"value" attribute keys and messages alternates between `human`and `gpt`, allowing for natural dialogue flow.

The other common format is OpenAI's ChatML format and is what Hugging Face defaults to. This is probably the most used format, and alternates between `user` and `assistant`

### Applying Chat Templates with Unsloth {#applying-chat-templates-with-unsloth}

For datasets that usually follow the common chatml format, the process of preparing the dataset for training or finetuning, consists of four simple steps:

- 
Check the chat templates that Unsloth currently supports:\ This will print out the list of templates currently supported by Unsloth. Here is an example output:\\- 
Use `get_chat_template` to apply the right chat template to your tokenizer:\\- 
Define your formatting function. Here's an example:\ This function loops through your dataset applying the chat template you defined to each sample.\- 
Finally, let's load the dataset and apply the required modifications to our dataset: \ If your dataset uses the ShareGPT format with "from"/"value" keys instead of the ChatML "role"/"content" format, you can use the `standardize_sharegpt` function to convert it first. The revised code will now look as follows: \
### Formatting Data Q&A {#formatting-data-q-and-a}

**Q:** How can I use the Alpaca instruct format?

**A:** If your dataset is already formatted in the Alpaca format, then follow the formatting steps as shown in the Llama3.1 [notebook ](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb#scrollTo=LjY75GoYUCB8). If you need to convert your data to the Alpaca format, one approach is to create a Python script to process your raw data. If you're working on a summarization task, you can use a local LLM to generate instructions and outputs for each example.

**Q:** Should I always use the standardize\_sharegpt method?

**A:** Only use the standardize\_sharegpt method if your target dataset is formatted in the sharegpt format, but your model expect a ChatML format instead.

 **Q:** Why not use the apply\_chat\_template function that comes with the tokenizer.

**A:** The `chat_template` attribute when a model is first uploaded by the original model owners sometimes contains errors and may take time to be updated. In contrast, at Unsloth, we thoroughly check and fix any errors in the `chat_template` for every model when we upload the quantized versions to our repositories. Additionally, our `get_chat_template` and `apply_chat_template` methods offer advanced data manipulation features, which are fully documented on our Chat Templates documentation [page](https://docs.unsloth.ai/basics/chat-templates).

**Q:** What if my template is not currently supported by Unsloth?

**A:** Submit a feature request on the unsloth github issues [forum](https://github.com/unslothai/unsloth). As a temporary workaround, you could also use the tokenizer's own apply\_chat\_template function until your feature request is approved and merged.

## Synthetic Data Generation {#synthetic-data-generation}

You can also use any local LLM like Llama 3.3 (70B) or OpenAI's GPT 4.5 to generate synthetic data. Generally, it is better to use a bigger like Llama 3.3 (70B) to ensure the highest quality outputs. You can directly use inference engines like vLLM, Ollama or llama.cpp to generate synthetic data but it will require some manual work to collect it and prompt for more data. There's 3 goals for synthetic data:

- Produce entirely new data - either from scratch or from your existing dataset
- Diversify your dataset so your model does not [overfit](https://docs.unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#avoiding-overfitting-and-underfitting) and become too specific
- Augment existing data e.g. automatically structure your dataset in the correct chosen format

### Using Unsloth for synthetic data {#using-unsloth-for-synthetic-data}

You can easily upload any unstructured or structured data into Unsloth Studio's [Data Recipes](https://docs.unsloth.ai/docs/new/studio/data-recipe) and it will automatically convert it into a useable / synthetic dataset. More details in [our guide](https://docs.unsloth.ai/docs/new/studio/data-recipe).

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FQ6e19jESrJg0VjHnX58c%252Fdata%2520recipes%2520final.png%3Falt%3Dmedia%26token%3D8d74e453-815d-4790-83d1-76d0bc80a3ce&width=768&dpr=3&quality=100&sign=b6655f499dccbe2336650237557f35cd&sv=3){width=2704 height=1454}

### Using a local LLM or ChatGPT for synthetic data {#using-a-local-llm-or-chatgpt-for-synthetic-data}

Your goal is to prompt the model to generate and process QA data that is in your specified format. The model will need to learn the structure that you provided and also the context so ensure you at least have 10 examples of data already. Examples prompts:

- **Prompt for generating more dialogue on an existing dataset**:
- 
**Prompt if you no have dataset**:{% code overflow\="wrap" %}{% endcode %}- 
**Prompt for a dataset without formatting**:{% code overflow\="wrap" %}{% endcode %}
It is recommended to check the quality of generated data to remove or improve on irrelevant or poor-quality responses. Depending on your dataset it may also have to be balanced in many areas so your model does not overfit. You can then feed this cleaned dataset back into your LLM to regenerate data, now with even more guidance.

## Dataset FAQ \+ Tips {#dataset-faq--tips}

### How big should my dataset be? {#how-big-should-my-dataset-be}

We generally recommend using a bare minimum of at least 100 rows of data for fine-tuning to achieve reasonable results. For optimal performance, a dataset with over 1,000 rows is preferable, and in this case, more data usually leads to better outcomes. If your dataset is too small you can also add synthetic data or add a dataset from Hugging Face to diversify it. However, the effectiveness of your fine-tuned model depends heavily on the quality of the dataset, so be sure to thoroughly clean and prepare your data.

### How should I structure my dataset if I want to fine-tune a reasoning model? {#how-should-i-structure-my-dataset-if-i-want-to-fine-tune-a-reasoning-model}

If you want to fine-tune a model that already has reasoning capabilities like the distilled versions of DeepSeek-R1 (e.g. DeepSeek-R1-Distill-Llama-8B), you will need to still follow question/task and answer pairs however, for your answer you will need to change the answer so it includes reasoning/chain-of-thought process and the steps it took to derive the answer. For a model that does not have reasoning and you want to train it so that it later encompasses reasoning capabilities, you will need to utilize a standard dataset but this time without reasoning in its answers. This is training process is known as [Reinforcement Learning and GRPO](https://docs.unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).

### Multiple datasets {#multiple-datasets}

If you have multiple datasets for fine-tuning, you can either:

- Standardize the format of all datasets, combine them into a single dataset, and fine-tune on this unified dataset.

### Can I fine-tune the same model multiple times? {#can-i-fine-tune-the-same-model-multiple-times}

You can fine-tune an already fine-tuned model multiple times, but it's best to combine all the datasets and perform the fine-tuning in a single process instead. Training an already fine-tuned model can potentially alter the quality and knowledge acquired during the previous fine-tuning process.

## Using Datasets in Unsloth {#using-datasets-in-unsloth}

### Alpaca Dataset {#alpaca-dataset}

See an example of using the Alpaca dataset inside of Unsloth on Google Colab:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-1d66d8714e44d90513dd87b9356eec67886ab3f7%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=65dee9d3748c0899b97f5393aaddf45b&sv=3){width=2314 height=760}

We will now use the Alpaca Dataset created by calling GPT-4 itself. It is a list of 52,000 instructions and outputs which was very popular when Llama-1 was released, since it made finetuning a base LLM be competitive with ChatGPT itself.

You can access the GPT4 version of the Alpaca dataset [here](https://huggingface.co/datasets/vicgalle/alpaca-gpt4.). Below shows some examples of the dataset:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-0dde50e386e7b245d3e8a57e10a4a81755b3769a%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=dc2276ba9a189f8ef0e04bc45a3bd175&sv=3){width=2848 height=1108}

You can see there are 3 columns in each row - an instruction, and input and an output. We essentially combine each row into 1 large prompt like below. We then use this to finetune the language model, and this made it very similar to ChatGPT. We call this process **supervised instruction finetuning**.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-8b3663c5d80adcb935ff77661500f08e13c9af2d%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=fce8bc7c932bceca9fa944ca84bfc502&sv=3){width=2738 height=585}

### Multiple columns for finetuning {#multiple-columns-for-finetuning}

But a big issue is for ChatGPT style assistants, we only allow 1 instruction / 1 prompt, and not multiple columns / inputs. For example in ChatGPT, you can see we must submit 1 prompt, and not multiple prompts.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-d90162c2685ced871f4151369aadcaee40a9c54f%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=f41ea7565cbdebc5e3f6a175c2533d96&sv=3){width=2880 height=1357}

This essentially means we have to "merge" multiple columns into 1 large prompt for finetuning to actually function!

For example the very famous Titanic dataset has many many columns. Your job was to predict whether a passenger has survived or died based on their age, passenger class, fare price etc. We can't simply pass this into ChatGPT, but rather, we have to "merge" this information into 1 large prompt.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a2df04874bfc879182cb66c789341d49700227ea%252FMerge.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=42ff0f5e620940722bfc0ad5f376839a&sv=3){width=1328 height=174}

For example, if we ask ChatGPT with our "merged" single prompt which includes all the information for that passenger, we can then ask it to guess or predict whether the passenger has died or survived.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-b3da2b36afe37469cd3962f37186e758871864a5%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=cc46252dbddde585443d7b0bed518df7&sv=3){width=2880 height=1357}

Other finetuning libraries require you to manually prepare your dataset for finetuning, by merging all your columns into 1 prompt. In Unsloth, we simply provide the function called `to_sharegpt` which does this in 1 go!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-62b94dc44f2e343020d31de575f52eb22be4b0fc%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=11be32b4c5e65b59afa48f0f661db386&sv=3){width=1836 height=557}

Now this is a bit more complicated, since we allow a lot of customization, but there are a few points:

- You must enclose all columns in curly braces `{}`. These are the column names in the actual CSV / Excel file.
- Optional text components must be enclosed in `[[]]`. For example if the column "input" is empty, the merging function will not show the text and skip this. This is useful for datasets with missing values.
- Select the output or target / prediction column in `output_column_name`. For the Alpaca dataset, this will be `output`.

For example in the Titanic dataset, we can create a large merged prompt format like below, where each column / piece of text becomes optional.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e6228cf6e5c0bb4e4b45e6f3e045910d567c33d2%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=d0d0bb8130e16058c44f269899576a7c&sv=3){width=2175 height=730}

For example, pretend the dataset looks like this with a lot of missing data:

Then, we do not want the result to be:

1. The passenger embarked from S. Their age is 23. Their fare is **EMPTY**.
2. The passenger embarked from **EMPTY**. Their age is 18. Their fare is $7.25.

Instead by optionally enclosing columns using `[[]]`, we can exclude this information entirely.

1. \[\[The passenger embarked from S.\]\] \[\[Their age is 23.\]\] \[\[Their fare is **EMPTY**.\]\]
2. \[\[The passenger embarked from **EMPTY**.\]\] \[\[Their age is 18.\]\] \[\[Their fare is $7.25.\]\]

becomes:

1. The passenger embarked from S. Their age is 23.
2. Their age is 18. Their fare is $7.25.

### Multi turn conversations {#multi-turn-conversations}

An issue if you didn't notice is the Alpaca dataset is single turn, whilst remember using ChatGPT was interactive and you can talk to it in multiple turns. For example, the left is what we want, but the right which is the Alpaca dataset only provides singular conversations. We want the finetuned language model to somehow learn how to do multi turn conversations just like ChatGPT.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2a65cd74ddd03a6bcbbc9827d9d034e4879a8e6a%252Fdiff.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=272c0899dbc6f24df221565132d13109&sv=3){width=2764 height=849}

So we introduced the `conversation_extension` parameter, which essentially selects some random rows in your single turn dataset, and merges them into 1 conversation! For example, if you set it to 3, we randomly select 3 rows and merge them into 1! Setting them too long can make training slower, but could make your chatbot and final finetune much better!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2b1b3494b260f1102942d86143a885225c6a06f2%252Fcombine.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=972ccd628098be5751bf41a0398d8dbf&sv=3){width=2752 height=852}

Then set `output_column_name` to the prediction / output column. For the Alpaca dataset dataset, it would be the output column.

We then use the `standardize_sharegpt` function to just make the dataset in a correct format for finetuning! Always call this!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-7bf83bf802191bda9e417bbe45afa181e7f24f38%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=97bc5c308ff2ff723d0199b90de8417f&sv=3){width=1900 height=392}

## Vision Fine-tuning {#vision-fine-tuning}

The dataset for fine-tuning a vision or multimodal model also includes image inputs. For example, the [Llama 3.2 Vision Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%2811B%29-Vision.ipynb#scrollTo=vITh0KVJ10qX) uses a radiography case to show how AI can help medical professionals analyze X-rays, CT scans, and ultrasounds more efficiently.

We'll be using a sampled version of the ROCO radiography dataset. You can access the dataset [here](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Funsloth%2FRadiology_mini). The dataset includes X-rays, CT scans and ultrasounds showcasing medical conditions and diseases. Each image has a caption written by experts describing it. The goal is to finetune a VLM to make it a useful analysis tool for medical professionals.

Let's take a look at the dataset, and check what the 1st example shows:

To format the dataset, all vision finetuning tasks should be formatted as follows:

We will craft an custom instruction asking the VLM to be an expert radiographer. Notice also instead of just 1 instruction, you can add multiple turns to make it a dynamic conversation.

Let's convert the dataset into the "correct" format for finetuning:

The first example is now structured like below:

Before we do any finetuning, maybe the vision model already knows how to analyse the images? Let's check if this is the case!

And the result:

For more details, view our dataset section in the [notebook here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%2811B%29-Vision.ipynb#scrollTo=vITh0KVJ10qX).
