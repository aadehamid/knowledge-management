title: Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation
description: Beginner's Guide for creating a customized personal assistant \(like ChatGPT\) to run locally on Ollama

# Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation

By the end of this tutorial, you will create a custom chatbot by **finetuning Llama-3** with [**Unsloth**](https://github.com/unslothai/unsloth) for free. It can run locally via [**Ollama**](https://github.com/ollama/ollama) on your PC, or in a free GPU instance through [**Google Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_%288B%29-Ollama.ipynb). You will be able to interact with the chatbot interactively like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-cf9aed2029e54afbb65889b480134e6d5e1cf3a7%252FAssistant%2520example.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=655bb08f30801957a468edd6fc642687&sv=3){width=2462 height=826}

**Unsloth** makes finetuning much easier, and can automatically export the finetuned model to **Ollama** with integrated automatic `Modelfile` creation! If you need help, you can join our Discord server: [https://discord.com/invite/unsloth](https://discord.com/invite/unsloth)

## 1. What is Unsloth? {#id-1.-what-is-unsloth}

[Unsloth](https://github.com/unslothai/unsloth) makes finetuning LLMs like Llama-3, Mistral, Phi-3 and Gemma 2x faster, use 70% less memory, and with no degradation in accuracy! We will be using Google Colab which provides a free GPU during this tutorial. You can access our free notebooks below:

#### ***You will also need to login into your Google account!*** {#you-will-also-need-to-login-into-your-google-account}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-bca149bda83c2192982b136cfeb096999c469a2e%252FColab%2520Screen.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=22a85e61cf9b727419b8f0e977be4339&sv=3){width=2880 height=1358}

## 2. What is Ollama? {#id-2.-what-is-ollama}

[Ollama ](https://github.com/ollama/ollama)allows you to run language models from your own computer in a quick and simple way! It quietly launches a program which can run a language model like Llama-3 in the background. If you suddenly want to ask the language model a question, you can simply submit a request to Ollama, and it'll quickly return the results to you! We'll be using Ollama as our inference engine!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-fd25844766001d93ed0949fc8f57957f49b1e6e5%252FOllama.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=4a66fa3dbf0a36f6a64ac1acd1df717a&sv=3){width=2880 height=1358}

## 3. Install Unsloth {#id-3.-install-unsloth}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4d1b1778f3c8bde62a40130d7b4395b8bb1ce90f%252FColab%2520Options.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=7661f002b76d4c1c400d448e39b407b5&sv=3){width=2869 height=1358}

If you have never used a Colab notebook, a quick primer on the notebook itself:

1. **Play Button at each "cell".** Click on this to run that cell's code. You must not skip any cells and you must run every cell in chronological order. If you encounter any errors, simply rerun the cell you did not run before. Another option is to click CTRL \+ ENTER if you don't want to click the play button.
2. **Runtime Button in the top toolbar.** You can also use this button and hit "Run all" to run the entire notebook in 1 go. This will skip all the customization steps, and can be a good first try.
3. **Connect / Reconnect T4 button.** You can click here for more advanced system statistics.

The first installation cell looks like below: Remember to click the PLAY button in the brackets \[ \]. We grab our open source Github package, and install some other packages.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-3ae88d2cf9ba1c59b13d701864750ac311a60426%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=5da7f39d49de12164052ed70892f5b92&sv=3){width=2542 height=302}

## 4. Selecting a model to finetune {#id-4.-selecting-a-model-to-finetune}

Let's now select a model for finetuning! We defaulted to Llama-3 from Meta / Facebook which was trained on a whopping 15 trillion "tokens". Assume a token is like 1 English word. That's approximately 350,000 thick Encyclopedias worth! Other popular models include Mistral, Phi-3 (trained using GPT-4 output) and Gemma from Google (13 trillion tokens!).

Unsloth supports these models and more! In fact, simply type a model from the Hugging Face model hub to see if it works! We'll error out if it doesn't work.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-4fb10a1ce3e457310c11f74ca5b6347ad556fab0%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=3ec477a99d72f38d7f53094e9ab9d0c1&sv=3){width=1725 height=1100}

There are 3 other settings which you can toggle:

1. This determines the context length of the model. Gemini for example has over 1 million context length, whilst Llama-3 has 8192 context length. We allow you to select ANY number - but we recommend setting it 2048 for testing purposes. Unsloth also supports very long context finetuning, and we show we can provide 4x longer context lengths than the best.
2. Keep this as None, but you can select torch.float16 or torch.bfloat16 for newer GPUs.
3. We do finetuning in 4 bit quantization. This reduces memory usage by 4x, allowing us to actually do finetuning in a free 16GB memory GPU. 4 bit quantization essentially converts weights into a limited set of numbers to reduce memory usage. A drawback of this is there is a 1-2% accuracy degradation. Set this to False on larger GPUs like H100s if you want that tiny extra accuracy.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a44ac84348a2c5973dd542866c4c6727a00b3744%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=372c5775504ba1161da0582b3ef8e640&sv=3){width=2063 height=709}

If you run the cell, you will get some print outs of the Unsloth version, which model you are using, how much memory your GPU has, and some other statistics. Ignore this for now.

## 5. Parameters for finetuning {#id-5.-parameters-for-finetuning}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-495edc79c5353f0f47c1eea58df045631bfef1e0%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=2efe452221b6eb8d9a2639cba1d7b634&sv=3){width=1980 height=728}

Now to customize your finetune, you can edit the numbers above, but you can ignore it, since we already select quite reasonable numbers.

The goal is to change these numbers to increase accuracy, but also **counteract over-fitting**. Over-fitting is when you make the language model memorize a dataset, and not be able to answer novel new questions. We want to a final model to answer unseen questions, and not do memorization.

1. The rank of the finetuning process. A larger number uses more memory and will be slower, but can increase accuracy on harder tasks. We normally suggest numbers like 8 (for fast finetunes), and up to 128. Too large numbers can causing over-fitting, damaging your model's quality.
2. We select all modules to finetune. You can remove some to reduce memory usage and make training faster, but we highly do not suggest this. Just train on all modules!
3. The scaling factor for finetuning. A larger number will make the finetune learn more about your dataset, but can promote over-fitting. We suggest this to equal to the rank `r`, or double it.
4. Leave this as 0 for faster training! Can reduce over-fitting, but not that much.
5. Leave this as 0 for faster and less over-fit training!
6. Options include `True`, `False` and `"unsloth"`. We suggest `"unsloth"` since we reduce memory usage by an extra 30% and support extremely long context finetunes.You can read up here: [https://unsloth.ai/blog/long-context](https://unsloth.ai/blog/long-context) for more details.
7. The number to determine deterministic runs. Training and finetuning needs random numbers, so setting this number makes experiments reproducible.
8. Advanced feature to set the `lora_alpha = 16` automatically. You can use this if you want!
9. Advanced feature to initialize the LoRA matrices to the top r singular vectors of the weights. Can improve accuracy somewhat, but can make memory usage explode at the start.

## 6. Alpaca Dataset {#id-6.-alpaca-dataset}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-1d66d8714e44d90513dd87b9356eec67886ab3f7%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=65dee9d3748c0899b97f5393aaddf45b&sv=3){width=2314 height=760}

We will now use the Alpaca Dataset created by calling GPT-4 itself. It is a list of 52,000 instructions and outputs which was very popular when Llama-1 was released, since it made finetuning a base LLM be competitive with ChatGPT itself.

You can access the GPT4 version of the Alpaca dataset here: [https://huggingface.co/datasets/vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4). An older first version of the dataset is here: [https://github.com/tatsu-lab/stanford\_alpaca](https://github.com/tatsu-lab/stanford_alpaca). Below shows some examples of the dataset:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-0dde50e386e7b245d3e8a57e10a4a81755b3769a%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=dc2276ba9a189f8ef0e04bc45a3bd175&sv=3){width=2848 height=1108}

You can see there are 3 columns in each row - an instruction, and input and an output. We essentially combine each row into 1 large prompt like below. We then use this to finetune the language model, and this made it very similar to ChatGPT. We call this process **supervised instruction finetuning**.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-8b3663c5d80adcb935ff77661500f08e13c9af2d%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=fce8bc7c932bceca9fa944ca84bfc502&sv=3){width=2738 height=585}

## 7. Multiple columns for finetuning {#id-7.-multiple-columns-for-finetuning}

But a big issue is for ChatGPT style assistants, we only allow 1 instruction / 1 prompt, and not multiple columns / inputs. For example in ChatGPT, you can see we must submit 1 prompt, and not multiple prompts.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-d90162c2685ced871f4151369aadcaee40a9c54f%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=f41ea7565cbdebc5e3f6a175c2533d96&sv=3){width=2880 height=1357}

This essentially means we have to "merge" multiple columns into 1 large prompt for finetuning to actually function!

For example the very famous Titanic dataset has many many columns. Your job was to predict whether a passenger has survived or died based on their age, passenger class, fare price etc. We can't simply pass this into ChatGPT, but rather, we have to "merge" this information into 1 large prompt.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a2df04874bfc879182cb66c789341d49700227ea%252FMerge.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=42ff0f5e620940722bfc0ad5f376839a&sv=3){width=1328 height=174}

For example, if we ask ChatGPT with our "merged" single prompt which includes all the information for that passenger, we can then ask it to guess or predict whether the passenger has died or survived.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-b3da2b36afe37469cd3962f37186e758871864a5%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=cc46252dbddde585443d7b0bed518df7&sv=3){width=2880 height=1357}

Other finetuning libraries require you to manually prepare your dataset for finetuning, by merging all your columns into 1 prompt. In Unsloth, we simply provide the function called `to_sharegpt` which does this in 1 go!

To access the Titanic finetuning notebook or if you want to upload a CSV or Excel file, go here: [https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp\=sharing](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing)

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

## 8. Multi turn conversations {#id-8.-multi-turn-conversations}

An issue if you didn't notice is the Alpaca dataset is single turn, whilst remember using ChatGPT was interactive and you can talk to it in multiple turns. For example, the left is what we want, but the right which is the Alpaca dataset only provides singular conversations. We want the finetuned language model to somehow learn how to do multi turn conversations just like ChatGPT.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2a65cd74ddd03a6bcbbc9827d9d034e4879a8e6a%252Fdiff.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=272c0899dbc6f24df221565132d13109&sv=3){width=2764 height=849}

So we introduced the `conversation_extension` parameter, which essentially selects some random rows in your single turn dataset, and merges them into 1 conversation! For example, if you set it to 3, we randomly select 3 rows and merge them into 1! Setting them too long can make training slower, but could make your chatbot and final finetune much better!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2b1b3494b260f1102942d86143a885225c6a06f2%252Fcombine.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=972ccd628098be5751bf41a0398d8dbf&sv=3){width=2752 height=852}

Then set `output_column_name` to the prediction / output column. For the Alpaca dataset dataset, it would be the output column.

We then use the `standardize_sharegpt` function to just make the dataset in a correct format for finetuning! Always call this!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-7bf83bf802191bda9e417bbe45afa181e7f24f38%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=97bc5c308ff2ff723d0199b90de8417f&sv=3){width=1900 height=392}

## 9. Customizable Chat Templates {#id-9.-customizable-chat-templates}

We can now specify the chat template for finetuning itself. The very famous Alpaca format is below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-59737e6dcb09fed15487d5a57c69f07cb40bb8e7%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=271cc813125d84381c6d5d25f4c5843d&sv=3){width=2738 height=585}

But remember we said this was a bad idea because ChatGPT style finetunes require only 1 prompt? Since we successfully merged all dataset columns into 1 using Unsloth, we essentially can create the below style chat template with 1 input column (instruction) and 1 output:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-d54582ae98c396d51bfb85628b46c54f2517d030%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=65afe56e29e9f71965b033754ee434ba&sv=3){width=2385 height=562}

We just require you must put a `{INPUT}` field for the instruction and an `{OUTPUT}` field for the model's output field. We in fact allow an optional `{SYSTEM}` field as well which is useful to customize a system prompt just like in ChatGPT. For example, below are some cool examples which you can customize the chat template to be:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-cc455dc380d3d44ef136e485754964159dc773d8%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=d19b8f982c5c2e94f203fc9b97ac5bd9&sv=3){width=1648 height=237}

For the ChatML format used in OpenAI models:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-15bfca9cfadf10d54b4d3f66e3050044317d62c5%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=486b035dd39c688261c978dcddae6b5f&sv=3){width=1554 height=414}

Or you can use the Llama-3 template itself (which only functions by using the instruct version of Llama-3): We in fact allow an optional `{SYSTEM}` field as well which is useful to customize a system prompt just like in ChatGPT.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-80a2ed4de2ca323ac192c513cac65e9e8bf475db%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=6eb3271025f1737e4085dbbdb063bed7&sv=3){width=1914 height=454}

Or in the Titanic prediction task where you had to predict if a passenger died or survived in this Colab notebook which includes CSV and Excel uploading: [https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp\=sharing](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing)

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-20911ab305c1a10e85859c703157b80175141eb1%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=f95a254cf239facbd35fac2db0800e28&sv=3){width=2451 height=554}

## 10. Train the model {#id-10.-train-the-model}

Let's train the model now! We normally suggest people to not edit the below, unless if you want to finetune for longer steps or want to train on large batch sizes.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-f55503cea4d84b5885d0bcea0563fd716a0d2ed6%252Fimage%2520%2843%29.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=315efbc9249cacf63b8b9a1817d4f2b2&sv=3){width=1467 height=1127}

We do not normally suggest changing the parameters above, but to elaborate on some of them:

1. Increase the batch size if you want to utilize the memory of your GPU more. Also increase this to make training more smooth and make the process not over-fit. We normally do not suggest this, since this might make training actually slower due to padding issues. We normally instead ask you to increase `gradient_accumulation_steps` which just does more passes over the dataset.
2. Equivalent to increasing the batch size above itself, but does not impact memory consumption! We normally suggest people increasing this if you want smoother training loss curves.
3. We set steps to 60 for faster training. For full training runs which can take hours, instead comment out `max_steps`, and replace it with `num_train_epochs = 1`. Setting it to 1 means 1 full pass over your dataset. We normally suggest 1 to 3 passes, and no more, otherwise you will over-fit your finetune.
4. Reduce the learning rate if you want to make the finetuning process slower, but also converge to a higher accuracy result most likely. We normally suggest 2e-4, 1e-4, 5e-5, 2e-5 as numbers to try.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-feb9b0f5763d41cecaec9a3a9cd227ad918f0ca7%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=532fef9e2847314dd2481babe6c877c5&sv=3){width=1382 height=968}

You’ll see a log of numbers during training. This is the training loss, which shows how well the model is learning from your dataset. For many cases, a loss around 0.5 to 1.0 is a good sign, but it depends on your dataset and task. If the loss is not going down, you might need to adjust your settings. If the loss goes to 0, that could mean overfitting, so it's important to check validation too.

## 11. Inference / running the model {#id-11.-inference-running-the-model}

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-f2d5f23fa62ec89e06bf20fea433f9a1e42a2fe3%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=4b3c1daeb3bdbb75fef3913530a9cfd1&sv=3){width=2524 height=724}

Now let's run the model after we completed the training process! You can edit the yellow underlined part! In fact, because we created a multi turn chatbot, we can now also call the model as if it saw some conversations in the past like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-cdf5d779635901dce7793df92531dbf3caf0fb0a%252Fimage%2520%2847%29.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=550e180a8b26c3914310745b808499a8&sv=3){width=2509 height=846}

Reminder Unsloth itself provides **2x faster inference** natively as well, so always do not forget to call `FastLanguageModel.for_inference(model)`. If you want the model to output longer responses, set `max_new_tokens = 128` to some larger number like 256 or 1024. Notice you will have to wait longer for the result as well!

## 12. Saving the model {#id-12.-saving-the-model}

We can now save the finetuned model as a small 100MB file called a LoRA adapter like below. You can instead push to the Hugging Face hub as well if you want to upload your model! Remember to get a Hugging Face token via [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and add your token!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-8c577103f7c4fe883cabaf35c8437307c6501686%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=c45907e48ea5d416f5dc32aa737e7d41&sv=3){width=2155 height=685}

After saving the model, we can again use Unsloth to run the model itself! Use `FastLanguageModel` again to call it for inference!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-1a1be852ca551240bdce47cf99e6ccd7d31c1326%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=c2ec65ab33451f07159854446b9c20ab&sv=3){width=2300 height=1054}

## 13. Exporting to Ollama {#id-13.-exporting-to-ollama}

Finally we can export our finetuned model to Ollama itself! First we have to install Ollama in the Colab notebook:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-24f9429ed4a8b3a630dc8f68dcf81555da0a80ee%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=d5e7fdb9a9e41790da5812fc2fc3c2e2&sv=3){width=2605 height=680}

Then we export the finetuned model we have to llama.cpp's GGUF formats like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-56991ea7e2685bb9905af9baf2f3f685123dcdd8%252Fimage%2520%2852%29.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=1697084fc41cc13f14030bd407cbfde5&sv=3){width=1990 height=961}

Reminder to convert `False` to `True` for 1 row, and not change every row to `True`, or else you'll be waiting for a very time! We normally suggest the first row getting set to `True`, so we can export the finetuned model quickly to `Q8_0` format (8 bit quantization). We also allow you to export to a whole list of quantization methods as well, with a popular one being `q4_k_m`.

Head over to [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) to learn more about GGUF. We also have some manual instructions of how to export to GGUF if you want here: [https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf)

You will see a long list of text like below - please wait 5 to 10 minutes!!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-271b392fdafd0e7d01c525d7a11a97ee5c34b713%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=b65c264e3802d1dd29872bf40f2830d6&sv=3){width=2025 height=964}

And finally at the very end, it'll look like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-a554bd388fd0394dd8cdef85fd9d208bfd7feee7%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=d352ecf9bd7b17f63d9ae851f6f40cbf&sv=3){width=1872 height=790}

Then, we have to run Ollama itself in the background. We use `subprocess` because Colab doesn't like asynchronous calls, but normally one just runs `ollama serve` in the terminal / command prompt.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e431609dfc5c742f0b5ab2388dbbd0d8e15c7670%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=4a645c6e92f7fbc5031163d9faa783ad&sv=3){width=2045 height=340}

## 14. Automatic `Modelfile` creation {#id-14.-automatic-modelfile-creation}

The trick Unsloth provides is we automatically create a `Modelfile` which Ollama requires! This is a just a list of settings and includes the chat template which we used for the finetune process! You can also print the `Modelfile` generated like below:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-6945ba10a2e25cfc198848c0e863001375c32c4c%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=39f59bfe3de38f16d93db9806e9b690e&sv=3){width=2610 height=792}

We then ask Ollama to create a model which is Ollama compatible, by using the `Modelfile`

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-d431a64613b39d913d1780c22cde37edc6564272%252Fimage.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=08705f2a7c8e22e2d4f5a79f7e009e66&sv=3){width=2410 height=455}

## 15. Ollama Inference {#id-15.-ollama-inference}

And we can now call the model for inference if you want to do call the Ollama server itself which is running on your own local machine / in the free Colab notebook in the background. Remember you can edit the yellow underlined part.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-49b93efa192fdd741f3ac8484cef8c3fd7415283%252FInference.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=a2efb43628d4cf8e9c9e67054143df3e&sv=3){width=2584 height=822}

## 16. Interactive ChatGPT style {#id-16.-interactive-chatgpt-style}

But to actually run the finetuned model like a ChatGPT, we have to do a bit more! First click the terminal icon![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-9c24108bc5152f946a7afab054974890318d2c02%252Fimage.png%3Falt%3Dmedia&width=300&dpr=3&quality=100&sign=3e2163fe2c23dc35b7214d02daffe5db&sv=3){width=68 height=53} and a Terminal will pop up. It's on the left sidebar.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-2239315eff2820bf9f224975f0b184d51bd89cb7%252FWhere_Terminal.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=b1590fb9f61c574db9483804527d59cb&sv=3){width=2880 height=1206}

Then, you might have to press ENTER twice to remove some weird output in the Terminal window. Wait a few seconds and type `ollama run unsloth_model` then hit ENTER.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-e83ac484e4257eacad1c7d033811d2ece59a444c%252FTerminal_Type.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=a74c12f22a34e7205d6a58d2fa31cc41&sv=3){width=2880 height=948}

And finally, you can interact with the finetuned model just like an actual ChatGPT! Hit CTRL \+ D to exit the system, and hit ENTER to converse with the chatbot!

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-120703475091e1ce74a38a05949ae51af0a36f72%252FAssistant.png%3Falt%3Dmedia&width=768&dpr=3&quality=100&sign=ae5cabb43c20f5ae4bd1cae3df4f53e4&sv=3){width=2880 height=941}

## You've done it! {#youve-done-it}

You've successfully finetuned a language model and exported it to Ollama with Unsloth 2x faster and with 70% less VRAM! And all this for free in a Google Colab notebook!

If you want to learn how to do reward modelling, do continued pretraining, export to vLLM or GGUF, do text completion, or learn more about finetuning tips and tricks, head over to our [Github](https://github.com/unslothai/unsloth#-finetune-for-free).

If you need any help on finetuning, you can also join our Discord server [here](https://discord.gg/unsloth). If you want help with Ollama, you can also join their server [here](https://discord.gg/ollama).

And finally, we want to thank you for reading and following this far! We hope this made you understand some of the nuts and bolts behind finetuning language models, and we hope this was useful!

To access our Alpaca dataset example click [here](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing), and our CSV / Excel finetuning guide is [here](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing).
