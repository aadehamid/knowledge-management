title: KV Cache Explained Intuitively. An intuitive walkthrough of how… | by Saad Ahmed Siddiqui | Medium
description: KV Cache Explained Intuitively Table of contents:- Tokens & Embeddings Idea about Decoder-Only Models Intuition about Attention Mechanism Inference in Basic Language Models Why It’s Inefficient …
author: Saad Ahmed Siddiqui

# KV Cache Explained Intuitively

## Table of contents:- {#0c0d}

- Tokens & Embeddings
- Idea about Decoder-Only Models
- Intuition about Attention Mechanism
- Inference in Basic Language Models
- Why It’s Inefficient Without KV Cache
- KV Cache
- Conclusion
- References

## Let’s Start {#8c91}

There’s a lot of jargon that gets thrown around when talking about language models - terms like “tokens,” “embeddings,” “encoder-decoder,” “attention,” and the list goes on.

So first I’ll try to familiarize you with like the very basic and commonly used terminologies knowing which will make life much easier.

This blog is for anyone who is interested in understanding the fundamental building blocks of language models like ChatGPT.\
And yes - even people who are not from an ML background are absolutely welcome!

Instead of just straight up jumping into KV Cache first we’ll take a look at some prerequisite stuff.

## Tokens & Embeddings {#314b}

As some of you might be aware, a machine learning model can only understand numbers and not natural language.

We’re all used to interacting with LLMs and stuff using natural language prompts like “*do this for me*” or “*explain this to me*” but we don’t just feed this directly into the model.

First we need to process it such that the model can understand it.

So we need to convert our prompt into some sort of numerical representation that the model can work with.

### Tokens {#7f5e}

We use a process called tokenization in which we divide the input sequence into small chunks called tokens.

In a very naive sense you can think of this as breaking the sentence into words.Though it’s not actually that simple, this is just to get a rough idea into your head.Just wanted to make that clear before people start coming after me*.*I’m just trying to give a basic idea.

### Embeddings {#bcf5}

Each of these tokens is assigned a numerical representation.

So we end up with a vector for each token which is called an embedding.

An embedding is basically a vector having some number of dimensions which captures the meaning of a word.

Essentially each of these dimensions in the embedding represent some latent feature which helps distinguish between different embeddings.

Let’s try to understand with an example

Though in real-world settings, embeddings usually consist of thousands of dimensions and we don’t know which dimension represents which property, the model can still use these latent features to distinguish between different embeddings effectively.

You can think of it as a point lying in some space and basically if two words are closely related to each other(like cat and dog) then their embedding will be near each other in that space whereas if the words are not related (like horse and building) then their embeddings will be far away .

We also add this thing called **positional encodings** to the input embeddings so that each word carries some information about its position in the sentence. Though we won’t be going into the specifics of it.

> Now I know y’all must be like: *“Okay okay, we get it - just get to the good part already. We’re here for the language model… enough of this!”* \
> So without further ado, let’s get into the real fun stuff.

## Idea About Decoder-Only Models {#61c2}

Since this blog is about **KV Cache**,it’s important to first have a basic understanding of the **Decoder**.So for a moment, I’ll be removing all the complicated stuff and let’s just try to get a feel for what a decoder does.

Below is a very naive sort of diagram showing a decoder.One thing to note is that it’s intentionally simplified-for example, I’ve used **masked self-attention** to keep things easy to follow. In practice, it’s more common to see **multi-head attention** or even things like **grouped multi-query attention** (don’t worry about the details if you’re not familiar with all this jargon).

Also, I’ve chosen to treat components like **skip connections**, **feedforward networks**, **normalization**, etc. as a bit of a black box here. Again, don’t worry if you find all these terms intimidating - we’re not going to be needing them.This blog is focused specifically on **KV Cache**, and I don’t want to get sidetracked from that main objective. So I’ll try to keep the focus on just the relevant parts.

A decoder takes some tokens as input and based on this it tries to predict which token is the most likely to appear next in the [sequence.It](http://sequence.It) keeps repeating this process until it generates the entire output sequence.

For example, if we pass the sequence ‘*the cat sat on the*’ to the decoder, it will probably predict something like ‘*mat*’ assuming the model has been trained properly.

Now let’s take a look at how this happens:

So the input embeddings are passed as input to the decoder. But these embeddings have a problem.

The input embeddings capture the meaning of the word they represent - but not its relationship with other words in the sequence or the context in which it is used.

For example, consider the following sentences:

*Only Saad wants coffee.* ⇒ This means that **only Saad** wants coffee - no one else.

*Saad only wants coffee.* ⇒ This means that Saad wants **only coffee - he doesn’t want anything else.**

We use this thing called **attention mechanism**(which we will take a look at in the next section)to convert these input embeddings into contextualized embeddings.

Contextualized embedding means each token becomes aware of its context in the sentence and its relationship with other tokens.

After this we do some “*stuff*” and what we get are these raw outputs called logits.These logits are then converted into **probabilities**, based on which the model generates the next token.

So this was a very basic, intuitive way of looking at a **decoder**.

> Now I know y’all must be left with questions like:\
>  *“Wait, what just happened? How did we go from our input embeddings to these contextualized ones? What is this ‘attention’ thing everyone keeps talking about?”* \
>  So ladies and gentlemen - here we go.

## Masked Self Attention {#dd2b}

Now it is time to look at **attention mechanism**. I’ll try to make it as intuitive as possible.

So first let’s understand what the input looks like.

We know that each token in the input sequence is converted into an embedding with some number of dimensions (say `d_model`).\
 So if we have `N` tokens in our input, then the input is essentially a matrix of size `N x d_model`. No need to panic if you don't get it - let's break it down.

We had `N` embeddings, each with `d_model` dimensions. So basically, what we did is stack these `N` embeddings - each having `d_model` dimensions - together to get a matrix of size `N x d_model`.

For each input embedding, we create three vectors called the query vector, key vector, and value vector. This is done by multiplying the input embeddings with matrices `Wq`, `Wk`, and `Wv` that are learned during training of the model.

Let’s get a feel for the **query** and **key** vectors.

So imagine all the tokens in the sequence are people - the first one is *Saad*, the next one is *Saad’s friend*, then there is *some random guy*, then there is *another random guy*, and then there is *Saad’s neighbour*.

Out of nowhere, *Saad* just shouts, “**Who is important to me?**” So *Saad’s friend* replies, “**I am super important!**”

Then *Saad’s neighbour* says, “**Though I guess I’m not that important, we do know each other - if that has any significance.**” And of course, *Saad* doesn’t care about the two random guys.

So here, the one doing the looking i.e. “*Saad”* is the **query**.

The ones being looked at - everyone, including *Saad* (which might feel weird, but yeah, in self-attention the query even looks at itself) - are the **keys**.

In this diagram, what we can see is:

- **Saad and his friend are strongly related.**
- **The archnemesis is also strongly related to Saad, but in the opposite way.**
- **Saad is literally Saad, so yes, he is related to himself.**
- **The NPCs are just chilling in the background.**
- **The neighbour is, in a way, related to Saad, but not as strongly as the friend or the archnemesis.**

### Self Attention {#5bfe}

Now that we’ve gotten a bit of intuition, let’s get into the actual math.

As we saw at the start of this section, we multiply the input `X` with matrices `Wq`, `Wk`, and `Wv` to get the queries, keys, and values.

When `X` (having shape `N x d_model`) is multiplied with `Wq` (having shape `d_model x d_k`), we get the matrix `Q` (of shape `N x d_k`), where each row is the query vector(having `d_k` dimensions) corresponding to a particular token.

When `X` is multiplied with `Wk` (shape `d_model x d_k`), we get the matrix `K` (shape `N x d_k`), where each row is the key vector(having `d_k` dimensions) corresponding to a particular token.

When `X` is multiplied with `Wv` (shape `d_model x d_v`), we get the matrix `V` (shape `N x d_v`), where each row is the value vector(having `d_v` dimensions) corresponding to a particular token.

> *So now we are going to take a look at the formula for*  ***attention*** *. It can look intimidating, but don’t worry, we’ll go through it step by step:*

First we will be taking a look at the following matrix :-

So after multiplying **Q** and the transpose of **K** we get a matrix which contains raw attention scores. Each of these raw attention scores tell us how strong is the relationship between one word and another.

Let’s try to understand with an example.

Consider that the input sequence is “*Saad is drinking coffee*”.

In the diagram shown below, I’ve tried to explain what each element of this matrix means and how it is calculated.

After this, we divide each of the raw attention scores by the square root of `d_k`, which ensures more stable training.

Now let’s talk about the **softmax** function, which looks like this:

It might look scary at first, but it’s actually pretty straightforward.

What it does is convert our raw attention scores into values between 0 and 1, in such a way that all the values in each **row** sum up to 1.

The resulting matrix still contains attention scores, which still tell us how strong the relationship is between one word and another , it’s just that now the values are between 0 and 1, and each row sums to 1.

Now we multiply this matrix with the **value matrix**. Let’s try to build some intuition for this step. Some words in the sequence will be strongly related to a particular token and will have **high attention scores**. Others might be related but not as strongly, and will have **lower attention scores**. Finally, some words may be completely unrelated, and their attention scores will be **very close to zero** (something like 0.0001).

When we perform this matrix multiplication, we are essentially **weighing the value vectors** by how important (relevant) they are for the current token:

- **Relevant tokens (high scores)** contribute more to the final output vector.
- **Irrelevant tokens (near-zero scores)** contribute almost nothing

Let’s see how the **contextualized embedding** is calculated for “*Saad*” along one of the dimensions.

We’ll assume that `d_v`\= 2, i.e., the **value vectors** have 2 dimensions-let’s call these dimensions d1 and d2.

In the diagram above, I’ve shown the calculation of the contextualized embedding for “*Saad*” along the dimension d1.

So basically, what’s happening is:

Earlier, the embedding of “Saad” only contained information about the **meaning of “Saad”** and its **position in the sentence**.

But now, we’re weighing how important each of the other tokens is - and based on those attention scores, some of those tokens **strongly influence** the final embedding (if they are related to “Saad”), while others **barely contribute** (if they’re not related to “Saad”).

What we get at the end is a **contextualized embedding** where each token becomes aware of its **context** with respect to the other tokens in the sequence.

### Masked Self Attention {#673a}

Now that we’ve understood **self-attention**, let’s see what **masked self-attention** is.

We don’t want the model to *“cheat”* by looking at **future tokens** when predicting the next word. So we apply a **mask** to prevent it from attending to tokens that come **after** the current one.

For example, in the sequence:

**“pokemon gotta catch em all”**

- The word **“pokemon”** can only look at **itself**, because all other tokens are *future tokens -* so they are **masked**.
- The word **“gotta”** can look at **“pokemon”** and **itself**.
- The word **“catch”** can look at **“pokemon”**, **“gotta”**, and **itself**.
- And so on.

Now, how we implement this **mask** is as follows:

Before applying the **softmax**, we set the values we want to hide (i.e., the positions corresponding to future tokens) to **-infinity**.

When softmax is applied, these values get converted to **0**, effectively removing any contribution from the **masked (future) tokens** in the attention calculation.

So now, hopefully, you’ve got a good intuition for how **attention** works and how **masked self-attention** prevents the model from peeking into the future.

This kind of setup - where a model only attends to past and present tokens - is known as causal attention, and models that use this are often referred to as causal language models.

> I know that was a lot to digest, so now we’re moving on to something a bit more fun and relatively easier.

## Inference in Language Models {#e004}

**Inference** is just a fancy way of saying we’re using a trained model to make **predictions** on some data.

For example, if you ask an LLM *“What is the capital of France?”* and the model replies with *“Paris”*, that’s an example of **inference**.

Now, let’s take a look at how inference works in language models.

I’m pretty sure you’re already familiar with **tokens**. Usually, when working with language models, we use some **special tokens**. In particular, two important tokens to be aware of are:

- `<SOS>` - **Start of Sentence**
- `<EOS>` - **End of Sentence**

The `<SOS>` token is added to the beginning of the input sequence, and `<EOS>` is used to signal the end of generation, so the model knows **when to stop** producing output.

Language models use a **decoder-only architecture**, and we’ve already gotten a general idea that it predicts the **next token** in a sequence.

Now let’s see how this actually happens.

Let’s say the user enters:\
 “I am”

First,this sequence is tokenized.

Then, we add a `<SOS>` (Start of Sentence) token at the beginning, so the input sequence becomes:\
 `"<SOS> I am"`

Next, each token is converted into an embedding.

We then pass this sequence to the model.\
The model calculates attention and does some more “stuff”. It then outputs raw outputs called logits .

We take the logits for the last token in the sequence and apply softmax to them. This gives us probabilities over the vocabulary.\
 The model then selects the token with the highest probability - let’s say it’s “drinking”.\
 This token is now appended to the input sequence.

Now the input becomes:\
 `"<SOS> I am drinking"`

We pass this updated sequence into the model again. The same process happens.\
 Let’s say the model now predicts “coffee”.\
 We append this again, so the input becomes:\
 `"<SOS> I am drinking coffee"`

We pass this into the model again, and this time it predicts the `<EOS>` token.\
 This tells the model that the sentence is complete, so it stops generating.

There are different **inference strategies** to choose the next token - such as:

- **Greedy decoding**: always pick the token with the highest probability.
- **Top-k sampling**: pick from the top *k* most probable tokens.
- **Beam search**: explores multiple possible sequences in parallel and picks the best one overall.
- **Top-p (nucleus) sampling**

But for simplicity, we are going with **greedy decoding** here.

> I hope by now you’ve got some idea of how a language model works. Now, you’re probably wondering - “Okay, we get how the model works, but what’s this KV Cache business all about?”
> So without further ado

## Why It’s Inefficient Without KV Cache {#6320}

At every time step during inference, we are only interested in the **last token output** by the model - because we already know the previous ones. However, the model still needs to **access all the previous tokens** to decide which token to generate next, since they form the context.

This becomes inefficient because the model ends up **recomputing attention values** for tokens that it has already processed in earlier time steps.

Let’s look at an example to understand this better.

In particular, we’ll focus on the part where attention is calculated during inference - because **that’s where the inefficiency lies**.

As shown in the diagram, at every time step after `t=1`, the model:

- Recalculates the attention values **for all the past tokens**, which were already computed in previous steps
- And in addition, calculates attention **for the new token**

This means the model is **doing redundant work -** recomputing attention weights for the same tokens again and again, instead of reusing them.

This is exactly what makes inference **inefficient** when **KV cache** is not used.

## KV Cache {#c107}

So now let’s get a feel for how we might go about getting rid of the inefficiency that we just saw.

So what exactly **is** KV Cache?

Instead of recalculating the **key** and **value** vectors for all the previous tokens at every time step, what if we simply **stored** them the first time they were computed?

That’s the basic idea behind KV Cache.

Also, to ensure that we only calculate attention for the **new token**, we use **only the query vector** of the new token during attention computation

This way, we only calculate **the attention values that we actually need**.

Every time we pass a token as input to the self-attention mechanism, we:

- Compute its **key** and **value** vectors
- **Store them in a buffer** called the **KV Cache**

Let’s take a look at an example for clarity:

- At **t \= 1**, we pass `"<SOS> I"` to the model.
- The model computes the key and value vectors for both `<SOS>` and `"I"`, and stores them in the **KV Cache**.
- After calculating attention and doing all the other “stuff,” let’s say the model predicts **"am"**.

At **t \= 2**, we input `"am"` to the model.

- The model computes the key and value vectors for `"am"` and appends them to the cache.
- For attention:
- We use the **query** vector for `"am"`
- We use the **cached key and value vectors** from `<SOS>`, `"I"`, and `"am"`
- The model predicts the next token - let’s say **"driving"**.

At **t \= 3**, we repeat the process:

- Compute key and value for `"driving"`
- Append them to the KV Cache
- Use the query for `"driving"` , and the entire cache as key-value inputs

And this continues until we reach the end of the sequence.

So once again, the main idea behind **KV Cache** is to **store the key and value vectors** that we’ve already calculated, so that we can **reuse them** during inference.

This way, when a new token arrives the only thing we have to do is

- calculate the key and value vector for this new token. Add it to the KV cache. Use the entire buffer as the keys and values.
- Calculate the **query vector** for the new token and use it to compute attention over the cached keys and values.

Now let’s take a look at self attention with KV Cache :

> *So from the above diagram, it should be clear how KV Cache makes inference in language models more efficient by computing*  ***only what is actually required*** *.*

We have 2 phases in KV Cache

**Prefilling**

- Usually, the prompt sent by the user contains **more than one token** (if we’re being realistic).
- At the **initial time step**, when we receive the full user prompt, we compute the **key** and **value** vectors for **all the tokens** in the prompt and add them to the **KV Cache** at once.

**Token Generation**

- After the prompt is processed, the model starts **generating one token at a time**.
- In the subsequent time steps, we only compute the **key** and **value** vectors for the **newly generated token**, and then append these to the existing **KV Cache** (which already contains the keys and values from the earlier steps).

So I guess that is all about KV Cache.

## Conclusion {#e990}

So it’s almost time to wrap up the blog. Let’s quickly go through the main ideas in a language model once again:

You start with a **user prompt**, which is broken down into **tokens**. Then you add the `<SOS>` token to the beginning of the sequence.

This sequence of tokens is converted into **numerical representations** called **embeddings**, which the model can understand. We also add **positional encodings** to the embeddings so the model knows the position of each token in the sequence.

Then we use the **attention mechanism** to convert these embeddings into **contextualized embeddings**.

To overcome inefficiencies during inference, we use **KV Cache**.

After computing the attention values , the model does some more “stuff” and we get **raw outputs** called **logits**. We take the logit corresponding to the **last token**, apply **softmax** to convert it into **probabilities**, and pick the token with the highest probability as the next predicted token.

That predicted token is added to the input sequence, and the whole process repeats until the model generates the `<EOS>` token.

> I guess that’s all from my side.
> If you actually made it this far - **thank you**, I really appreciate it!!

If you liked my work, you can follow me on for more such blogs and cool ML-related stuff!

## References {#c877}

- [**LLaMA Video by Umar Jamil**](https://www.youtube.com/watch?v=Mn_9W1nCFLo)
- [**The Illustrated Transformer by Jay Alammar**](https://jalammar.github.io/illustrated-transformer/)
- [**Wolfe, Cameron R. *Language Model Training and Inference***](https://cameronrwolfe.substack.com/p/language-model-training-and-inference)
- [**Tokenization docs by Mistral**](https://docs.mistral.ai/guides/tokenization/)
- [**Understanding and Coding the KV Cache in LLMs from Scratch by Sebastian Raschka**](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
- [**Key Query Value Attention Explained by Alex-AI**](https://www.youtube.com/watch?v=H-4bmOxiKyU)
- [**Attention is all you need Explained by Umar Jamil**](https://www.youtube.com/watch?v=bCz4OMemCcA&t=1491s)
