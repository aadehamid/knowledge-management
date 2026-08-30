title: The Anatomy of an LLM | Interactive Visual Guide to How Language Models Work
description: An interactive visual explainer for developers showing how LLMs work, from tokenization and embeddings to attention, transformers, training, KV cache, and quantization.

# The Anatomy of an LLM | Interactive Visual Guide to How Language Models Work

Introduction

## Introduction {#intro-heading}

 Large language models can feel like black boxes. You type a prompt, something smart comes back, and somewhere in the middle billions of parameters supposedly did "AI". 

This guide opens that box.

 We will follow one chain from beginning to end. First, text is split into tokens. Those tokens become vectors. The vectors move through layers of attention and feed-forward networks. At the end, the model produces scores for possible next tokens, and a decoding strategy chooses what comes out. 

 The goal is not to memorize every formula. The goal is to understand what changes at each step, and why that step exists at all. 

 If you are looking for how LLMs work, how transformers work, or how attention, tokenization, KV cache, and quantization fit together, this page keeps those ideas connected in one visual path. 

By the end, you should be able to trace the full path:

01 **Text**  02 **Tokens**  03 **Vectors**  04 **Transformer blocks**  05 **Logits**  06 **Sampling**  07 **Output**

 And once you can trace that path, the black box becomes a lot smaller. 

What you get

Concrete visuals, small numbers first, and interactive controls that make each transformation inspectable.

How to use it

Scroll top to bottom as a single narrative, or jump between chapters for a specific concept.

Who made this

 Roy van Rijn working at [openvalue](https://www.openvalue.eu) 

Table of contents

1.  [ 01 Tokenization ](#tokenization) 
2.  [ 02 Vector Embeddings ](#embeddings) 
3.  [ 03 Neuron Activation ](#neuron-activation) 
4.  [ 04 Feed-Forward Neural Network ](#feed-forward-network) 
5.  [ 05 Logits and Sampling ](#logits) 
6.  [ 06 Backpropagation ](#backpropagation) 
7.  [ 07 Optimizers ](#optimizers) 
8.  [ 08 Attention: Q, K, and V ](#qkv) 
9.  [ 09 Multi-Head Attention ](#multi-head-attention) 
10.  [ 10 RoPE ](#rope) 
11.  [ 11 Transformer Block ](#transformer-block) 
12.  [ 12 Training Phases ](#training-phases) 
13.  [ 13 Post-Training ](#post-training) 
14.  [ 14 Context and KV Cache ](#kv-cache) 
15.  [ 15 Quantization ](#quantization) 

Chapter 01

## Tokenization {#chapter-01}

Before a model can think about text, the text has to become numbers.

 A language model does not read words and sentences the way we do. It reads a sequence of token IDs: integers produced by a tokenizer. 

 That makes tokenization the real entrance to the model. Everything after this point works with numbers, not raw characters. 

 A token can be a whole word, part of a word, punctuation, whitespace, or a piece of something strange like code, emoji, or a name. This is why tokenization often looks a bit weird when you first see it. The tokenizer is not trying to split text the way a human would. It is trying to represent text efficiently using a fixed vocabulary. 

 If every token were a full word, the vocabulary would explode. If every token were a single character or byte, every sentence would become very long. Modern tokenizers live between those extremes. 

Slicing up the text

Before text can enter a language model, it has to be rewritten as numbers.

Tokenization is the step that does this. It splits text into small reusable pieces called **tokens**. A token can be a whole word, part of a word, punctuation, a number, or even a space plus the start of the next word.

Each token has an entry in the tokenizer's vocabulary and is replaced by its corresponding integer ID. From that point on, the model is no longer working with characters directly. It sees an ordered list of token IDs.

Why not just use words?

Whole words are too rigid. New names, typos, code, inflections, compound words, and multilingual text would constantly produce words the model has never seen before.

Why not just use letters or bytes?

That solves the "unknown word" problem, but makes every input much longer. More pieces means more work for the model and less context fits in the same window. Subword tokens are the reasonable compromise: common text stays compact, while unusual text can still be built from smaller pieces.

Below you can experiment with OpenAI's `o200k_base` tokenizer. Try switching sentences and watch where the boundaries land.

Later in this explainer, when the model predicts the *next* token, it predicts over this same vocabulary.

Technical note: the examples below are generated with [`tiktoken`](https://github.com/openai/tiktoken) using the `o200k_base` encoding.

Raw sentence

If the human brain were so simple that we could understand it, we would be so simple that we couldn't.

**102** characters

**22** tokens

**5** chars/token on average

Tokenized result

If

\#3335

·the

\#290

·human

\#5396

·brain

\#12891

·were

\#1504

·so

\#813

·simple

\#4705

·that

\#484

·we

\#581

·could

\#2023

·understand

\#4218

·it

\#480

\#11

·we

\#581

·would

\#1481

·be

\#413

·so

\#813

·simple

\#4705

·that

\#484

·we

\#581

·couldn't

\#21149

\#13

Important takeaway

 Tokenization is not just preprocessing. It determines what the model can see in one context window, how expensive your text is, and which pieces the model is allowed to predict next. 

One word is not one token

 Different models use different tokenizers. The same sentence can become a different number of tokens depending on the model. 

Chapter 02

## Vector Embeddings {#chapter-02}

Token IDs are just labels. Embeddings turn those labels into something the network can work with.

 After tokenization, every token is represented by an integer ID. But an ID by itself has no useful geometry. Token `15339` is not "close to" token `15340` in any meaningful way. The numbers are just labels, like row numbers in a table. 

 The embedding layer solves this by turning each token ID into a vector: a list of learned numbers. Technically, this is a lookup. The model has an embedding matrix, and each token ID selects one row from that matrix. 

 Conceptually, this is the moment where discrete symbols enter a continuous space. Once tokens become vectors, the model can compare them, combine them, rotate them, project them, and gradually reshape them. 

 The values inside these vectors are learned during training. Tokens that appear in similar contexts often end up with related vectors, but this is not a clean dictionary of meanings. It is more like a messy, high-dimensional coordinate system full of useful signals. 

 The initial embedding is mostly context-free. The token "bank" starts with the same embedding in "river bank" and "investment bank". Later layers use surrounding tokens to rewrite that vector into something more specific. 

From token ID to embedding vector

Embedding lookup

After tokenization, each token ID is used as an index into an embedding table. The selected row is a high-dimensional vector that becomes the model's starting representation for that token.

For readability, this chapter uses a toy embedding width of 24 dimensions. Real model widths are usually much larger, common production widths include 768, 1024, 1536, 3072, and even higher.

An embedded vector is just a list of floating point numbers: dog \= \[0.7292, -0.3786, 0.1065, 0.3674, 0.1902, -0.7881, ... \]

Example sentence  Token in sentence 

 -> -> 

Embedding values (24 dimensions)

This explainer shows all 24 values from the toy vector.

The same token ID always maps to the same embedding vector.

In real models, these embedding values are learned during training. Tokens that appear in similar contexts are gradually moved to useful regions of vector space, so the vectors end up encoding patterns the model can build on.

Tokens that often play similar roles get nudged in similar directions. For example, the tokens cat, dog, and rabbit often appear in sentence templates like "The \_\_\_ is sleeping", "I fed the \_\_\_", or "The \_\_\_ ran away". Because they appear in similar contexts, their vectors may end up close together.

But cat and car usually appear in very different contexts, so their vectors tend to end up farther apart.

The embedding space is not hand-designed. Nobody tells the model “put animals over here” or “put verbs over there”. Those patterns emerge because moving the vectors that way helps the model predict text better.

2D analogy intuition

Distances between embedding vectors often similar if they have a similar relationship.

Important takeaway

 An embedding is the token's starting representation, not its final meaning. The rest of the model will keep rewriting that vector as context flows through the network. 

Toy scale

 In this explainer we use small vectors because they fit on screen. Real models use much wider vectors: hundreds, thousands, or more dimensions per token. 

Chapter 03

## Neuron Activation {#chapter-03}

A weighted sum is not enough. The non-linearity is where the network gets expressive.

 A neuron takes inputs, multiplies them by weights, adds them together, and produces a number. But if that were the whole story, deep learning would not be very deep. 

 Without activation functions, stacking layers would still behave like one large linear transformation. You could multiply matrices together and collapse the whole stack into a single matrix. 

 The activation function breaks that linearity. It decides how much of a signal passes through. Some values are amplified, some are softened, some are pushed toward zero. 

 This lets the network build curved, conditional, non-linear transformations instead of only scaling and rotating vectors. Real models do this in huge batches using matrix operations, with millions of activations happening at once. 

Single-neuron transformation

A neuron takes inputs, applies weights, and then runs the result through an activation function. This non-linear step is what lets networks model richer patterns.

`z = w1*x1 + w2*x2 + w3*x3`

`output = activation(z)`

Neuron diagram

Inputs

x1 0.70 x2 -0.25 x3 0.45

Weights

w1 1.10 w2 -0.85 w3 0.55

Activation

Smoothly gates values by magnitude instead of hard clipping. Common in transformer blocks; a bit heavier to compute than ReLU.

Neuron output **1.0953**

Activation curve

Marker position updates live as the weighted input `z` changes.

Important takeaway

 The activation function is not decoration. It is what lets stacked layers become more than one big linear calculation. 

Modern choices

 Modern transformer models may use GELU, SiLU, or gated variants like SwiGLU. The exact choice changes both the forward signal and how gradients flow during training. 

Chapter 04

## Feed-Forward Neural Network {#chapter-04}

A real layer is not one neuron. It is many simple computations running in parallel.

 A single neuron is a useful teaching tool, but models do not process one neuron at a time. A feed-forward network applies many learned transformations in parallel. 

 Instead of drawing every neuron and every connection, implementations usually express the same thing as matrix multiplication. The friendly diagram says inputs flow through neurons. The implementation says multiply a matrix, apply an activation, multiply another matrix. 

 Those are the same story at different scales. 

 In transformer blocks, the feed-forward part usually works position by position. Each token vector is expanded into a wider hidden representation, passed through a non-linearity, and projected back to the model width. 

 Attention moves information between positions. The feed-forward network transforms the information inside each position. 

Dense layer math, visually

Instead of training a full network here, we focus on one forward pass. A dense layer simply means every node in one layer connects to every node in the next layer.

The same math from one neuron is now done in parallel using matrices:

`X(1x2) · W1(2x3) = Z1(1x3)`, then `A1 = activation(Z1)`, then `A1(1x3) · W2(3x2) = Z2(1x2)`.

Matrix multiplication is just many weighted sums at once. Each output column is one neuron, and each row in the input contributes through its matching weight row.

Fully connected view

Hover the top labels to inspect matrices. Green border means firing, red means suppressed.

x1 0.80 x2 -0.30 Activation 

Matrix inspector

Hover one of the top labels (`X`, `W1`, `A1`, `W2`, `A2`) to inspect that matrix and the multiplication step.

How multiplication maps to connections

Column `j` in `W1` contains weights feeding hidden neuron `j`. Row `i` corresponds to input feature `i`. So each hidden pre-activation is: `z1_j = x1*w1_1j + x2*w1_2j`.

The second layer repeats that pattern with `A1` as input: `z2_k = a1_1*w2_1k + a1_2*w2_2k + a1_3*w2_3k`. This is exactly the graph computation, just vectorized.

In matrix form, we avoid writing each neuron separately: `[x1 x2] · W1 = [z1_1 z1_2 z1_3]`, then activation applies element-wise to produce `A1`. That `A1` row is then multiplied by `W2` to produce both output neurons at once.

Example from the current sliders: `z1_1 = +0.80*+0.70 + -0.30*+0.10 = +0.53`. If activation suppresses this value (for example ReLU on negative values), that path contributes less or zero to the next layer.

Important takeaway

 The feed-forward network is where each token vector gets rewritten. It is not about moving information between tokens; it is about transforming the representation at each token position. 

Matrix view

 The matrix view is not a less intuitive version of the neuron diagram. It is the scalable version of the same computation. 

Chapter 05

## Logits and Sampling {#chapter-05}

The model does not directly output a word. It outputs scores for possible next tokens.

 After the model has processed the input, it still has not chosen a word. What it has produced is a vector of raw scores: one score for every token in the vocabulary. These scores are called logits. 

 A logit is not a probability. It is just an unnormalized score. Higher usually means "the model thinks this token fits better here", but the numbers do not yet add up to 100%. 

 To turn logits into probabilities, we apply softmax. Then comes decoding: the policy for choosing the next token from that distribution. 

 Greedy decoding always picks the most likely token. Temperature changes the shape of the distribution. Top-k limits the choice to the k most likely tokens. Top-p, also called nucleus sampling, chooses from the smallest group of tokens whose total probability passes a threshold. 

 The model produces the distribution. The decoder decides how adventurous we are when sampling from it. 

From logits to generated output

A model converts the final hidden vector into one score per vocabulary token. Those raw scores are logits. Softmax turns them into probabilities, and sampling chooses the next token.

01 **Hidden**  02 **Vocab projection**  03 **Logits**  04 **Softmax(T)**  05 **Probabilities**  06 **Sampled token**

Temperature 1.00 Mode Top-k (optional) 

Logits

 calm  `0.81`

 inside  `0.34`

 outside  `0.49`

 cold  `-0.45`

 angry  `-0.84`

.  `-0.06`

,  `-0.03`

Probabilities (after softmax)

 calm  `27.3%`

 inside  `17.1%`

 outside  `19.6%`

 cold  `7.7%`

 angry  `5.2%`

.  `11.4%`

,  `11.7%`

Sampled output

Generated sequence (10 tokens): `(click generate)`

Important takeaway

 The model usually does not contain one fixed answer. At each generation step, it produces a probability distribution over possible next tokens. 

Token by token

 A chatbot answer is built one token at a time. After each sampled token, the new token is added to the context and the process repeats. 

Chapter 06

## Backpropagation {#chapter-06}

To learn, the model needs to know which parameters helped cause the mistake.

 Training starts with a simple question: how wrong was the model? 

 The model predicts a distribution over the next token. We know which token actually came next in the training text. The loss measures how far the prediction was from that target. 

 But measuring the loss is not enough. The model has billions of parameters. Which ones should change? And by how much? 

 Backpropagation answers that question. It sends the error signal backward through the computation graph and calculates gradients: how sensitive the loss is to each parameter. 

 The core idea is the chain rule. Every operation only needs to know how its output changes with respect to its input. By chaining those local derivatives together, training can calculate how a tiny change deep inside the model would affect the final loss. 

Error becomes learning signal

We will train on one tiny example and reveal each step in order: forward prediction, backward gradients, then the weight update.

01 **Select target**  02 **Forward snapshot**  03 **Calculate backward**  04 **Apply update**

Step 1 - Predict from input

Three, two, one... \_\_\_  Learning rate 0.35

Important takeaway

 Backpropagation is not a second mysterious intelligence inside the model. It is an efficient way to calculate gradients through a large composed computation. 

Three passes

 Forward pass: make a prediction. Backward pass: calculate how to change the parameters. Optimizer step: actually change them. 

Chapter 07

## Optimizers {#chapter-07}

Gradients point downhill. Optimizers decide how to walk.

 A gradient tells us which direction should reduce the loss. But it does not fully answer how to update the model. 

 How big should the step be? Should we trust the current gradient completely? Should we remember previous gradients? What if different parameters have wildly different gradient scales? 

 That is the job of the optimizer. 

 SGD, or stochastic gradient descent, is the simplest common version. It looks at a small batch of training examples, calculates the gradient, and takes one step in the direction that should reduce the loss. It is direct and easy to understand, but each step can be noisy because it only sees a slice of the training data. 

 Momentum improves on this by remembering direction. If gradients keep pointing roughly the same way, momentum builds speed. If they zigzag, momentum smooths the path. 

 Adam tracks both a moving average of the gradients and a moving estimate of their scale. That lets it adapt update sizes per parameter. 

 The optimizer is not just a detail after backpropagation. It is part of the learning behavior. 

Different update rules, same gradients

Backprop gives gradients. Optimizers decide how to turn those gradients into actual parameter updates.

01 **Same gradients**  02 **Different update rules**  03 **Different trajectories**

Optimizer trajectories on one toy loss surface

SGD

loss start: `3.3000`

loss end: `0.0118`

delta: `-3.2882`

Momentum

loss start: `3.3000`

loss end: `0.2362`

delta: `-3.0638`

Adam

loss start: `3.3000`

loss end: `0.1374`

delta: `-3.1626`

Learning rate 0.110 Steps 18

All optimizers see the same gradients. Their update rules differ, so their paths differ.

Important takeaway

 Gradients tell the model where improvement may be. The optimizer decides how aggressively and in what style the model moves there. 

Same gradients, different path

 SGD, Momentum, and Adam can start from the same point and see the same gradients, yet follow different paths because each optimizer keeps different internal state. 

Chapter 08

## Attention: Q, K, and V {#chapter-08}

Attention lets tokens pull useful information from other tokens.

 Embeddings alone are too context-free. Take a word like "mole". It might mean a small animal, a mark on skin, a spy, or a unit in chemistry. The starting embedding is the same token representation, but the meaning depends on the surrounding words. 

 The model needs a way for tokens to talk to each other. That is what attention does. 

 For each token, the model creates three learned views: query, key, and value. The query represents what this token is looking for. The key represents what this token can be matched on. The value represents the information this token can contribute. 

 The model compares queries to keys to produce attention scores. Those scores are turned into weights, and the weights are used to mix the value vectors. So Q and K decide where information flows. V is the information that flows. 

How tokens exchange information

Right now we only have tokens. But sentences encode extra meaning through relationships between nearby words and references.

Select one token to inspect which key tokens it matches with (arrows), then how those weights mix into one updated value representation.

Context Scenario

A blue car crashed into a concrete wall, it was speeding.

Sentence Tokens

Pick any token to compute attention links and value mixing.

Important takeaway

 Attention is information routing. Query and key determine relevance; value carries the content that gets mixed in. 

Self-attention

 In self-attention, tokens attend to other tokens in the same sequence. In a decoder-only LLM, causal masking prevents a token from attending to future tokens during generation. 

Chapter 09

## Multi-Head Attention {#chapter-09}

One attention pattern is useful. Many attention patterns in parallel are much more powerful.

 A sentence contains many kinds of relationships at once. An adjective may modify a noun. A pronoun may refer to something earlier. A closing bracket may match an opening bracket. A verb may depend on the subject. 

 One attention head can learn one way of routing information. But one routing pattern is not enough. Multi-head attention runs several attention heads in parallel. Each head has its own learned projections, so each head can learn a different kind of relationship. 

 After the heads produce their outputs, those outputs are combined and projected back into the model dimension. This does not mean every head has a clean human-readable job. Attention weights are useful clues, not perfect explanations. 

 Modern models often use grouped-query attention. Groups of query heads share key/value heads, reducing memory use during inference, especially in the KV cache, while keeping much of the benefit of many query heads. 

Raw scores -> softmax weights -> value mixing

We also introduce **multi-head attention** here. In modern Transformer models each block doesn't just have a single attention head, but multiple. Different heads can learn different routing patterns, then their outputs are combined.

Each token creates three learned views of itself:

`Q` - the question this token asks.

`K` - what this token advertises about itself.

`V` - the information this token contributes.

For one selected query token, we compare its `Q` vector with every `K` vector.

Only after softmax do these scores become attention weights. Those weights decide how much of each V vector is mixed into this token’s next representation.

Tensor Shapes

We start with token vectors, project them into `Q`, `K`, and `V`, compute query-key compatibility scores, then convert those scores into attention weights and mix values.

`Q = XWq` -> `K = XWk` -> `V = XWv` -> `scores = QK^T / sqrt(d_k)` -> `weights = softmax(scores)` -> `output = weights·V`

This example uses unmasked self-attention, so every token can attend to every token. A GPT-style causal decoder would mask future tokens.

Which token is asking a question?

Selected token: `blue`

Its query asks: "Which other tokens help me understand `blue`?"

Emphasizes modifiers routing to the noun they describe (for example blue -> car).

Q View

`blue` embedding `[+0.200, +0.900, +0.400]`

↓ multiply by `Wq`

`Q_blue = [+0.310, +0.720, +0.650]`

K View

Each token embedding times `Wk` gives its advertised key vector.

`K_The`, `K_blue`, `K_car`, `K_hit`, `K_the`, `K_wall`

V View

Each token embedding times `Wv` gives value content to mix if attended.

`V_The`, `V_blue`, `V_car`, `V_hit`, `V_the`, `V_wall`

Raw Query-Key Scores (Not Attention Yet)

| Q \ K | The | blue | car | hit | the | wall |
|----|----|----|----|----|----|----|
| The | \+0.212 | \+0.191 | \+0.276 | \+0.366 | \+0.190 | \+0.297 |
| blue | \+0.307 | \+0.384 | \+2.270 | \+0.703 | \+0.284 | \+0.261 |
| car | \+0.425 | \+1.293 | \+1.122 | \+0.988 | \+0.393 | \+1.124 |
| hit | \+0.400 | \+0.503 | \+0.739 | \+0.846 | \+0.363 | \+0.798 |
| the | \+0.193 | \+0.184 | \+0.265 | \+0.339 | \+0.173 | \+0.283 |
| wall | \+0.415 | \+0.730 | \+1.047 | \+0.978 | \+0.382 | \+1.473 |

Step 1 · Selected Query Dot Keys

`blue` query · `The` key \= `+0.307`

`blue` query · `blue` key \= `+0.384`

`blue` query · `car` key \= `+2.270`

`blue` query · `hit` key \= `+0.703`

`blue` query · `the` key \= `+0.284`

`blue` query · `wall` key \= `+0.261`

Step 2 · Softmax To Attention Weights

`softmax([+0.307, +0.384, +2.270, +0.703, +0.284, +0.261])`

The  `7.9%`

blue  `8.6%`

car  `56.4%`

hit  `11.8%`

the  `7.7%`

wall  `7.6%`

Row sum: `7.9 + 8.6 + 56.4 + 11.8 + 7.7 + 7.6 = 100.0%`

Step 3 · Weighted Value Mix

Attention decides which value vectors get mixed into this token's next representation.

`output[1] = sum_i weights[1,i] * V[i]`

`head_output_blue = [+0.587, +0.970, +0.680]`

Highest attention target: `car` (56.4%).

Important takeaway

 Multi-head attention gives the model several ways to route information at the same time. Grouped-query attention is a practical modern variant that makes this cheaper during inference. 

Interpretation caveat

 Attention heads are not little thought modules. They are learned projections that may specialize, overlap, or behave in ways that are hard to summarize cleanly. 

Chapter 10

## RoPE {#chapter-10}

Attention needs to know order. RoPE gives position information directly to the attention mechanism.

 Attention compares tokens by content. But language also depends on order. "Dog bites man" and "man bites dog" contain the same words, but they do not mean the same thing. 

 Older transformer explanations often describe positional encodings as vectors added to token embeddings. That works, but many modern decoder-only models use something more integrated with attention: RoPE, or Rotary Positional Embeddings. 

 RoPE rotates parts of the query and key vectors based on their token positions. When attention compares a query with a key, the comparison should depend on both content and relative position. 

 Because RoPE modifies Q and K, it changes the attention scores. It does not directly rotate the value vectors, and it does not decide attention by itself. It changes which query/key pairs line up well. 

Relative position through rotation

**Problem.** Attention sees tokens, but it also needs word order. `dog bites man` and `man bites dog` contain the same words, but positions change meaning.

**Naive idea.** One option is to add a position vector to each token. RoPE does something different.

**RoPE idea.** RoPE makes attention position-aware by rotating `Q` and `K` vectors according to token position before their dot product is computed. It does not rotate `V`.

Word Order Matters

`dog bites man` **is not the same as** `man bites dog`

Same tokens, different positions. RoPE makes `Q·K` sensitive to that position change.

Same Token, Different Position

Example sentence: `The small dog chased the ball.`

In this visual, clicking a word temporarily treats that word as relative index `0`. RoPE is relative in this sense: if you look from a different token, the position offsets change, so the rotations you compare change too.

Click any token to make it the reference frame. That token stays unrotated while all other tokens rotate relative to it.

Relative offset insight

The selected token `The` is the anchor. Other tokens rotate by their position difference to this anchor. In the dot product, the important angle is `theta_m - theta_n`, so compatibility depends on relative offset `m - n`.

In this toy pair, `dot(before rotation) = +0.734` and `dot(after RoPE rotation) = -0.157`. As positions change, relative angle changes, and the query-key dot product changes too.

Multi-frequency pairs

Real vectors have many dimension pairs. Different pairs rotate at different speeds: fast pairs capture nearby offsets, while slow pairs preserve longer-range position patterns.

Connect back to attention

RoPE changes the score matrix before softmax. It does not directly decide attention by itself; it changes which `Q/K` pairs are compatible at different relative positions. RoPE gives attention a position-dependent bias, and the model still has to learn how to use it.

Important takeaway

 RoPE injects position into attention by rotating query and key vectors. It helps the model reason about relative position while computing attention. 

Compatibility, not payload

 RoPE affects compatibility, not payload. Q and K are rotated; V is not the main carrier of positional rotation here. 

Chapter 11

## Transformer Block {#chapter-11}

This is where the pieces become the repeated structure of the model.

 A transformer is built by stacking blocks. Each block takes in a sequence of token vectors and returns a sequence of token vectors with the same basic shape. The rows still correspond to token positions. The width is still the model dimension. 

 What changes is the information inside those vectors. 

 A modern decoder block usually normalizes the input, applies attention so tokens can exchange information, adds the result back through a residual connection, normalizes again, applies a feed-forward network, and adds that result back too. 

 The residual stream is the running representation that moves through the network. Attention mixes information between positions. The feed-forward network transforms each position. Normalization helps keep values stable. Residual connections preserve a path for information and gradients through many layers. 

 Layer by layer, the initially context-free embeddings become rich context-aware representations. 

One modern decoder block, end-to-end

This chapter combines what we learned into one full transformer block: normalization, multi-headed attention, residual paths, and a feed-forward network.

Let's look at an actual example of how all these elements are combined to build one Transformer block in a modern decoder-only model.

Click any block part to inspect its role, input/output dimensions, and jump back to the chapter where that part was introduced in detail.

Modern Decoder Block Dimensions

Reference style: `Modern Llama-style decoder block dimensions`

| Sequence length shown | 8 |
|----|----|
| Model width (`d_model`) | 4096 |
| Layers | 32 |
| Query heads | 32 |
| KV heads | 8 |
| Head width (`d_head`) | 128 |
| Q shape per token | 32 x 128 |
| K/V shape per token | 8 x 128 |
| Concat attention output | 4096 |
| FFN hidden width | 14336 |
| Norm | RMSNorm |
| Position encoding | RoPE |
| MLP | SwiGLU |
| Attention | causal \+ grouped-query attention |
| Block input/output shape | \[8 x 4096\] |

How This Scales In A Full Model

One block is rarely used alone. Decoder-only Transformers repeat this block many times before the final output projection over the vocabulary. In a Llama-8B-style setup, this is typically around `32` stacked blocks (layers).

Important takeaway

 A transformer block keeps the sequence shape mostly stable while repeatedly changing what each token vector represents. 

Modern decoder details

 In Llama-like models, you also see choices such as RMSNorm, RoPE, SwiGLU-style feed-forward layers, causal attention, and grouped-query attention. 

Chapter 12

## Training Phases {#chapter-12}

Training is not magic. It is many small prediction errors turned into parameter updates.

 From the outside, training often looks like one smooth curve going down. Reality is messier. 

 At the basic level, pretraining is simple to describe: show the model a lot of text and train it to predict the next token. It makes a prediction, measures the loss, computes gradients, and updates parameters. 

 Repeat that billions or trillions of times, and the model slowly becomes better at modeling text. But "loss goes down" is not the whole story. 

 Some patterns are learned early. Others appear much later. A model can improve on training data before it generalizes well. Sometimes better generalization arrives surprisingly late. 

 For large language models, training is also a scaling problem. Model size, dataset size, data quality, sequence length, optimizer settings, batch size, and compute budget all interact. 

How behavior changes across training

Training is often staged, not perfectly smooth: fast fitting first, slower consolidation, and sometimes delayed generalization.

This chart is an illustrative curve, not a claim about one exact production run.

Toy training curve (loss vs optimization steps)

Step marker 

Auto-detected phase summary

### Phase 1: Fit training data

**Train:** Training loss falls quickly.

**Validation:** Validation improves a bit, then slows.

Model memorizes useful local patterns first.

What is being learned in this phase

In large-scale pre-training, the model is mostly learning broad structure: world knowledge, language regularities, code patterns, and reasoning traces from text continuation.

This is why early improvements can look mostly statistical, while later improvements reflect better internal representations. The model is not yet being optimized for assistant behavior such as refusal style or helpful tone.

Where alignment and safety enter

Alignment behavior is primarily shaped after pre-training. Post-training adds objectives such as following instructions, refusing unsafe requests, formatting answers clearly, asking clarifying questions, and staying helpful.

So this chapter is mostly about capability learning dynamics; the next chapter focuses on behavior shaping.

Important takeaway

 Pretraining teaches broad capability through next-token prediction. The loss curve is a useful signal, but it is only one view of what the model is learning. 

Loss is not the whole story

 A lower loss generally means better prediction. It does not automatically mean better reasoning, better honesty, or better assistant behavior. 

Chapter 13

## Post-Training {#chapter-13}

Pretraining gives the model capability. Post-training shapes how that capability behaves.

 A pretrained language model has learned a huge amount about text. It can continue patterns, imitate styles, answer some questions, write code, and represent many facts and concepts. 

 But that does not automatically make it a good assistant. A base model is trained to predict likely next tokens. If you ask it a question, it might answer, but it might also continue the prompt, imitate a webpage, produce messy completions, or behave inconsistently. 

 Post-training teaches the model how we want it to respond. Instruction tuning shows the model examples of prompts and good task-oriented answers. Preference tuning compares possible answers and trains the model toward the ones people prefer: clearer, safer, more useful, better formatted, less rambling. 

 Different systems use different methods: supervised fine-tuning, RLHF, DPO, constitutional approaches, and many variations. The details differ, but the high-level goal is the same. 

From capability to assistant behavior

Pre-training creates broad capability; post-training shapes behavior. The same underlying model can respond very differently depending on which training stage it has gone through.

In practice, we can think of this as: pre-training learns *knowledge and patterns*, while post-training learns *assistant behavior*.

Capability vs Behavior

Pre-training

world knowledge, language, code, reasoning patterns

Post-training

follows instructions, refuses unsafe requests, formats answers, asks clarifying questions, uses a helpful tone

Three-stage pipeline

→ →

### 1. Base model (after pre-training)

**Objective:** Predict next token over large text/code corpora.

**Signal:** Web, books, code, and other broad unlabeled text.

Key message: pre-training gives broad latent capability, while instruction and preference tuning mostly steer behavior, format, and alignment.

Alignment and safety are not one switch; they are reinforced through multiple post-training signals, evaluations, and policy constraints.

Example prompt:

`Explain why the sky is blue.`

`Sunlight passes through the atmosphere and shorter blue wavelengths scatter more than longer wavelengths. This process is called Rayleigh scattering and makes the sky appear blue from most viewing angles.`

Not every model is trained with RLHF-style preference optimization. Some models stop at supervised instruction tuning, while others add direct preference objectives.

The goal is to make outputs more helpful, safer, and better aligned with human expectations when multiple answers are all technically plausible.

In short: pre-training teaches what the model *can* say, while preference tuning helps steer what it *should* say in assistant contexts.

How RLHF-Style Preference Tuning Works

Step 1 · Candidate answers

For one prompt, generate multiple candidate responses from the current model.

Step 2 · Pairwise ranking

Human raters (or policy-based systems) choose which answer is better in pairs. Example: `A > B` for helpfulness and safety.

Step 3 · Preference objective

Train a preference signal from those comparisons, then optimize the model so preferred responses become more likely.

Mini pairwise example

**Prompt:** `How can I recover a deleted file?`

**Answer A:** Gives clear, cautious, platform-specific recovery steps.

**Answer B:** Vague and omits safety checks.

**Ranking:** `A > B` (more useful and safer).

Important takeaway

 Pretraining mostly teaches what the model can do. Post-training strongly influences how, when, and in what style the model does it. 

Assistant behavior

 A post-trained assistant is not just a base model with more facts. It is a base model whose behavior has been shaped toward following instructions and user preferences. 

Chapter 14

## Context and KV Cache {#chapter-14}

Generating text one token at a time would be painfully wasteful without caching.

 Decoder-only language models generate text autoregressively: one token at a time. Each new token depends on the tokens before it. So after generating a token, the model appends it to the context and runs another step to predict the next one. 

 Naively, this would repeat a lot of work. If the prompt has already been processed, why recompute the same keys and values for all earlier tokens again and again? 

 The KV cache solves that. During attention, the model computes key and value vectors for each token. These are exactly the things future tokens need when they attend back to previous context. So the model stores them. 

 During generation, each new token only needs to compute its own new keys and values and attend to the cached previous ones. The cache saves compute, but it uses memory. The longer the context, the larger the KV cache becomes. 

 It helps to separate two phases: prefill processes the prompt and builds the initial cache; decode generates new tokens one by one while reusing the cache. 

Compute-memory tradeoff during inference

Decoding is autoregressive: each new token is generated after all previous tokens. KV cache changes the cost by reusing key/value tensors from earlier steps instead of recomputing them every time.

Decode setup

Prompt/context length 10,581 tokens Generated tokens 63 tokens

Autoregressive decode loop

Compute reduction from caching

**62.8x** less repeated attention work in this toy estimate

Without cache

At each step, recompute attention keys/values for the full seen sequence.

Relative compute: `668,619`

Memory behavior: lower KV storage, higher repeated compute.

With cache

Reuse stored K/V from previous tokens; compute only for the new token each step.

Relative compute: `10,644`

Estimated KV memory: `34.1 MB` for `10,644` seen tokens.

Without cache

668,619

With cache

10,644

KV memory

34.1 MB

These values are illustrative relative estimates. Exact memory and speed depend on architecture, precision, head counts, and runtime implementation.

Important takeaway

 The KV cache is not a summary of the conversation. It is stored attention data that avoids recomputing previous keys and values during generation. 

Speed vs memory

 KV cache speeds up repeated attention over previous tokens, but it increases memory use as the context grows. 

Chapter 15

## Quantization {#chapter-15}

Big models are often limited by memory. Quantization makes them smaller by storing numbers with fewer bits.

 Neural networks are mostly numbers. A large language model contains billions of weights, and during inference it also creates intermediate activations and KV-cache tensors. Storing all of that at high precision takes a lot of memory. 

 Quantization reduces that memory pressure by representing numbers with fewer bits. Instead of storing a weight as a 16-bit or 32-bit floating-point value, we may store an approximation using 8 bits, 4 bits, or another compact format. 

 The basic trade-off is simple: less precision -> less memory -> often faster or cheaper inference -> some approximation error. 

 But "4-bit" or "8-bit" is not the whole story. Different quantization methods make different choices. Some quantize only weights. Some also quantize activations. Some protect outlier channels. Some target the KV cache. 

 This is why two 4-bit models can behave differently. For local inference, quantization can be the difference between a model that does not fit in memory and a model that runs comfortably. 

Bit-width vs quality and memory

Quantization stores model weights with fewer bits. The goal is to reduce memory and make local inference more practical, while accepting a small quality trade-off.

Quantization selector

**FP32:** Maximum precision, largest memory footprint.

**Bits per value:** `32` bits

Stored directly as floating-point values.

Weight Matrix (FP32)

Quantized values at selected precision

| \+0.18371234 | -1.20491236 | \+0.00712091 | \+2.91823411 | -0.55291337 |
|----|----|----|----|----|
| \+0.44204588 | -0.99123817 | \+1.33100214 | -0.22345518 | \+0.07620133 |
| \+3.12019843 | -2.01444274 | \+0.55193302 | -0.04721129 | \+1.77231055 |
| -0.80911403 | \+2.20133044 | -1.48320182 | \+0.19441726 | -0.00990127 |
| \+0.61544281 | -0.33611945 | \+1.00993218 | -2.44211706 | \+0.43120572 |

**Unique values in this 5×5 matrix:** `25`

**Value range:** `-2.44211706` to `+3.12019843`

8B Model Size (Guestimate)

FP32 baseline

32.0 GB

FP32 estimate

32.0 GB

Saved

0.0 GB

Reduction

0%

Tradeoff: lower precision can slightly reduce accuracy or response quality, but it is often the key enabler for running strong models locally on consumer hardware.

Why numbers still look like floats in INT8/INT4: the model stores compact integers, then runtime kernels dequantize them back to approximate floating-point values during compute.

This chapter uses simplified estimates and symmetric quantization for intuition; real runtimes also include metadata, activation precision choices, and kernel-specific optimizations.

Important takeaway

 Quantization is controlled approximation. It reduces memory and often improves practical inference, but the quality depends on what is quantized and how. 

A family of trade-offs

 Quantization is not one technique. It is a family of compression and inference trade-offs. 

Closing

## Putting It All Together {#closing-heading}

 You have now followed the full path through a language model. 

 Text becomes tokens. Tokens become vectors. Attention moves information between positions. Feed-forward layers rewrite each token representation. Transformer blocks repeat that pattern many times. The final representation is projected into logits. Softmax and sampling turn those logits into the next token. 

 Then the new token is appended, and the process repeats. 

 Training teaches the model to build useful internal representations by predicting text. Post-training shapes those capabilities into assistant-like behavior. During inference, techniques like KV caching and quantization make the whole system practical enough to run at interactive speed. 

 This guide simplified many details on purpose. Real production LLMs include data pipelines, distributed training, specialized GPU kernels, safety systems, evaluation loops, alignment methods, serving infrastructure, and many engineering trade-offs. 

But the core path is now visible:

01 **Symbols** become vectors

02 **Vectors** exchange information

03 **Layers** rewrite representations

04 **Final state** becomes a next-token distribution

 The black box is still big, but it is no longer sealed. 

Final takeaway

 The model is no longer just "AI magic". It is a chain of transformations that can be traced and reasoned about. 

What we simplified

 Real models use huge datasets, distributed training, mixed precision, specialized kernels, safety systems, and many architecture-specific details. 

Where to go next

 Watch visual explanations, read illustrated transformer walkthroughs, implement a tiny transformer, experiment with tokenizers, and compare real model configs. 

References

- [3Blue1Brown — Neural Networks / Transformers](https://www.3blue1brown.com/topics/neural-networks): a visual, math-friendly series explaining neural networks, attention, and transformer internals.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): a classic visual explanation of the original Transformer architecture and attention flow.
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): a code-first path from tiny neural networks to building a GPT-style model from scratch.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/): a modern course on building language models end-to-end: data, tokenization, training, scaling, evaluation, and deployment.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original Transformer paper.

 © 2026 · Created by Roy van Rijn · Built with OpenAI Codex 
