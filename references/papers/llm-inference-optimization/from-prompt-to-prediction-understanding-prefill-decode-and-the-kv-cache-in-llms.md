title: From Prompt to Prediction: Understanding Prefill, Decode, and the KV Cache in LLMs - MachineLearningMastery.com From Prompt to Prediction: Understanding Prefill, Decode, and the KV Cache in LLMs - MachineLearningMastery.com
description: In the previous article, we saw how a language model converts logits into probabilities and samples the next token. But where do these logits come from? In this tutorial, we take a hands-on approach to understand the generation pipeline: How the prefill phase processes your entire prompt in a single parallel pass How the decode \[…\] In the previous article, we saw how a language model converts logits into probabilities and samples the next token. But where do these logits come from? In this tutorial, we take a hands-on approach to understand the generation pipeline: How the prefill phase processes your entire prompt in a single parallel pass How the decode phase generates tokens one at…
author: Yoyo Chan

# From Prompt to Prediction: Understanding Prefill, Decode, and the KV Cache in LLMs - MachineLearningMastery.com

In the [previous article](https://machinelearningmastery.com/how-llms-choose-their-words-a-practical-walk-through-of-logits-softmax-and-sampling/), we saw how a language model converts logits into probabilities and samples the next token. But where do these logits come from?

In this tutorial, we take a hands-on approach to understand the generation pipeline:

- How the prefill phase processes your entire prompt in a single parallel pass
- How the decode phase generates tokens one at a time using previously computed context
- How the KV cache eliminates redundant computation to make decoding efficient

By the end, you will understand the two-phase mechanics behind LLM inference and why the KV cache is essential for generating long responses at scale.

Let’s get started.

![](https://machinelearningmastery.com/wp-content/uploads/2026/03/neda-astani-KWTkd7mHqKE-unsplash-scaled.jpg){width=800 height=599}

From Prompt to Prediction: Understanding Prefill, Decode, and the KV Cache in LLMs\
Photo by [Neda Astani](https://unsplash.com/photos/green-grass-field-near-lake-under-white-clouds-and-blue-sky-during-daytime-KWTkd7mHqKE). Some rights reserved.

## Overview

This article is divided into three parts; they are:

- How Attention Works During Prefill
- The Decode Phase of LLM Inference
- KV Cache: How to Make Decode More Efficient

## How Attention Works During Prefill

Consider the prompt:

> Today’s weather is so …

As humans, we can infer the next token should be an *adjective*, because the last word “*so*” is a setup. We also know it probably describes weather, so words like “*nice*” or “*warm*” are more likely than something unrelated like “*delicious*“.

Transformers arrive at the same conclusion through attention. During prefill, the model processes the entire prompt in a single forward pass. Every token attends to itself and all tokens before it, building up a contextual representation that captures relationships across the full sequence.

The mechanism behind this is the scaled dot-product attention formula:

$$\
 \text{Attention}(Q, K, V) \= \mathrm{softmax}\left(\frac{QK\^\top}{\sqrt{d\_k}}\right)V\
 $$


We will walk through this concretely below.


To make the attention computation traceable, we assign each token a scalar value representing the information it carries:


| Position | Tokens | Values |
|----|----|----|
| 1 | Today | 10 |
| 2 | weather | 20 |
| 3 | is | 1 |
| 4 | so | 5 |


Words like “*is*” and “*so*” carry less semantic weight than “*Today*” or “*weather*“, and as we’ll see, attention naturally reflects this.


### Attention Heads


In real transformers, attention weights are continuous values learned during training through the $Q$ and $K$ dot product. The behavior of attention heads are learned and usually impossible to describe. No head is hardwired to “attend to even positions”. The four rules below are simplified illustration to make attention mechanism more intuitive, while the weighted aggregation over $V$ is the same.


Here are the rules in our toy example:


1. Attend to tokens at even number positions
2. Attend to the last token
3. Attend to the first token
4. Attend to every token


For simplicity in this example, the outputs from these heads are then **combined (averaged)**.


Let’s walk through the prefill process:


**Today**


1. Even tokens → none
2. Last token → Today → 10
3. First token → Today → 10
4. All tokens → Today → 10


**weather**


1. Even tokens → weather → 20
2. Last token → weather → 20
3. First token → Today → 10
4. All tokens → average(Today, weather) → 15


**is**


1. Even tokens → weather → 20
2. Last token → is → 1
3. First token → Today → 10
4. All tokens → average(Today, weather, is) → 10.33


**so**


1. Even tokens → average(weather, so) → 12.5
2. Last token → so → 5
3. First token → Today → 10
4. All tokens → average(Today, weather, is, so) → 9


### Parallelizing Attention


If the prompt contained 100,000 tokens, computing attention step-by-step would be extremely slow. Fortunately, attention can be expressed as tensor operations, allowing all positions to be computed in parallel.


This is the key idea of **prefill phase** in LLM inference: When you provide a prompt, there are multiple tokens in it and they can be processed in parallel. Such parallel processing helps speed up the response time for the first token generated.


To prevent tokens from seeing future tokens, we apply a causal mask, so they can only attend to itself and earlier tokens.


Producing a causal mask

1

2

3

4

5

6

7

8

9

10

11

12

import torch

tokens  \=  \[ "Today" ,  "weather" ,  "is" ,  "so" \]

n  \=  len ( tokens )

d\_k \= 64

V  \=  torch . tensor ( \[ \[ 10. \] ,  \[ 20. \] ,  \[ 1. \] ,  \[ 5. \] \] ,  dtype \= torch . float32 )

positions  \=  torch . arange ( 1 ,  n  \+  1 ) . float ( )  # 1-based: \[1, 2, 3, 4\]

idx  \=  torch . arange ( n )

causal\_mask  \=  idx . unsqueeze ( 1 )  >\=  idx . unsqueeze ( 0 )

print ( causal\_mask )

Output:


1234

tensor(\[\[ True, False, False, False\],

        \[ True, True, False, False\],

        \[ True, True, True, False\],

        \[ True, True, True, True\]\])

Now, we can start writing the “rules” for the 4 attention heads.


Rather than computing scores from learned $Q$ and $K$ vectors, we handcraft them directly to match our four attention rules. Each head produces a score matrix of shape `(n, n)`, with one score per query-key pair, which gets masked and passed through softmax to produce attention weights:


Causal attention to produce attention weights

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

def selector ( condition ,  size ) :

 "" "Return a (size, d\_k) tensor of \+1/-1 depending on condition." ""

val  \=  torch . where ( condition ,  torch . ones (

size ) ,  - torch . ones ( size ) )  # (size,)

 # (size, d\_k)

return  val . unsqueeze ( 1 ) . expand ( size ,  d\_k ) . contiguous ( )

\# Shared query: every row asks for a property, and K encodes which tokens match it.

Q  \=  torch . ones ( n ,  d\_k )

\# Head 1: select even positions# K says whether each token is at an even position.

K1  \=  selector ( positions  %  2  \=\=  0 ,  n )

scores1  \=  ( Q  @  K1 . T )  /  ( d\_k \* \*  0.5 )

\# Head 2: select the last token# K says whether each token is the last one.

K2  \=  selector ( positions  \=\=  n ,  n )

scores2  \=  ( Q  @  K2 . T )  /  ( d\_k \* \*  0.5 )

\# Head 3: select the first token# K says whether each token is the first one.

K3  \=  selector ( positions  \=\=  1 ,  n )

scores3  \=  ( Q  @  K3 . T )  /  ( d\_k \* \*  0.5 )

\# Head 4: select all visible tokens uniformly# K says all the tokens

K4  \=  selector ( positions  \=\=  positions ,  n )

scores4  \=  ( Q  @  K4 . T )  /  ( d\_k \* \*  0.5 )

\# Stack all head score matrices: shape (4, n, n)

scores  \=  torch . stack ( \[ scores1 ,  scores2 ,  scores3 ,  scores4 \] ,  dim \= 0 )

\# Apply causal mask so position i can only attend to positions \<\= i

scores  \=  scores . masked\_fill ( \~ causal\_mask . unsqueeze ( 0 ) ,  - 1e9 )

\# Convert logits to attention weights

weights  \=  torch . softmax ( scores ,  dim \= - 1 )

\# Optional safeguard for fully masked rows

all\_masked  \=  ( scores  \<\=  - 1e4 ) . all ( dim \= - 1 ,  keepdim \= True )

weights  \=  torch . where ( all\_masked ,  torch . zeros\_like ( weights ) ,  weights )

\# Compute contexts: (heads, n, n) @ (n, 1) -> (heads, n, 1)

contexts  \=  ( weights  @  V ) . squeeze ( - 1 )

print ( "Contexts by attention head (rows) x token position (columns):\n" ,  contexts )

context4  \=  contexts \[ : ,  - 1 \]

print ( "\nContext for final prompt position:\n" ,  context4 )

Output:


1

2

3

4

5

6

7

8

Contexts by attention heads (rows) x token position (columns):

 tensor(\[\[10.0000, 20.0000, 20.0000, 12.5000\],

        \[10.0000, 15.0000, 10.3333,  5.0000\],

        \[10.0000, 10.0000, 10.0000, 10.0000\],

        \[10.0000, 15.0000, 10.3333,  9.0000\]\])

Context for final prompt position:

 tensor(\[12.5000,  5.0000, 10.0000,  9.0000\])

The result of this step is called a context vector, which represents a weighted summary of all previous tokens.


### From contexts to logits


Each attention head has learned to pick up on different patterns in the input. Together, the four context values `[12.5, 5.0, 10.0, 9.0]` form a summary of what “*Today’s weather is so…*” represents. It will then project to a matrix, which each column encodes how strong a given vocabulary is associated with each attention head’s signal, to give logit score per word.


Compute logits as projection from context vector

12

. . .

logits  \=  context  @  W\_vocab

For our example, let’s say we have “*nice*”, “*warm*”, and “*delicious*” in the vocab:


Example projection matrix

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

. . .

vocab  \=  \[ "nice" ,  "warm" ,  "delicious" \]

\# Each column corresponds to a vocab word# Each row corresponds to one attention head feature

W\_vocab  \=  torch . tensor ( \[

\[ 0.8 ,  0.6 ,  0.1 \] ,  # head 1 weights → nice, warm, delicious

\[ 0.5 ,  0.4 ,  0.2 \] ,  # head 2 weights

\[ 0.1 ,  0.2 ,  0.5 \] ,  # head 3 weights

\[ 0.2 ,  0.3 ,  0.1 \] ,  # head 4 weights

\] ) # shape: (4, 3)

logits  \=  context4  @  W \_vocab # (4,) @ (4, 3) → (3,)

for  word ,  logit in  zip ( vocab ,  logits ) :

 print ( f "{word:10s} {logit.item():.3f}" )\` \` \`

So the logits for “*nice*” and “*warm*” are much higher than “*delicious*”.


123

nice       15.300warm       14.200delicious  8.150


## The Decode Phase of LLM Inference


Now suppose the model generates the next token: “*nice*“. The task is now to generate the next token with the extended prompt:


> Today’s weather is so nice …


The first four words in the extended prompt are the same as the original prompt. And now we have the fifth word in the prompt.


During **decode**, we do not recompute attention for all previous tokens as the result would be the same. Instead, we compute attention only for the new token to save time and compute resources. This produces a single new attention row.


Decode phase appends new row each time a new token is generated

1

2

3

4

5

6

7

8

9

10

new\_token \= "nice"

tokens  \=  tokens  \+  \[ new\_token \]

new\_value  \=  torch . tensor ( \[ \[ 7.0 \] \] )  # value of "nice" is 7

V  \=  torch . cat ( \[ V ,  new\_value \] ,  dim \= 0 )

n  \=  len ( tokens )

idx  \=  torch . arange ( n )

pos  \=  torch . arange ( 1 ,  n  \+  1 ) . float ( )  # \[1, 2, 3, 4, 5\]

print ( "New tokens: " ,  tokens )

print ( "New Values: " ,  V )

Output:


1

2

3

4

5

6

New tokens:  \['Today', 'weather', 'is', 'so', 'nice'\]

New Values:  tensor(\[\[10.\],

        \[20.\],

        \[ 1.\],

        \[ 5.\],

        \[ 7.\]\])

Now, we apply the 4 attention heads and compute the new context vector:


Attention during decode: Full query and only the new row in key

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

\# Rebuild all K matrices for the next token (n\=5)# We will introduce KV-cache later

K1\_new  \=  selector ( pos  %  2  \=\=  0 ,  n )  # even positions → \+1

K2\_new  \=  selector ( pos  \=\=  n ,  n )  # last token     → \+1

K3\_new  \=  selector ( pos  \=\=  1 ,  n )  # first token    → \+1

K4\_new  \=  selector ( pos  \=\=  pos ,  n )  # all tokens     → \+1

\# During decode, only compute Q for the NEW token (one row)

Q\_new  \=  torch . ones ( 1 ,  d\_k )

scores1\_new  \=  ( Q \_new @  K1\_new . T )  /  ( d\_k \* \*  0.5 )  # (1, 5)

scores2\_new  \=  ( Q \_new @  K2\_new . T )  /  ( d\_k \* \*  0.5 )  # (1, 5)

scores3\_new  \=  ( Q \_new @  K3\_new . T )  /  ( d\_k \* \*  0.5 )  # (1, 5)

scores4\_new  \=  ( Q \_new @  K4\_new . T )  /  ( d\_k \* \*  0.5 )  # (1, 5)

\# Stack: shape (4, 1, 5)

new\_scores  \=  torch . stack (

\[ scores1\_new ,  scores2\_new ,  scores3\_new ,  scores4\_new \] ,  dim \= 0 )

\# No causal mask needed — new token can see all previous tokens by definition

new\_weights  \=  torch . softmax ( new\_scores ,  dim \= - 1 )  # (4, 1, 5)

context5  \=  ( new \_weights @  V ) . squeeze ( )  # (4,)

print ( "Visible tokens:" ,  tokens )

print ( "Context for new token position:\n" ,  context5 )

Output:


123

Visible tokens: \['Today', 'weather', 'is', 'so', 'nice'\]

Context for new token position:

 tensor(\[12.5000, 7.0000, 10.0000, 8.6000\])

However, unlike prefill where the entire prompt is processed in parallel, decoding must generate tokens one at a time (autoregressively) because the future tokens have not yet been generated. Without caching, every decode step would recompute **keys** and **values** for all previous tokens from scratch, making the total work across all decode steps $O(n\^2)$ in sequence length. KV cache reduces this to $O(n)$ by computing each token’s $K$ and $V$ exactly once.


## KV Cache: How to Make Decode More Efficient


To make the autoregressive docoding efficient, we can store the keys ($K$) and values ($V$) for every token separately for each attention head. In this simplified example we would use only one cache. Then, during decoding, when a new token is generated, the model does not recompute keys and values for all previous tokens. It computes the query for the new token, and attends to the cached keys and values from previous tokens.


If we look at the previous code again, we can see that there is no need to recompute $K$ for the entire tensor:


1

K1\_new  \=  selector ( pos  %  2  \=\=  0 ,  n )  # even positions → \+1

Instead, we can simply compute K for the new position, and attach it to the K matrix we have already computed and saved in cache:


12

K1\_new  \=  selector ( new\_pos  %  2  \=\=  0 ,  1 )  # is pos 5 even? → -1

K1\_cache  \=  torch . cat ( \[ K1 ,  K1\_new \] ,  dim \= 0 )  # (4→5, d\_k)

Here’s the full code for **decode** phase using KV cache:


Algorithm for decode phase

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

\# In decode we only compute the query for the NEW token (position 5).

new\_pos  \=  pos \[ - 1 : \]  # tensor(\[5.\])

\# Compute ONLY the new token's key for each head

K1\_new  \=  selector ( new\_pos  %  2  \=\=  0 ,  1 )  # is pos 5 even?  → -1

K2\_new  \=  selector ( new\_pos  \=\=  n ,  1 )  # is pos 5 last?  → \+1

K3\_new  \=  selector ( new\_pos  \=\=  1 ,  1 )  # is pos 5 first? → -1

K4\_new  \=  selector ( new\_pos  \=\=  new\_pos ,  1 )  # always          → \+1

\# Append new key to the cached prefill keys

K1\_cache  \=  torch . cat ( \[ K1 ,  K1\_new \] ,  dim \= 0 )  # (4→5, d\_k)

K2 \[ - 1 \]  \=  - torch . ones ( d\_k )  # position 4 is no longer last

K2\_cache  \=  torch . cat ( \[ K2 ,  K2\_new \] ,  dim \= 0 )

K3\_cache  \=  torch . cat ( \[ K3 ,  K3\_new \] ,  dim \= 0 )

K4\_cache  \=  torch . cat ( \[ K4 ,  K4\_new \] ,  dim \= 0 )

\# Q is only for the new token

Q\_dec  \=  torch . ones ( 1 ,  d\_k )

scores1\_dec  \=  ( Q \_dec @  K1\_cache . T )  /  ( d\_k \* \*  0.5 )

scores2\_dec  \=  ( Q \_dec @  K2\_cache . T )  /  ( d\_k \* \*  0.5 )

scores3\_dec  \=  ( Q \_dec @  K3\_cache . T )  /  ( d\_k \* \*  0.5 )

scores4\_dec  \=  ( Q \_dec @  K4\_cache . T )  /  ( d\_k \* \*  0.5 )

\# Stack → (4 heads × 1 query × n keys)

scores\_dec  \=  torch . stack ( \[ scores1\_dec ,  scores2\_dec ,  scores3\_dec ,  scores4\_dec \] ,  dim \= 0 )

\# Softmax over key dimension

weights\_dec  \=  torch . softmax ( scores\_dec ,  dim \= - 1 )

\# Edge case: all-masked rows → zero context (same guard as prefill)

all\_masked\_dec  \=  ( scores\_dec  \<\=  - 1e4 ) . all ( dim \= - 1 ,  keepdim \= True )

weights\_dec  \=  torch . where ( all\_masked\_dec ,  torch . zeros\_like ( weights\_dec ) ,  weights\_dec )

\# Context vectors: (4 × 1 × n) @ (n × 1) → (4 × 1 × 1) → squeeze → (4,)

contexts\_dec  \=  ( weights \_dec @  V ) . squeeze ( - 1 ) . squeeze ( - 1 )

print ( "\nDecode context for 'nice' (one value per head):\n" ,  contexts\_dec )

Output:


12

Decode context for 'nice' (one value per head): tensor(\[12.5000, 6.0000, 10.0000, 8.6000\])

Notice this is identical to the result we computed without the cache. KV cache doesn’t change what the model computes, but it eliminates redundant computations.


KV cache is different from the **cache** in other application that the object stored is not replaced but updated. Every new token added to the prompt appends a new row to the tensor stored. Implementing a KV cache that can efficiently update the tensor is the key to make LLM inference faster.


## Further Readings


Below are some resources that you may find useful:


- Sasha Rush and Gail Weiss, [Thinking Like Transformers](https://srush.github.io/raspy/)
- [Caching](https://huggingface.co/docs/transformers/en/cache_explanation). Hugging Face transformers documentation.
- Benjamin Merkel, [Prefill and Decode for Concurrent Requests – Optimizing LLM Performance](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests), 2025
- [Cache strategies](https://huggingface.co/docs/transformers/en/kv_cache). Hugging Face transformers documentation.


## Summary


In this article, we walked through the two phases of LLM inference. During prefill, the full prompt is processed in one parallel forward pass and the keys and values for every token are computed and stored. During decode, the model generates one token at a time, using only the new token’s query against the cached keys and values to avoid redundant recomputation. Prefill warms up the KV cache and decode updates it. Faster prefill means sooner you see the first token in the response and faster decode means faster you see the rest of the response. Together, these two phases explain why LLMs can process long prompts quickly but generate output token by token, and why KV cache is essential for making that generation practical at scale.



### More On This Topic

- [![Build-an-Inference-Cache-to-Save-Cost-in-High-Traffic-LLM-Apps](https://machinelearningmastery.com/wp-content/uploads/2025/10/Build-an-Inference-Cache-to-Save-Cost-in-High-Traffic-LLM-Apps-200x200.png "Build an Inference Cache to Save Costs in High-Traffic LLM Apps"){width=200 height=200} Build an Inference Cache to Save Costs in…](https://machinelearningmastery.com/build-an-inference-cache-to-save-costs-in-high-traffic-llm-apps/)
- [![mlm-understanding-rag-9](https://machinelearningmastery.com/wp-content/uploads/2025/03/mlm-understanding-rag-9-200x200.png "Understanding RAG Part IX: Fine-Tuning LLMs for RAG"){width=200 height=200} Understanding RAG Part IX: Fine-Tuning LLMs for RAG](https://machinelearningmastery.com/understanding-rag-part-ix-fine-tuning-llms-for-rag/)
- [![roman-kraft-g_gwdpsCVAY-unsplash](https://machinelearningmastery.com/wp-content/uploads/2025/12/roman-kraft-g_gwdpsCVAY-unsplash-200x200.jpg "Creating a Llama or GPT Model for Next-Token Prediction"){width=200 height=200} Creating a Llama or GPT Model for Next-Token Prediction](https://machinelearningmastery.com/creating-a-llama-or-gpt-model-for-next-token-prediction/)
- [![mlm-gulati-pycaret-time-series-01](https://machinelearningmastery.com/wp-content/uploads/2025/02/mlm-gulati-pycaret-time-series-01-200x200.png "Time Series Forecasting with PyCaret: Building Multi-Step Prediction Model"){width=200 height=200} Time Series Forecasting with PyCaret: Building…](https://machinelearningmastery.com/time-series-forecasting-with-pycaret-building-multi-step-prediction-model/)
- [![kote-puerto-so5nsYDOdxw-unsplash](https://machinelearningmastery.com/wp-content/uploads/2024/07/kote-puerto-so5nsYDOdxw-unsplash-200x200.jpg "CatBoost Essentials: Building Robust Home Price Prediction Systems"){width=200 height=200} CatBoost Essentials: Building Robust Home Price…](https://machinelearningmastery.com/catboost-essentials-building-robust-home-price-prediction-systems/)
- [![carry-kung-5W5Fb6bHOQc-unsplash](https://machinelearningmastery.com/wp-content/uploads/2023/02/carry-kung-5W5Fb6bHOQc-unsplash-150x150.jpg "LSTM for Time Series Prediction in PyTorch"){width=150 height=150} LSTM for Time Series Prediction in PyTorch](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)
