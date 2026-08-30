title: Prefill vs Decode: LLM Inference Optimization
description: In this blog, we will learn about Prefill vs Decode, the two phases of LLM inference, and how understanding them helps us optimize the speed of an LLM. We will also see how the prefill and decode phases work, how the KV cache connects them, how they differ and when to use which one based on our use case, and how we optimize each phase to make an LLM faster.

# Prefill vs Decode: LLM Inference Optimization

![Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/_next/image?url=%2Fstatic%2Fimages%2Fblog%2Fprefill-vs-decode-llm-inference-optimization.png&w=3840&q=75)

In this blog, we will learn about Prefill vs Decode, the two phases of LLM inference, and how understanding them helps us optimize the speed of an LLM. We will also see how the prefill and decode phases work, how the KV cache connects them, how they differ and when to use which one based on our use case, and how we optimize each phase to make an LLM faster.

We will cover the following:

- What is LLM inference
- The two phases: Prefill and Decode
- Prefill explained in simple words
- Decode explained in simple words
- A diagram of the two phases and the KV cache flow
- The KV cache as the bridge between the two phases
- A step-by-step walkthrough of a few decode steps
- Prefill vs Decode comparison table
- Why this split matters: compute-bound vs memory-bound
- The key metrics: TTFT, TPOT, throughput, and end-to-end latency
- Optimization techniques mapped to each phase
- Conclusion

I am **Amit Shekhar**, Founder @ [Outcome School](https://outcomeschool.com), I have taught and mentored many developers, and their efforts landed them high-paying tech jobs, helped many tech companies in solving their unique problems, and created many open-source libraries being used by top companies. I am passionate about sharing knowledge through open-source, blogs, and videos.

I teach [AI and Machine Learning](https://outcomeschool.com/program/ai-and-machine-learning) at Outcome School.

Let's get started.

## What is LLM inference {#what-is-llm-inference}

Before we learn about prefill and decode, we must understand what LLM inference means.

An **LLM** is a Large Language Model. It is the model that powers chat assistants like the ones we use every day.

**Inference** is the act of using a trained model to produce an answer. In simple words, inference is the model actually doing its job, which means reading your question and writing back a reply.

Here is the most important thing to understand. An LLM does not write out the whole answer at once. It writes the answer one small piece at a time. Each small piece is called a **token**. A token is a small chunk of text, like a word or a part of a word.

So, the model generates one token, then the next token, then the next, until the full answer is complete. This is why, when you use a chat assistant, you can see the words appear one after another like someone typing. This live, token-by-token appearance is called [**streaming**](https://outcomeschool.com/blog/how-does-token-streaming-work).

To get a feel for scale, a short sentence is roughly 15 to 20 tokens. So, a 1,000 token prompt is about a page of text, and a 200 token answer is a few sentences.

A **request** is simply one user's question sent to the model. When we say the server handles many requests, we mean it answers many users' questions.

Now, we have understood what LLM inference is. It is the model generating text token by token.

## The two phases: Prefill and Decode {#the-two-phases-prefill-and-decode}

Every time an LLM answers a request, it goes through two distinct phases.

- **Prefill** is the first phase. It reads and processes the whole input prompt that you sent.
- **Decode** is the second phase. It generates the output tokens one at a time.

In simple words, prefill is the model reading your question, and decode is the model writing the answer.

Both phases use the exact same model. They do not use different models or different weights. The **weights** are the millions of numbers the model learned during training. They are what the model "knows", and there are so many of them that they take up a large part of the GPU's memory. We will use this fact soon. Only the way the two phases work is different, and that small difference changes everything, as we will see.

Let's understand each phase deeply.

## Prefill explained in simple words {#prefill-explained-in-simple-words}

**Prefill is the phase where the model reads and processes your entire input prompt in one single pass and produces the very first output token.**

The best way to learn this is by taking an example.

Suppose you paste a question that is 1,000 tokens long. In the prefill phase, the model does not read these 1,000 tokens one by one. It reads all 1,000 tokens together, at the same time.

Let's say a student walks into an exam hall. Before writing anything, the student silently reads the entire question paper. The student scans every question at once and jots down quick notes about what matters. This reading-the-whole-paper part is prefill.

Because the whole prompt is already known up front, there is no need to wait. The model can process every input token at the same time. This is called **parallel processing**. Means, many things happen together instead of one after another.

While reading the prompt, the model also prepares some notes for later. For every input token, it computes and stores two special pieces of data, called the **Key (K)** and the **Value (V)**. All of these stored Keys and Values together form the [**KV cache**](https://outcomeschool.com/blog/kv-cache-in-llms). Do not worry about what exactly Key and Value mean right now. For now, just understand that the KV cache is the model's memory of your prompt. We will learn about it in detail soon.

So, prefill fills up this KV cache with the model's notes about every input token. And at the end of prefill, the model produces the very first output token.

Here is a very important point. **Prefill produces only the first output token, not the whole answer.** All the remaining tokens come later, in the decode phase.

**Prefill is compute-bound.** The compute-bound means the speed is held back by the math, nothing else. A [**GPU**](https://outcomeschool.com/blog/how-does-a-gpu-work-for-deep-learning) is the powerful chip that runs the model. Inside the GPU there are **math units**, which are the parts that actually do the multiplication. During prefill, these math units are kept very busy doing heavy math. On a high-end GPU, the math units can stay highly used during prefill, often approaching 90 percent or more on large prompts. The GPU is well used.

Why is prefill so heavy on math? Because when many tokens are processed together, the work becomes very large multiplication of big tables of numbers. A big table of numbers like this is called a matrix, and multiplying two big tables together is exactly what GPUs are built for. So, the GPU runs near its full power.

There is one more thing to know. Longer prompts mean longer prefill. A short question feels almost instant. But if we paste a huge document in front of the model, prefill takes longer, and we wait longer before the first word appears.

In fact, part of the prefill work grows even faster than the prompt length. Inside the model there is a step, called [**attention**](https://outcomeschool.com/blog/self-attention-in-transformers), where every token must look at every other token to understand the context. So, doubling the prompt roughly quadruples this part of the work. For a concrete number, doubling the prompt from 100 to 200 tokens makes this part of the work about 4 times bigger, not 2 times. Means, very long prompts can feel slow before the first word appears. Here is a simple picture. In a room of 10 people, everyone shaking hands with everyone is manageable. But in a room of 100 people, the number of handshakes explodes. The bigger the room, which means the longer the prompt, the much faster the work piles up.

This wait before the first word is so important that it has its own name, which we will learn in the metrics section. For now, remember this simple line.

> Faster prefill means sooner you see the first token.

Now, we have understood prefill. Let's move to decode.

## Decode explained in simple words {#decode-explained-in-simple-words}

**Decode is the phase where the model generates the output tokens one at a time, reusing the KV cache that prefill prepared.**

Let's continue our exam example. Reading the question paper was prefill. Now the student starts writing the answer, word by word. The student writes one word, thinks about it, then writes the next word, and so on. This slow, word-by-word writing is decode.

In decode, the model generates only one new token per step. Then it feeds that token back into itself to help produce the next token. Because each new token depends on the token just produced, the tokens must come strictly one after another. This is called [**autoregressive**](https://outcomeschool.com/blog/autoregressive-models) generation. In simple words, the model keeps feeding its own output back as new input.

So, decode is sequential. It cannot process all the output tokens at once the way prefill processes all the input tokens at once. Each token must wait for the one before it.

Now, here is the clever part. At each decode step, the model does not re-read and re-process your whole prompt again. That would be very slow and wasteful. Instead, it reads the KV cache, which is the memory that prefill already prepared. It computes the Key and Value for just the one new token, reads all the older Keys and Values from the cache, produces the next token, and then appends the new token's Key and Value to the cache.

So, at each decode step, the cache grows by exactly one entry.

**Decode is memory-bandwidth-bound.** This is the opposite of prefill. The speed of decode is held back by how fast data can be moved from the GPU's memory into its math units, not by the math itself.

Let's understand why, one small step at a time.

In decode, only one token is processed per step. So, instead of multiplying two big tables together like prefill does, decode multiplies the big table by a single thin row of numbers, which is the one new token. A single thin row like this is called a vector. So, the math for one token is tiny.

But to produce even that one token, the GPU must still read the entire model's weights and the whole growing KV cache from its memory. Remember, the weights are huge, so this is a lot of data to move. **Memory bandwidth** is the speed at which data can be moved out of the GPU's memory. Here is the key fact to hold on to. Inside a GPU, doing the math is very fast, but fetching a large amount of data from its memory is comparatively slow.

So, the GPU spends most of its time waiting for data to arrive, and its math units sit mostly idle. During decode, the math units of a high-end GPU can sit only around 20 to 40 percent busy for a single request.

Here is a simple analogy. Think of a chef who must walk into a giant pantry, grab one ingredient, walk back, chop it, and repeat, one ingredient per trip. The walking back and forth (moving data from memory) takes far longer than the chopping (the tiny math). The knives (the GPU's math units) sit unused most of the time.

So, prefill keeps the GPU busy with math, but decode keeps the GPU waiting on memory.

Here is one more interesting point. For a typical request, prefill often finishes in a fraction of a second, but decode can run for several seconds, because it produces many tokens one after another. So, decode usually takes most of the total time, even though it uses only a small part of the GPU's math power. This is exactly why so many optimizations focus on the decode phase.

Remember this simple line.

> Faster decode means faster you see the rest of the answer.

Now, we have understood both prefill and decode. Let's visualize them together.

## A diagram of the two phases and the KV cache flow {#a-diagram-of-the-two-phases-and-the-kv-cache-flow}

Let's see a clean diagram of how a request flows through prefill and then decode.

```
INPUT PROMPT (for example, 1,000 tokens)
        |
        v
+-----------------------------+
|   PREFILL  (one big step)   |
|  - reads ALL input tokens   |
|    in parallel              |
|  - writes the KV cache      |
|  - GPU compute very busy    |
+-----------------------------+
        |
        | produces the FIRST output token
        v
+-----------------------------+        +------------------+
|   DECODE  (many small steps)| <----> |     KV CACHE     |
|  - one token per step       |  read  |  (the memory of  |
|  - reads the KV cache       |  and   |   all tokens so  |
|  - appends one new entry    |  grow  |   far)           |
|  - GPU mostly waits on      |        +------------------+
|    memory                   |
+-----------------------------+
        |
        | token 2, token 3, token 4, ... token N
        v
   FULL ANSWER
```

Here, we can see the timeline clearly. There is exactly one prefill pass, and then there are many decode steps. If your answer is 200 tokens long, that is 1 prefill pass plus 200 sequential decode steps. The KV cache is written once by prefill, and then it is read and extended by every single decode step.

This is how the two phases work together.

## The KV cache as the bridge between the two phases {#the-kv-cache-as-the-bridge-between-the-two-phases}

The KV cache is the bridge that connects prefill and decode. So, we must understand it well.

Let's understand the problem it solves first.

Inside the model, to produce the next token, the model must look back at every previous token. For every token, the model computes three things: a [**Query (Q)**, a **Key (K)**, and a **Value (V)**](https://outcomeschool.com/blog/math-behind-attention-qkv). The new token compares its Query against the Keys of all previous tokens, and then it mixes their Values to decide what to say next.

Here is the key observation. A past token's Key and Value never change once they are computed. They do not depend on tokens that come after. So, instead of computing them again and again at every step, we can compute them once and store them. This store of Keys and Values is the KV cache.

**The KV cache stores only the Keys and the Values of past tokens. It does not store the Queries.** The Query is freshly computed for the current token at every step, so there is no need to store it.

Now, let's see why this matters so much with a small numeric walkthrough. For the sake of understanding, think of one entry in the cache as one token's stored Key and Value.

Suppose prefill has already processed a 3-token prompt. So, the KV cache already has 3 entries. Now decode begins.

**Step 1:** The model takes the first output token's input, computes its Key and Value, and reads the 3 cached entries to look back. It produces token 4. Then it appends token 4's Key and Value to the cache. The cache now has 4 entries.

**Step 2:** The model takes token 4, computes its Key and Value, and reads the 4 cached entries to look back. It produces token 5. Then it appends token 5's Key and Value to the cache. The cache now has 5 entries.

**Step 3:** The model takes token 5, computes its Key and Value, and reads the 5 cached entries to look back. It produces token 6. Then it appends token 6's Key and Value to the cache. The cache now has 6 entries.

Let's visualize how the cache grows across these steps.

```
After prefill   |P1|P2|P3|           ->  cache has 3 entries
Decode step 1   |P1|P2|P3|T4|        ->  append T4, now 4 entries
Decode step 2   |P1|P2|P3|T4|T5|     ->  append T5, now 5 entries
Decode step 3   |P1|P2|P3|T4|T5|T6|  ->  append T6, now 6 entries
```

Here, the boxes P1, P2, and P3 are the prompt tokens written by prefill, and T4, T5, and T6 are appended by decode, one box per step.

Here, we can notice the pattern. At every step, the model computes the Key and Value for only one new token. It reads all the older Keys and Values from the cache instead of computing them again. And the cache grows by exactly one entry per step.

**Note:** Without the KV cache, every decode step would recompute the Keys and Values of all previous tokens from scratch. Since step N redoes the work for all N earlier tokens, the total work summed over the whole answer grows roughly with the square of the length, which is very slow. The KV cache reduces this so that each token's Key and Value is computed exactly once. This is why the KV cache is not just a small optimization. Without it, decode would be much slower.

But there is a catch. The KV cache keeps growing. It starts at the size of the prompt after prefill, and it grows by one entry for every generated token. It also grows with the number of users being served together, because each user needs their own cache. In extreme cases, such as very long contexts combined with many concurrent users, the total KV cache can even exceed the size of the model weights and fill up the GPU's memory. This growing cache is the reason many of the optimizations we will see later exist.

This is how the KV cache works as the bridge between prefill and decode.

To master the KV Cache, the Q, K, and V matrices, and build a Large Language Model (LLM) from scratch, check out our [AI and Machine Learning Program](https://outcomeschool.com/program/ai-and-machine-learning) at Outcome School.

## Prefill vs Decode comparison table {#prefill-vs-decode-comparison-table}

Let me tabulate the differences between Prefill and Decode for your better understanding.

| Aspect | Prefill | Decode |
|----|----|----|
| What it does | Processes the whole input prompt | Generates output tokens one by one |
| Parallelism | Parallel (all prompt tokens at once) | Sequential (one token at a time) |
| Steps | One big step | Many small steps (N of them) |
| Math shape | Big table times big table | Big table times a thin vector |
| Bottleneck | Compute-bound (limited by math speed) | Memory-bandwidth-bound (limited by data movement) |
| GPU utilization | High (math units busy, can approach 90 percent or more) | Low (math units mostly idle, around 20 to 40 percent) |
| KV cache action | Writes the cache | Reads and extends the cache |
| Latency metric | Time To First Token (TTFT) | Time Per Output Token (TPOT) |
| Cost driver | Grows with prompt length | Streams full weights plus KV cache every step |

This table sums up the whole story. Prefill and decode are mirror images of each other.

**A quick note for you**

No matter which tech domain you work in, get familiar with these topics:

- LLM
- RAG
- MCP
- Agent
- Fine-tuning
- Quantization

We put it all together in one video:

[AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, and Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)

No need to stop reading - bookmark it and watch later when you get time. Future you will thank you.

Now, let's get back to the topic.

## Why this split matters: compute-bound vs memory-bound {#why-this-split-matters-compute-bound-vs-memory-bound}

Now, the question is: why do we care so much about this split into two phases?

The answer is that the two phases have opposite bottlenecks, and almost every optimization comes from this single fact. A **bottleneck** is the one slow part that holds back the whole process, like the narrow neck of a bottle that limits how fast water pours out.

**Prefill is compute-bound.** It does heavy math on many tokens at once, so it keeps the GPU's math units very busy. The work becomes large table-times-table multiplication, which GPUs love.

**Decode is memory-bandwidth-bound.** It does tiny math on one token, but it must stream the whole model plus the growing KV cache from memory at every step. So, the GPU spends most of its time waiting on memory, and the math units sit idle.

Here is a simple way to remember the difference. Prefill is like a chef chopping a whole pile of vegetables at once, where the hands are fully busy. Decode is like plating one finished dish at a time while walking to the fridge for every single dish, where the trips to the fridge are the slow part, not the plating.

Let's put the two phases side by side.

```
PREFILL (compute-bound)           DECODE (memory-bound)
many tokens processed at once     one token processed at a time

GPU math units: [#########-]      GPU math units: [##--------]
  about 90% busy                    about 20 to 40% busy

held back by: doing the MATH      held back by: MOVING DATA
                                  (model weights + KV cache)
```

Here, we can see the mirror image clearly. Prefill keeps the math units almost full, while decode leaves them mostly idle and waits on memory instead.

This single difference explains two big things.

First, it explains why long prompts make us wait longer before the answer starts. Long prompts mean more prefill math.

Second, it explains why we can serve many users at the same time during decode without much extra cost. Since one decode step barely uses the GPU's math, we can run many users' decode steps together. We will see this idea, called batching, in the optimization section.

Because the two phases want different things from the hardware, some serving systems even run prefill and decode on separate GPUs. We will cover that too.

## The key metrics: TTFT, TPOT, throughput, and end-to-end latency {#the-key-metrics-ttft-tpot-throughput-and-end-to-end-latency}

To optimize anything, we must first measure it. There are a few key metrics, and each one maps cleanly to a phase. So, this section ties everything together.

**TTFT (Time To First Token)** is the time from sending your request until the very first output token appears. It is set mainly by prefill, because prefill must finish before the first token can be produced. Roughly, TTFT is the prefill time plus one decode step. So, longer prompts mean larger TTFT. In simple words, TTFT is how fast the answer starts appearing.

**TPOT (Time Per Output Token)** is the average time to produce each token after the first one. It is also called **ITL (Inter-Token Latency)**, which means the time gap between two tokens. TPOT is set by decode, because decode produces these tokens one at a time. In simple words, TPOT is the typing speed of the answer once it has started. For intuition, a TPOT of 50 milliseconds means 20 tokens per second, and a TPOT of 25 milliseconds means 40 tokens per second. So, halving the TPOT doubles the streaming speed each user feels.

**Note:** For a single request, you can treat TPOT and ITL as the same idea, which is the per-token gap. Across many requests they are measured slightly differently, but for understanding they mean the same thing.

**Throughput** is the total number of tokens generated per second across all users being served at the same time. This is a system-wide metric, not a single-user metric. In simple words, throughput is how many users the server can serve together. Batching, which means running many requests through the GPU together, raises throughput. The group of requests run together is called a **batch**, and the number of requests in it is the **batch size**.

**End-to-end latency** is the total time from sending the request to receiving the final token. We can build an intuition for it with a simple formula:

```
End-to-end latency = TTFT + (number of output tokens - 1) x TPOT
```

Here, we have the first token already counted inside TTFT, which is why we use the number of output tokens minus one. First the prompt is processed (TTFT), then each remaining output token adds one TPOT.

Let's do a quick worked example. Suppose TTFT is 400 milliseconds, the answer is 200 tokens long, and TPOT is 25 milliseconds per token. The answer is 200 tokens, but the first token is already counted in the TTFT, so 199 tokens remain, each taking 25 milliseconds. Then the end-to-end latency is roughly 400 milliseconds plus 199 multiplied by 25 milliseconds, which is about 5.4 seconds.

Let's visualize this timeline.

```
request
  sent
    |
    v
    |<------ TTFT ----->|  TPOT  |  TPOT  |  TPOT  |  ...  |  TPOT  |
    | prefill + token 1 | token 2| token 3| token 4|  ...  | token N|

End-to-end latency  =  TTFT  +  (N - 1) x TPOT
```

Here, we can see that TTFT covers everything up to the first token, and then each later token arrives one TPOT apart. So, the answer starts after TTFT and then streams at the speed of TPOT.

Also, we must understand one core trade-off. TTFT and TPOT are per-user latencies, where smaller is better. Throughput is a system metric, where bigger is better. They pull against each other. Bigger batches push more tokens per second through the GPU (higher throughput) but make each user wait longer (worse TTFT and TPOT). Smaller batches feel snappier but waste GPU capacity. We tune the batch size based on our use case. A chatbot wants low TTFT and low TPOT, while a bulk document-summarization job wants high throughput.

Now, we know which metric each phase owns. Prefill owns TTFT. Decode owns TPOT and throughput. So, when we optimize, we always ask which metric we are trying to improve.

If we want to go deep into LLM Inference Optimization and Model Deployment and Serving, we cover it end to end in our [AI and Machine Learning Program](https://outcomeschool.com/program/ai-and-machine-learning) at Outcome School.

## Optimization techniques mapped to each phase {#optimization-techniques-mapped-to-each-phase}

Now, it's time to learn about the main optimization techniques. The beautiful thing is that each one targets a specific phase. So, we will say in plain words what each technique does and which phase or metric it helps.

### KV cache (the foundation, targets decode) {#kv-cache-the-foundation-targets-decode}

We have already learned this one. The KV cache stores the Keys and Values of all past tokens so that each decode step only does work for the one new token. This is what makes decode fast in the first place. But the cache grows with every token and fills GPU memory, which is exactly why all the other techniques below exist to manage it.

### Continuous batching (targets throughput) {#continuous-batching-targets-throughput}

Let's first see the old approach and its problem.

In the old approach, called static batching, the server groups several requests together and then waits for the slowest one in the group to finish before starting the next group. The issue with this approach is that short requests finish early and their slots sit empty, wasting the GPU. Let's see how the next approach solves this issue.

[**Continuous batching**](https://outcomeschool.com/blog/continuous-batching-in-llms) makes the batching decision every single decode step instead of once per group. The moment one request finishes, its slot is freed, and a waiting request joins the batch on the very next step. So, the batch stays full and the GPU stays busy.

Let's visualize the difference.

```
STATIC BATCHING  (finished slots sit empty -> GPU wasted)
  slot 1: R1=====                  (idle, wasted) .........
  slot 2: R2==========================
  slot 3: R3==========       (idle, wasted) ...............
          |------- whole batch waits for the slowest -------|

CONTINUOUS BATCHING  (a finished slot is refilled at once -> GPU busy)
  slot 1: R1=====R4====================
  slot 2: R2==========================
  slot 3: R3==========R5================
          (R4 enters the moment R1 ends, R5 enters when R3 ends)
```

Here, we can see that static batching leaves empty slots when short requests finish early, while continuous batching fills those slots right away.

Here is a simple analogy. It is like a shared taxi that, the moment one passenger gets out, immediately picks up the next person waiting at the curb, instead of driving around empty until everyone's trip is over. The taxi never runs with empty seats.

Continuous batching typically gives a large jump in throughput compared to the old static approach.

### Chunked prefill (targets smooth streaming for other users) {#chunked-prefill-targets-smooth-streaming-for-other-users}

Let's see the problem first. A very long prompt's prefill runs as one giant step. While that giant step runs, it hogs the GPU and freezes the token streaming of other users who are already in their decode phase. So, their answers stutter.

**Chunked prefill** splits one long prompt's prefill into several smaller chunks instead of one giant step. The model then slips small decode steps in between the chunks. So, no single long prefill blocks everyone else. This protects the token streaming of the other users so their answers do not stutter. As a bonus, mixing compute-heavy prefill with memory-heavy decode in one batch uses the hardware better.

Let's visualize this.

```
WITHOUT chunked prefill:
  long prompt: [============= ONE BIG PREFILL =============]
  other users: tok ....... (frozen, stuttering) ....... tok

WITH chunked prefill:
  long prompt: [chunk 1][chunk 2][chunk 3][chunk 4][chunk 5]
  other users: tok   tok   tok   tok   tok   tok
                  ^ a decode step slips in between the chunks
```

Here, we can see that one giant prefill freezes the other users, while chunked prefill lets their tokens keep flowing between the chunks.

Here is a simple analogy. It is like a chef with one stove. If a customer orders a huge banquet, cooking it all at once freezes every other table's food. Instead, the chef cooks the banquet in small batches, slipping in the quick single dishes between them, so nobody's plate goes cold.

**Note:** Chunked prefill mainly protects ongoing answers from stalling. It does not change the total math, so it does not by itself make prefill faster.

### Prefix caching, also called prompt caching (targets prefill and TTFT) {#prefix-caching-also-called-prompt-caching-targets-prefill-and-ttft}

Many requests start with the same long beginning. For example, the same fixed system instructions, the same document for a question, or the same earlier turns of a chat. This shared beginning is called a **prefix**.

[**Prefix caching**](https://outcomeschool.com/blog/how-does-prompt-caching-work) reuses the KV cache for this shared prefix across requests. The prefill for that shared part is computed once and then reused, instead of being recomputed for every request. So, it skips redundant work and shrinks TTFT.

Here is a simple analogy. It is like a teacher who writes the same long instructions on the board at the start of every class. Instead of rewriting them each time, the teacher keeps a photo of the board and reuses it, spending the saved time only on each student's new question.

In production, this can give very high cache hit rates and large cost savings on repeated prompts. A high cache hit rate means most requests found their shared beginning already stored and ready, so that work was skipped. **Note:** Prefix caching only helps when requests actually share a common beginning. If there is no shared prefix, it falls back to a full prefill.

### Disaggregated serving, also called prefill-decode separation (targets both TTFT and TPOT) {#disaggregated-serving-also-called-prefill-decode-separation-targets-both-ttft-and-tpot}

We learned that prefill is compute-bound and decode is memory-bandwidth-bound. When one GPU does both, a heavy prefill burst stalls the ongoing decodes of other users, and we cannot tune each phase separately.

**Disaggregated serving** runs prefill on one pool of GPUs and decode on another pool of GPUs. A pool just means a group of GPUs working together. The KV cache produced by the prefill GPU is transferred over a fast link to the decode GPU. Now each pool is set up for its own bottleneck, and each can meet its own latency target independently.

Let's visualize the two pools.

```
PREFILL POOL                             DECODE POOL
   +-------------------------+              +-------------------------+
   | GPUs tuned for heavy    |              | GPUs tuned for fast     |
   | math (compute-bound)    |   KV cache   | memory (memory-bound)   |
   |                         |  ========>   |                         |
   | reads prompt, writes    |  fast link   | reads cache, streams    |
   | the KV cache, makes     |              | the answer token by     |
   | the first token         |              | token                   |
   +-------------------------+              +-------------------------+
```

Here, we can see that the prefill pool builds the KV cache and passes it over a fast link to the decode pool, and each pool is tuned for its own bottleneck.

Here is a simple analogy. It is like a restaurant that puts the heavy prep kitchen in one room and the steady plating-and-serving line in another. Each room is staffed and equipped for its own kind of work, and the prepped food (the KV cache) is passed between them, so neither slows the other down.

**Note:** Disaggregation is not free. It adds the cost of moving the KV cache over the network, and it duplicates the model across pools. It pays off mainly at large scale when we must hit strict TTFT and TPOT targets at the same time. For smaller setups, continuous batching plus chunked prefill on a single GPU is often enough.

The SGLang serving system supports this prefill-decode disaggregation, along with chunked prefill and prefix caching. We have a detailed blog on [how SGLang works](https://outcomeschool.com/blog/how-does-sglang-work) that covers these in depth.

### Speculative decoding (targets decode and TPOT) {#speculative-decoding-targets-decode-and-tpot}

This one is clever, and it works because decode is memory-bound.

[**Speculative decoding**](https://outcomeschool.com/blog/speculative-decoding) uses a small fast model, called the draft model, to guess the next several tokens cheaply. Then the big model, called the target model, checks all of those guesses in one single parallel pass. The correct guesses are kept. The first wrong guess and everything after it is thrown away and corrected by the big model.

Why does this help? Because decode is memory-bound, the big model's weights must be read from memory anyway. Checking 5 guessed tokens at once moves almost the same amount of data from memory as producing 1 token. So, every correct guess is almost free. This turns a slow one-at-a-time process into a faster parallel one.

Let's visualize one round of speculative decoding.

```
Draft model (small, fast) proposes 5 tokens in one cheap pass:
        g1   g2   g3   g4   g5

Target model (big) verifies ALL of them in ONE parallel pass:
        ok   ok   ok   X    -
                       ^
                       first wrong guess: corrected by the big model,
                       g5 is thrown away

Result: 3 guesses accepted + 1 fix, for about the cost of 1 normal step
```

Here, we can see that the big model accepts the correct guesses and fixes the first wrong one in a single pass, so several tokens are produced for about the cost of one.

Here is a simple analogy. A fast junior assistant guesses the next few words of the boss's sentence, and the boss reads the whole guess in one glance, then either nods at the correct part or fixes the first mistake. Checking a guess is far quicker than composing from scratch.

**Very important:** Speculative decoding is designed to preserve the model's output quality, and it only changes the speed. When the model picks the single most likely token every time, the tokens are exactly what the big model would have produced on its own. When the model picks tokens with some randomness, the output is drawn from the very same set of probabilities the big model uses, so the quality is preserved even though the exact wording can differ run to run. Typical speedups are around 2 to 3 times for decode, depending on how often the small model's guesses are accepted.

### PagedAttention (targets KV cache memory efficiency) {#pagedattention-targets-kv-cache-memory-efficiency}

Let's see the problem first. In a naive system, the KV cache for each request is stored as one big continuous block of memory, sized for the longest possible answer. The issue with this approach is that most of that reserved space stays empty and is wasted. This waste can be very large.

[**PagedAttention**](https://outcomeschool.com/blog/paged-attention-in-llms) stores the KV cache in small fixed-size blocks, typically 16 tokens each, that can live anywhere in GPU memory, instead of one big continuous chunk per request. A small table, called the block table, remembers which scattered blocks belong to which request. A new block is added only when the sequence actually grows.

Let's visualize the difference.

```
WITHOUT paging (one big contiguous block reserved up front):
  Request A: [used][used][used][------ reserved but empty ------]
  Request B: [used][used][--------- reserved but empty ---------]
             wasted empty space sits unused inside each block

WITH paging (small fixed-size blocks, given only when needed):
  Request A -> block 7, block 2, block 9     (scattered anywhere)
  Request B -> block 4, block 1
  a block table remembers which blocks belong to which request
  shared prefix? two requests can point to the SAME block
```

Here, we can see that the contiguous approach reserves a big block and wastes the empty part, while paging hands out small blocks only as the sequence grows.

This borrows an old, proven computer trick, called paging, where memory is handed out in small equal pieces only when they are actually needed.

Here is a simple analogy. Instead of demanding one big empty parking lot for each car's luggage, the valet uses many small numbered lockers scattered around the building and keeps a slip saying which lockers belong to whom. No space is wasted on empty reserved lots. And if two people have identical luggage (a shared prefix), they can simply share the same locker.

PagedAttention nearly removes the wasted memory, lets us fit bigger batches, and raises throughput. It is also the foundation that makes prefix caching possible, because shared prefix blocks can simply point to the same physical block.

**Note:** PagedAttention is the memory layout that prevents waste, while prefix caching is the reuse of already-computed KV across requests. They are distinct ideas, but PagedAttention is what makes prefix caching's block sharing possible.

PagedAttention was introduced by the vLLM serving system. We have a detailed blog on [how vLLM works](https://outcomeschool.com/blog/how-does-vllm-work) that covers this end to end.

### A quick map of which technique helps which phase {#a-quick-map-of-which-technique-helps-which-phase}

Let me tabulate these techniques and the phase or metric they target for your better understanding.

| Technique | Main target | What it improves |
|----|----|----|
| KV cache | Decode | Makes decode fast by avoiding recomputation |
| Continuous batching | Both phases | Throughput (more users served together) |
| Chunked prefill | Serving smoothness for other users | Prevents long prefills from stalling ongoing token streaming (protects others' ITL) |
| Prefix caching | Prefill | TTFT (skips redundant prompt processing) |
| Disaggregated serving | Both phases | TTFT and TPOT independently, at scale |
| Speculative decoding | Decode | TPOT (faster token generation) |
| PagedAttention | KV cache memory | Throughput (bigger batches, less waste) |

This table is a handy mental map. When we want a faster first word, we look at the prefill side. When we want faster streaming or more users at once, we look at the decode side.

## Conclusion {#conclusion}

Now, let's recap everything in a simple timeline.

An LLM answers in two phases. **Prefill** reads your whole prompt in one parallel pass, fills the KV cache, and produces the first token. It is compute-bound, and it sets TTFT, which is how fast the answer starts. **Decode** then generates the rest of the answer one token at a time, reading and extending the KV cache at each step. It is memory-bandwidth-bound, and it sets TPOT, which is how fast the answer streams.

The **KV cache** is the bridge between the two phases. Prefill writes it, and decode reads and grows it. It saves us from recomputing the past at every step, but it grows with every token and fills GPU memory.

Because the two phases have opposite bottlenecks, each optimization targets one of them. Continuous batching and PagedAttention raise throughput. Prefix caching and chunked prefill help prefill and smooth streaming. Speculative decoding speeds up decode. Disaggregated serving splits the two phases so each can hit its own target.

Each of these optimization techniques has its own dedicated deep-dive, collected in our [LLM Inference Optimization](https://outcomeschool.com/blog/llm-inference-optimization) blog.

This was all about Prefill vs Decode.

Prepare yourself for AI Engineering Interview: [AI Engineering Interview Questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions)

That's it for now.

Thanks

**Amit Shekhar** \
Founder @ [Outcome School](https://outcomeschool.com)

You can connect with me on:

- [LinkedIn](https://www.linkedin.com/in/amit-shekhar-iitbhu)
- [GitHub](https://github.com/amitshekhariitbhu)

Follow Outcome School on:

- [YouTube](https://youtube.com/@OutcomeSchool)
- [LinkedIn](https://www.linkedin.com/company/outcomeschool)
- [GitHub](https://github.com/OutcomeSchool)

[**Read all of our high-quality blogs here.**](https://outcomeschool.com/blog)

Subscribe to our [newsletter](https://outcomeschool.substack.com/subscribe) to get our latest AI and Machine Learning blogs straight to your inbox.
