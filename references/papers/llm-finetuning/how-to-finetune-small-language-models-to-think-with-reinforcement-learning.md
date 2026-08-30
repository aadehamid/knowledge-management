title: How to Fine-Tune Small Language Models to Think with Reinforcement Learning | Towards Data Science
description: A visual tour and from-scratch guide to train GRPO reasoning models in PyTorch
author: Avishek Biswas

# How to Fine-Tune Small Language Models to Think with Reinforcement Learning | Towards Data Science

**Reasoning models** are currently in fashion. DeepSeek-R1, Gemini-2.5-Pro, OpenAI's O-series models, Anthropic's Claude, Magistral, and Qwen3 — there is a new one every month. When you ask these models a question, they go into a *chain of thought* before generating an answer.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/results_gif3-2.gif "A simple demonstration of what reasoning looks like. When asked a question, the Language Model (LM) generates a chain of thought first, followed by the answer. (Illustration by the Author)")

I recently asked myself the question, "Hmm... I wonder if I should write a Reinforcement Learning loop from scratch that teaches this 'thinking' behaviour to *really* small models — *like only 135 million* *parameters*". It should be easy, right?

Well, it wasn't.

> **Small models simply do not have the world knowledge that large models do. This makes \< 1B parameter model lack the "common sense" to easily reason through complex logical tasks. Therefore, you cannot just rely on compute to train them to reason.**  \
>  \
>  **You need additional tricks up your sleeve.**

In this article, I won't just cover tricks though. I will cover the major ideas behind training reasoning behaviours into language models, share some simple code snippets, and some practical tips to fine-tune Small Language Models (SLMs) with RL.

**This article is divided into 5 sections:**

1. Intro to RLVR (Reinforcement Learning with Verifiable Rewards) and why it is uber cool
2. A visual overview of the GRPO algorithm and the clipped surrogate PPO loss.
3. A code walkthrough!
4. Supervised fine-tuning and practical tips to train reasoning models
5. Results!

*Unless otherwise mentioned, all images used in this article are illustrations produced by the author.*

*At the end of this article, I will link to the 50-minute companion YouTube video of this article. If you have any queries, that video likely has the answers/clarification you need. You can also reach out to me on X (* [*@neural\_avb*](https://x.com/neural_avb) *).*

## 1. Reinforcement Learning with Verifiable Rewards (RLVR) {#1-reinforcement-learning-with-verifiable-rewards-rlvr}

Before diving into specific challenges with Small models, let's first introduce some terms.

Group Relative Policy Optimization, or GRPO, is a (rather new) Reinforcement Learning (RL) technique that researchers are using to fine-tune Large Language Models (LLMs) on logical and analytical tasks. Since its inception, a new term has been circulating in the LLM research space: **RLVR**, or **R**einforcement **L**earning with **V**erifiable **R**ewards.

To understand what makes RLVR unique, it's helpful to contrast it with the most common application of RL in language models: RLHF (**R**einforcement **L**earning with **H**uman **F**eedback). In RLHF, an RL module is trained to maximize scores from a separate reward model, which acts as a proxy for human preferences. This reward model is trained on a dataset where humans have ranked or rated different model responses. 

> **In other words, RLHF is trained so LLMs can output responses that are more aligned with human preferences. It tries to make models follow instructions more closely.**
> **RLVR tries to solve a different problem. RLVR teaches a model to be verifiably correct, often by learning to generate it's own chain of thought.**

Where RLHF had a *subjective* reward model, RLVR uses an *objective* verifier. The core idea is to provide rewards based on whether an answer is demonstrably correct, not on a prediction of what a human might prefer.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/RLVR2-1024x682.png "An illustration of how RLVR works (Illustrated by the Author)")

This is exactly why this system is called 'RL with *verifiable rewards*'. Not every question's answer can be verified easily. Especially open-ended questions like *"What iPhone should I buy?*" or "*Where should I go to college?"*. Some use cases, however, do fit easily in the "verifiable rewards" paradigm, like math, logical tasks, and code-writing, to name a few. In the `reasoning-gym` section below, we will look into how exactly these tasks can be simulated and how the rewards can be generated.

*But before that, you might ask: well where does "reasoning" fit into all of this?*

We will train the LLM to generate arbitrarily long chain of thought reasoning texts before generating the final answer. We instruct the model to wrap its thinking process in `<think>` tags and its final conclusion in `<answer>` tags. 

The full language model response will look something like this:

markup

```
<think>User has asked me to count the number of r's in strawberry.Let's do a cumulative count.s=0, t=0, r=1, a=0, w=0, b=0, e=0, r=2, r=3, y=4It seems there are 3 r's in strawberry. I notice that there is an r in straw and 2 r's in berry.Since 1+2=3 I am more confident there are 3 r's</think><answer>3</answer>
```

This structure allows us to easily extract just the final answer and check if it's correct. The verifier is a single source of truth, and can be a simple piece of code that (literally) counts alphabets.

python

```
def count_alphabets(word, letter):    return sum([1 for l in word if l == letter])reward = 1 if (lm_answer == count_alphabets("strawberry", "r") else -1
```

We will keep a record of the model's experiences — its responses and the corresponding rewards received from the verifier. The RL algorithm will then train to promote behaviours that increase the likelihood of correct final answers.

> **By consistently rewarding correct answers and good formatting, we would increase the likelihood of reasoning tokens that lead to correct answers.** 
> **Get this: we don't**  ***need***  **to directly evaluate the intermediate reasoning tokens. By simply rewarding the final answer, we will indirectly elicit reasoning steps into the LLM's chain of thought that lead to correct answers!**

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/DR1-Zero-1-1024x451.png "Source: Some exercepts from the DeepSeek-R1 paper (License: Free)")

## 2. GRPO (Group Relative Policy Optimization) {#2-grpo-group-relative-policy-optimization}

I am going to skip the usual *Reinforcement Learning 101* intro here, I expect most of you who read this far to understand the basics of RL. There is an agent who observes states from the environment and takes an action — the environment rewards the agent depending on how good the action was — the agent stores these experiences and trains to take better actions in the future that lead to higher rewards. *RL 101* class dismissed.

*But how do we transfer the RL paradigm to language?*

Let's talk about our algorithm of choice — **G**roup **R**elative **P**olicy **O**ptimization to understand how. GRPO works in two iteratively self-repeating phases — an experience collection phase where the Language Model (LM) accumulates experiences in the environment with its current weights. And a training phase where it uses the collected memories to update its weights to improve. After training, it once again goes into an experience collection step with the updated weights.

### Experience Collection {#experience-collection}

Let's dissect each step in the experience collection phase now.

- **Step 1:** The environment is a black box that generates questions about logical or math tasks. We will discuss this in an upcoming section with the `reasoning-gym` library.

- **Step 2:** We tokenize the input questions into a sequence of integer tokens.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/GRPO1-1-1024x576.png "Sample questions, tokenized them, forward pass through LM, and generate multiple responses for each question! (Illustrated by the Author)")

- **Step 3:** The "agent" or the "policy" is the current SLM we are training. It observes the environment's tokenized questions and generates responses. The LLM response gets converted into text and returned to the environment. The environment rewards each response.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/GRPO2-1-1024x576.png "The Environment acts as the verifier and assigns a reward to the agent. (Illustrated by the Author)")

- **Step 4:** From the rewards, we calculate the **advantage** of each response. In GRPO, the advantage is the relative goodness of each response in the group. Importantly, advantages are calculated per group, i.e. we do not standardize rewards *across* different questions.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/GRPO3-1-1024x576.png "Advantages define how favourable a specific response is relative to other responses to the same question (Illustrated by the Author)")

- **Step 5:** The original question, the log probabilities for each LM-generated token, and the advantages are all accumulated inside a memory buffer.
- Steps 1-5 are repeated till the buffer size reaches the desired threshold.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/GRPO4-1024x576.png "Saving experiences in the buffer! (Illustrated by the Author)")

### Training Phase {#training-phase}

After the end of the experience collection phase, our goal is to enter the training phase. Here, we will learn from the reward patterns the LLM observed and use RL to improve its weights. Here is how that works:

1. Randomly sample a minibatch of memories. Remember, each memory already contained its group-relative-advantage (Step 5 from the experience collection phase). Randomly sampling question-answer pairs improves the robustness of the training as the gradients are calculated as an average of a diverse set of experiences, preventing over-fitting on any single question.
2. For each minibatch, we want to maximize this term following the standard PPO (Proximal Policy Optimization) formulation. **The major difference with GRPO is that we do not need an additional reward model or a value network to calculate advantages. Instead, GRPO samples multiple responses to the same question to calculate the relative advantage of each response.** The memory footprint is significantly reduced since we won't need to train those additional models!
3. Repeat the above steps.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/2_steps-1-1024x576.jpg "GRPO operates in 2 repeating phases — collect experiences, train on experiences, repeat. (Illustrated by the Author)")

### What the PPO Loss means {#what-the-ppo-loss-means}

Let me explain the PPO Loss in an intuitive step-by-step fashion. The PPO Loss looks like this. 

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/PPO_Loss_Full-1024x576.png "The PPO Loss Function. Let me break it down for you. (Illustration by the Author)")

- Here, `pi_old` is the old-policy neural network that we used during the data collection phase.

- `π` is the current policy neural network we are training. Since the weights of `π` change after each gradient update, `π` and `π_old` do not remain the same during the training phase — hence the distinction.

- `G` is the number of generated responses for a single question. `|o_i|` is the length of the i-th response in the group. Therefore, those summation and normalization operation computes a mean over all the tokens over all responses. What does it compute the mean of? Well it is `π/π_old * A_{it}`. What does that mean?

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/PPO_Loss_ADV-1024x576.png "The simplest way to assign an advantage to each token is by copying the advantage of the entire response (Illustrated by the Author)")

- `A_it` is the advantage of the t-th token in the i-th response. Remember when we calculated the advantage of each response in Step 5 during experience collection? The easiest way to assign an advantage to each token is by simply duplicating the same advantage to each token — this means we are saying that every token is equally responsible for generating the correct answer.

- Lastly, what is `π(o_it | q, o_i < t)`? It means what is the probability of the `t-th` token in the `i-th` response? Meaning, how *likely* was that token when it was generated?
- The importance sampling ratio reweights the advantages between the current updating policy and the old exploration policy.
- The clipping term ensures that the updates to the network do not become too large and the weights do not move too far away from the old policy. This adds more stability to the training process by keeping the model updates close to "a trust region" from the data-collection policy.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/Eqn2-1024x576.png "The PPO objective broken down into individual components. (Illustrated by the Author)")

> **When we are maximizing the PPO objective, we are effectively asking the LLM to**  ***increase***  **the log-probability of the tokens that led to a high advantage, while**  ***decreasing***  **the log-probability of tokens that had a low advantage.**
> In other words: make tokens that generate good advantages more likely and tokens that generate low advantages less likely.

### Understanding the PPO Loss with an example {#understanding-the-ppo-loss-with-an-example}

Let's forget about the clipping term and the `π_old`for now, and let's just see what maximizing `𝜋(𝑜_i) * A_i` means. To remind you, this part of the equation simply means, "the product of the probability of the i-th token (o\_i) and the advantage of the i-th token (A\_i)

Let's say for a question, the LLM generated these two sequences: "A B C" and "D E F", and it got an advantage of \+1 for the former and -1 for the latter\*. Let's say we have the log probabilities for each of the 3 tokens as shown below. 

\* *actually since group-relative advantages always have a standard deviation of 1, the correct advantages should be \+0.707 and -0.707.*

Notice what happens when you multiply the advantages `A_it` by the current logprobs `pi`. Now really think about what it means to maximize the mean of that product matrix.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/Slide2-1024x576.jpeg "A toy example to show what it means to maximize the product of the probability of a token with it's advantage (Illustrated by the Author)")

Remember we can only change the probabilities coming out of the LLM. The advantages come from the environment and are therefore treated as constants. Increasing this expected score would therefore mean increasing the probability of tokens with a positive advantage, and decreasing the value of the negative advantage example. 

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/Slide6-1024x576.jpeg "To increase the mean of the product tensor, we must increase each value in the tensor, so we must increase the probs of positive advantage-tokens, and decrease the probs of negative-advantage tokens. (Illustrated by the Author)")

Below, you will find an example of how log-probs change after a few rounds of training. Notice how the blue line is moving closer to zero when the advantage is high? This indicates that the log-probabilities increased (or the probabilities increased) after going through RL Training. Compare that to the plot on the right, which shows a different response with a low advantage. The blue line is moving away from 0, becoming less probable for selection in later rounds.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/comparison-log-probs-1024x464.png "A comparison of how RL fine-tuning affects log-probs of tokens after training (Illustration by the Author)")

In the next section, let's take a look at the `reasoning-gym` library and understand how we could sample tasks.

## 3. Implementation {#3-implementation}

So, to do RL, we first need tasks. A common way to do this is by using an existing dataset of math problems, like the GSM-8K dataset. In this article, let's look at a different case — generating tasks procedurally with a Python library called [reasoning-gym](https://github.com/open-thought/reasoning-gym).

For my experiments, I used two tasks: **syllogism** and **propositional logic**. `reasoning-gym` contains a host of different repositories of varying difficulty.

A **syllogism task** is a type of logical puzzle designed to test deductive reasoning. Basically, we will provide the LLM with two premises and ask if the conclusion is correct or not. The **propositional logic** task is a symbolic reasoning task where the LLM is provided tasks with symbols and asked to generate the conclusion. Unlike syllogism, this is not a YES/NO classification response — they have to generate the correct conclusion directly. This makes this task considerably harder.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/results_gif2.gif "Example of the Syllogism Task (Footage of my RL-trained model)")

*Before we begin coding, I guess it is customary to specify what I mean by "small" models.* 

The jury is still out on what qualifies as a "small" model (some say \<14B, some say \<7B), but for my YouTube video, I picked even smaller models: SmolLM-135M-Instruct, SmolLM-360M-Instruct, and Qwen3-0.6B. These are \~135M, \~360M, and \~600M models, respectively.

Let's see how to set up the basic training loop. First, we can use Huggingface's `transformers` library to load in a model we want to train, let's say the little 135M param model `SmolLM-135M-Instruct`.

To generate some propositional logic tasks, for example, you just call this `reasoning_gym.create_dataset function` as shown below.

python

```
import refrom reasoning_gym import create_dataset, get_score_answer_fnfrom transformers import AutoModelForCausalLM, AutoTokenizerimport torchmodel_name = "HuggingfaceTB/SmolLM-135M-Instruct"# load model from huggingfacelm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)tokenizer = AutoTokenizer.from_pretrained(model_name)# This sets all models as trainablefor param in lm.parameters():    param.requires_grad = True# In my experiments, I used a LORA adapter (more on this later)# specify name of the env environment_name = "propositional_logic"# In practice, you should wrap this with a torch dataloader # to sample a minibatch of questionsdataset = create_dataset(    environment_name, seed=42, size=DATA_SIZE)for d in dataset:    question = d["question"] # Accessing the question         # We will use this later to verify if answer is correct    validation_object = d["metadata"]["source_dataset"]    score_fn = get_score_answer_fn(validation_object)
```

To generate reasoning data, we want the LM to generate thinking, followed by the response. Below is the system prompt we will be using.

python

```
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the userwith the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.Do not generate new code. Do not write python code.You may also be given examples by the user telling you the expected response format.Follow the format of the examples, but solve the specific problem asked by the user, not the examples.Very important - Remember again, your output format should be:<think> reasoning process here </think><answer> answer here </answer>Your response will be scored by extracting the substring between the <answer>...</answer> tags.It is critical to follow the above format.feature_extraction_utilsling to follow the response format will result in a penalty."""
```

To generate answers, we first tokenize the system prompt and the question as shown below.

python

```
# Create messages structuremessages = [    {"role": "system", "content": system_prompt},    {"role": "user", "content": question}, # Obtained from reasoning-gym]# Create tokenized representationinputs = tokenizer.apply_chat_template(    messages,    tokenize=True,    return_tensors="pt",    add_generation_prompt=True)
```

Then we pass it through the LM — generate multiple responses using the `num_return_sequences` parameter, and detokenize it back to get a string response. No gradients are calculated during this stage.

python

```
generated_response = lm.generate(    input_ids=inputs["input_ids"],    attention_mask=inputs["attention_mask"],    max_new_tokens=max_new_tokens, # The max number of tokens to generate    do_sample=True,                # Probabilistic sampling    top_p=0.95,                    # Nucleus sampling    num_return_sequences=G,        # Number of sequences per question    temperature=1,                 # Increase randomness    eos_token_id=eos_token_id,    pad_token_id=eos_token_id,)
```

We also write the `extract_answer` function, which uses regular expressions to extract answers between the answer tags.

python

```
def extract_answer(response):    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)    if answer is not None:        return answer.group(1).strip()    else:        return ""
```

Finally, we use the score function we got previously to generate a reward depending on whether the LM's response was correct. To calculate rewards, we add a format reward and a correction reward. The correction reward comes from the environment, and the format reward is awarded if the model correctly generates the `<think> ... </think>` and `<answer> ... </answer>` tags.

The advantages are calculated by standardizing across each group.

python

```
# Response is an array of string of length [B*G]# B is the number of questions, G is the number of responses per questioncorrectness_reward = score_fn(response, validation_object)format_reward = calculate_format_reward(response)# Total reward is a weighted sum of correctness and formatting rewardsreward = correctness_reward * 0.85 + format_reward * 0.15 # Convert rewards from [B*G, 1] -> [B, G]rewards = rewards.reshape(B, G) # Calculate advantagesadvantages = (rewards - np.mean(rewards, axis=1, keepdims=True)) / (    np.std(rewards, axis=1, keepdims=True) + 1e-8)advantages = advantages.reshape(-1, 1)
```

Store the (old) log probs, advantages, responses, and response masks in a memory buffer.

python

```
# A function that returns the log prob of each selected tokenlog_probs = calculate_log_probs(lm, generated_response)buffer.extend([{    "full_response": generated_response[i],    "response_mask": response_mask[i], # A binary mask to denote which tokens in generated response are AI generated, 0 for system prompt and questions    "old_log_probs": log_probs[i],    "advantages": advantages[i]} for i in range(len(generated_response))])
```

After multiple experience collection step, once the buffer is full, we initiate our training loop. Here, we sample minibatches from our experience, calculate the log probs, compute loss, and backdrop. 

python

```
# full_response, response_mask, old_log_probs, advantages <--- Buffer# Recompute the new log_probs. Notice no torch.no_grad(), so gradients WILL BE USED here.logits = llm(input_ids=full_response).logits# Extract log probs from the logits# Does log_softmax over the vocabulary and extracts the log-prob of each selected tokenlog_probs = calculate_log_probs(     logits,     full_responses)# Calculate the clipped surrogate lossreasoning_loss = calculate_ppo_loss(     log_probs,       # Trainable     old_log_probs,   # Obtained from exploration, not trainable     advantages,      # Obtained from environment, not trainable     response_mask    # Obtained from exploration, not trainable) # Optimizaiton stepsaccelerator.backward(reasoning_loss)optimizer.step()optimizer.zero_grad()
```

You can use additional entropy losses here, or minimize KLD with your reference model as suggested in the original Deepseek-R1 paper, but future papers have concluded that these leash the training process and not a requirement.

## 4. Warming up with Supervised Fine-tuning {#4-warming-up-with-supervised-fine-tuning}

Technically, we can try to run a big RL training right now and hope that the small models can pull through and conquer our tasks. However, the probability of that is incredibly low.

There is one big problem — our small models are not appropriately trained to generate formatted outputs or perform well on these tasks. Off the box, their responses do have *some* logical flow to them, thanks to the pretraining or instruction tuning from their original developers, but they are not good enough for our target task.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/slm_fails-1024x576.png "Comparing the outputs of a small model with a Large LM (Illustration by Author)")

Think about it — RL trains by collecting experiences and updating the policy to maximize the good experiences. But if most of the experiences are completely bad and the model receives 0 rewards, it has no way to optimize, because it gets no signal to improve at all. So the recommended approach is to first teach the model the behavior you want to train using supervised fine-tuning. Here is a simple script:

python

```
client = openai.AsyncClient()ENVIRONMENT = "propositional_logic"model = "gpt-4.1-mini"semaphore = asyncio.Semaphore(50)num_datapoints = 200system_prompt = (    system_prompt    + """You will also be provided the real answer. Your thinking should eventually result in producing the real answer.""")dataloader = create_dataset(name=ENVIRONMENT, size=num_datapoints)@backoff.on_exception(backoff.expo, openai.RateLimitError)async def generate_response(item):    async with semaphore:        messages = [            {"role": "system", "content": system_prompt},            {                "role": "user",                "content": f"""    Question: {item['question']}    Metadata: {item['metadata']}    Answer: {item['answer']}                    """,            },        ]        response = await client.chat.completions.create(messages=messages, model=model)        return {            "question": item["question"],            "metadata": item["metadata"],            "answer": item["answer"],            "response": response.choices[0].message.content,        }async def main():    responses = await asyncio.gather(*[generate_response(item) for item in dataloader])    fname = f"responses_{ENVIRONMENT}_{model}.json"    json.dump(responses, open(fname, "w"), indent=4)    print(f"Saved responses to {fname}")if __name__ == "__main__":    asyncio.run(main())
```

To generate the fine-tuning dataset, I first generated the thinking and answer tags with a small LLM-like GPT-4.1-mini. Doing this is incredibly simple — we sample 200 or so examples for each task, call the OpenAI API to generate a response, and save it on disk.

During SFT, we load the base model we want to train, attach a trainable LORA adapter ,and do parameter-efficient fine-tuning. Here are the LORA configurations I used.

python

```
lora:  r: 32  lora_alpha: 64  lora_dropout: 0  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj",                    "up_proj", "down_proj", "gate_proj"]
```

LORA allows the training process to be more memory efficient and also reduces the risk of corrupting the original model. You can find the details of parameter-efficient supervised fine-tuning in my YouTube video right here.

[\[YouTube Video\]](https://www.youtube.com/embed/bZcKYiwtw1I)

I trained a LORA adapter on 200 examples of syllogism data with the smallest language model I could find — the HuggingfaceTB/SmolLM-135M-Instruct, and it got us an accuracy of 46%. Roughly, this means that we generate a correct answer 46% of the time. More importantly, we often get the formatting right, so our regex can safely extract answers from the responses more often than not.

### Some more optimizations for SLMs and practical considerations {#some-more-optimizations-for-slms-and-practical-considerations}

1. Not all reasoning tasks can be solved by all models. **An easy way to verify if a task is too hard or too easy for the model is to just check the base accuracy of the model on your task**. If it is, let's say below 10-20%, the task is likely very hard and you need additional supervised warmup fine-tuning.
2. **SFT, even on small datasets, can generally show massive accuracy gains on small models.** If you can acquire a good dataset, you may not even need to do Reinforcement Learning in many scenarios. SLMs are immensely tunable.
3. Papers like [DAPO](https://arxiv.org/abs/2503.14476) and [Critical Perspectives on R1](https://arxiv.org/abs/2503.20783) have claimed that the original loss normalization from [DeepSeek](https://arxiv.org/pdf/2402.03300) has a **length bias**. They have proposed other normalization methods that are worth looking at. For my project, the regular DeepSeek loss just worked.
4. DAPO also mentions **removing the KLD term** in the original R1 paper. Originally, the goal of this loss was to ensure that the updating policy is never too far away from the base policy, but DAPO suggests not using this because the behaviour of the policy can drastically change during reasoning, making this KLD term an unnecessary regularisation term that will restrict the model's intelligence.
5. **Generating diverse responses IS KEY** to making RL possible. If you only generated correct responses, or if you only generated incorrect responses, the advantage will be 0, and this will give the RL algorithm no training signal at all. We can generate diverse responses by increasing the `temperature`, `top_p`, and `num_return_sequences` parameters in the `generate()`.
6. You can also generate **diverse rewards**, by adding more terms into the reward function. For example, a length reward that penalizes overly long reasoning.
7. The following parameters increase the **stability of training at the cost of more computation**: increasing num generations per rollout, increasing the size of the buffer and lowering the learning rate.
8. Use **gradient accumulation** (or even gradient checkpointing) if you have limited resources to train these models.
9. There is some fine print I skipped in this article related to **padding**. When saving experiences into buffer, it's best practice to remove the pad tokens altogether — and recreate them when loading a minibatch during training.
10. It is best to leave whitespace around \<think> and \<answer> (and their closing tags). This results in **consistent tokenization** and makes training slightly easier for the SLMs.

## 4. Results {#4-results}

Here is my YouTube video that explains everything in this blog post more pictorially and provides a hands-on tutorial on how to code such a thing.

[\[YouTube Video\]](https://www.youtube.com/embed/yGkJj_4bjpE)

On the supervised-fine-tuned SmolLM-135M on the syllogism task, we got a bump to 60%! You can see the reward curve here — the healthy standard deviation of the rewards shows that we were indeed getting diverse responses throughout, which is a healthy thing if we want to train with RL.

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/image-53-1024x576.png "Rewards curve of the Syllogism task on SmolLM-135M after SFT (Illustration by Author)")

Here is a set of hyperparameters that worked well for me.

yaml

```
config:  name: "path/to/sft_model"  max_new_tokens: 300 # reasoning + answer token budget  exploration_batchsize: 8  # number of questions per batch during rollout  G: 6  # num responses per group  temperature: 0.7  batch_size: 16  # minibatch size during training  gradient_accumulation_steps: 12  learning_rate: 0.000001  # Advisable to keep this low, like 1e-6 or 1e-7  top_p: 0.95  buffer_size: 500
```

I also repeated this experiment with larger models — the SmolLM-360M-Instruct and the Qwen3-0.6B model. In the latter, I was able to get accuracies up to 81% which is awesome! We got a 20% additive bump on average in the syllogism task!

In the propositional logic task, which in my opinion is a harder reasoning task, I also saw similar gains across all small models! I am sure that with more instruction tuning and RL fine-tuning, possibly on multiple tasks at once, we can raise the intelligence of these models a lot higher. Training on a single task can generate quick results which is what I wanted for this Youtube video, but it can also act as a bottleneck for the model's overall intelligence.

Let's end this article with a GIF of the small models outputting reasoning data and solving tasks. Enjoy, and stay magnificent!

![](https://assets.insightmediagroup.io/media/wp-content/uploads/2025/07/results_gif1-1.gif "SmolLM-135M after training on Propositional Logic Tasks (Source: Author)")

---

## References {#references}

**Author's YouTube channel**: [https://www.youtube.com/@avb\_fj](https://www.youtube.com/@avb_fj)

**Author's Patreon**: [www.patreon.com/NeuralBreakdownwithAVB](https://www.patreon.com/c/NeuralBreakdownwithAVB)

**Author's Twitter (X) account**: [https://x.com/neural\_avb](https://x.com/neural_avb)

Deepseek Math: https://arxiv.org/pdf/2402.03300 \
DeepSeek R1: https://arxiv.org/abs/2501.12948 \
DAPO: https://arxiv.org/abs/2503.14476 \
Critical Perspectives on R1: https://arxiv.org/abs/2503.20783\
Reasoning Gym Library: [github.com/open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym)

A good place to read about Reasoning: [https://github.com/willccbb/verifiers](https://github.com/willccbb/verifiers) \
 \
A great place to study code: [https://github.com/huggingface/trl/blob/main/trl/trainer/grpo\_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) \
