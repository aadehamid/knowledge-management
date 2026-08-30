# Prompt Engineering with Llama 2

Prompt engineering is using natural language to produce a desired response from a large language model (LLM).

This interactive guide covers prompt engineering & best practices with Llama 2.

## Introduction

### Why now?

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) introduced the world to transformer neural networks (originally for machine translation). Transformers ushered an era of generative AI with diffusion models for image creation and large language models (`LLMs`) as **programmable deep learning networks**.

Programming foundational LLMs is done with natural language – it doesn't require training/tuning like ML models of the past. This has opened the door to a massive amount of innovation and a paradigm shift in how technology can be deployed. The science/art of using natural language to program language models to accomplish a task is referred to as **Prompt Engineering**.

### Llama Models

In 2023, Meta introduced the [Llama language models](https://ai.meta.com/llama/) (Llama Chat, Code Llama, Llama Guard). These are general purpose, state-of-the-art LLMs.

Llama 2 models come in 7 billion, 13 billion, and 70 billion parameter sizes. Smaller models are cheaper to deploy and run (see: deployment and performance); larger models are more capable.

#### Llama 2
1. `llama-2-7b` - base pretrained 7 billion parameter model
1. `llama-2-13b` - base pretrained 13 billion parameter model
1. `llama-2-70b` - base pretrained 70 billion parameter model
1. `llama-2-7b-chat` - chat fine-tuned 7 billion parameter model
1. `llama-2-13b-chat` - chat fine-tuned 13 billion parameter model
1. `llama-2-70b-chat` - chat fine-tuned 70 billion parameter model (flagship)


Code Llama is a code-focused LLM built on top of Llama 2 also available in various sizes and finetunes:

#### Code Llama
1. `codellama-7b` - code fine-tuned 7 billion parameter model
1. `codellama-13b` - code fine-tuned 13 billion parameter model
1. `codellama-34b` - code fine-tuned 34 billion parameter model
1. `codellama-7b-instruct` - code & instruct fine-tuned 7 billion parameter model
2. `codellama-13b-instruct` - code & instruct fine-tuned 13 billion parameter model
3. `codellama-34b-instruct` - code & instruct fine-tuned 34 billion parameter model
1. `codellama-7b-python` - Python fine-tuned 7 billion parameter model
2. `codellama-13b-python` - Python fine-tuned 13 billion parameter model
3. `codellama-34b-python` - Python fine-tuned 34 billion parameter model

## Getting an LLM

Large language models are deployed and accessed in a variety of ways, including:

1. **Self-hosting**: Using local hardware to run inference. Ex. running Llama 2 on your Macbook Pro using [llama.cpp](https://github.com/ggerganov/llama.cpp).
    * Best for privacy/security or if you already have a GPU.
1. **Cloud hosting**: Using a cloud provider to deploy an instance that hosts a specific model. Ex. running Llama 2 on cloud providers like AWS, Azure, GCP, and others.
    * Best for customizing models and their runtime (ex. fine-tuning a model for your use case).
1. **Hosted API**: Call LLMs directly via an API. There are many companies that provide Llama 2 inference APIs including AWS Bedrock, Replicate, Anyscale, Together and others.
    * Easiest option overall.

### Hosted APIs

Hosted APIs are the easiest way to get started. We'll use them here. There are usually two main endpoints:

1. **`completion`**: generate a response to a given prompt (a string).
1. **`chat_completion`**: generate the next message in a list of messages, enabling more explicit instruction and context for use cases like chatbots.

## Tokens

LLMs process inputs and outputs in chunks called *tokens*. Think of these, roughly, as words – each model will have its own tokenization scheme. For example, this sentence...

> Our destiny is written in the stars.

...is tokenized into `["our", "dest", "iny", "is", "written", "in", "the", "stars"]` for Llama 2.

Tokens matter most when you consider API pricing and internal behavior (ex. hyperparameters).

Each model has a maximum context length that your prompt cannot exceed. That's 4096 tokens for Llama 2 and 100K for Code Llama. 


## Notebook Setup

The following APIs will be used to call LLMs throughout the guide. As an example, we'll call Llama 2 chat using [Replicate](https://replicate.com/meta/llama-2-70b-chat) and use LangChain to easily set up a chat completion API.

To install prerequisites run:

```python
pip install langchain replicate
```

```python
from typing import Dict, List
from langchain.llms import Replicate
from langchain.memory import ChatMessageHistory
from langchain.schema.messages import get_buffer_string
import os

# Get a free API key from https://replicate.com/account/api-tokens
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

# We'll default to the smaller 13B model for speed; change to LLAMA2_70B_CHAT for more advanced (but slower) generations
DEFAULT_MODEL = LLAMA2_13B_CHAT

def completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    llm = Replicate(
        model=model,
        model_kwargs={"temperature": temperature,"top_p": top_p, "max_new_tokens": 1000}
    )
    return llm(prompt)

def chat_completion(
    messages: List[Dict],
    model = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        elif message["role"] == "assistant":
            history.add_ai_message(message["content"])
        else:
            raise Exception("Unknown role")
    return completion(
        get_buffer_string(
            history.messages,
            human_prefix="USER",
            ai_prefix="ASSISTANT",
        ),
        model,
        temperature,
        top_p,
    )

def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def complete_and_print(prompt: str, model: str = DEFAULT_MODEL):
    print(f'==============\n{prompt}\n==============')
    response = completion(prompt, model)
    print(response, end='\n\n')

```

### Completion APIs

Llama 2 models tend to be wordy and explain their rationale. Later we'll explore how to manage the response length.

```python
complete_and_print("The typical color of the sky is: ")
```

```python
complete_and_print("which model version are you?")
```

### Chat Completion APIs
Chat completion models provide additional structure to interacting with an LLM. An array of structured message objects is sent to the LLM instead of a single piece of text. This message list provides the LLM with some "context" or "history" from which to continue.

Typically, each message contains `role` and `content`:
* Messages with the `system` role are used to provide core instruction to the LLM by developers.
* Messages with the `user` role are typically human-provided messages.
* Messages with the `assistant` role are typically generated by the LLM.

```python
response = chat_completion(messages=[
    user("My favorite color is blue."),
    assistant("That's great to hear!"),
    user("What is my favorite color?"),
])
print(response)
# "Sure, I can help you with that! Your favorite color is blue."
```

### LLM Hyperparameters

#### `temperature` & `top_p`

These APIs also take parameters which influence the creativity and determinism of your output.

At each step, LLMs generate a list of most likely tokens and their respective probabilities. The least likely tokens are "cut" from the list (based on `top_p`), and then a token is randomly selected from the remaining candidates (`temperature`).

In other words: `top_p` controls the breadth of vocabulary in a generation and `temperature` controls the randomness within that vocabulary. A temperature of ~0 produces *almost* deterministic results.

[Read more about temperature setting here](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683).

Let's try it out:

```python
def print_tuned_completion(temperature: float, top_p: float):
    response = completion("Write a haiku about llamas", temperature=temperature, top_p=top_p)
    print(f'[temperature: {temperature} | top_p: {top_p}]\n{response.strip()}\n')

print_tuned_completion(0.01, 0.01)
print_tuned_completion(0.01, 0.01)
# These two generations are highly likely to be the same

print_tuned_completion(1.0, 1.0)
print_tuned_completion(1.0, 1.0)
# These two generations are highly likely to be different
```

## Prompting Techniques

### Explicit Instructions

Detailed, explicit instructions produce better results than open-ended prompts:

```python
complete_and_print(prompt="Describe quantum physics in one short sentence of no more than 12 words")
# Returns a succinct explanation of quantum physics that mentions particles and states existing simultaneously.
```

You can think about giving explicit instructions as using rules and restrictions to how Llama 2 responds to your prompt.

- Stylization
    - `Explain this to me like a topic on a children's educational network show teaching elementary students.`
    - `I'm a software engineer using large language models for summarization. Summarize the following text in under 250 words:`
    - `Give your answer like an old timey private investigator hunting down a case step by step.`
- Formatting
    - `Use bullet points.`
    - `Return as a JSON object.`
    - `Use less technical terms and help me apply it in my work in communications.`
- Restrictions
    - `Only use academic papers.`
    - `Never give sources older than 2020.`
    - `If you don't know the answer, say that you don't know.`

Here's an example of giving explicit instructions to give more specific results by limiting the responses to recently created sources.

```python
complete_and_print("Explain the latest advances in large language models to me.")
# More likely to cite sources from 2017

complete_and_print("Explain the latest advances in large language models to me. Always cite your sources. Never cite sources older than 2020.")
# Gives more specific advances and only cites sources from 2020
```

### Example Prompting using Zero- and Few-Shot Learning

A shot is an example or demonstration of what type of prompt and response you expect from a large language model. This term originates from training computer vision models on photographs, where one shot was one example or instance that the model used to classify an image ([Fei-Fei et al. (2006)](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf)).

#### Zero-Shot Prompting

Large language models like Llama 2 are unique because they are capable of following instructions and producing responses without having previously seen an example of a task. Prompting without examples is called "zero-shot prompting".

Let's try using Llama 2 as a sentiment detector. You may notice that output format varies - we can improve this with better prompting.

```python
complete_and_print("Text: This was the best movie I've ever seen! \n The sentiment of the text is: ")
# Returns positive sentiment

complete_and_print("Text: The director was trying too hard. \n The sentiment of the text is: ")
# Returns negative sentiment
```


#### Few-Shot Prompting

Adding specific examples of your desired output generally results in more accurate, consistent output. This technique is called "few-shot prompting".

In this example, the generated response follows our desired format that offers a more nuanced sentiment classifer that gives a positive, neutral, and negative response confidence percentage.

See also: [Zhao et al. (2021)](https://arxiv.org/abs/2102.09690), [Liu et al. (2021)](https://arxiv.org/abs/2101.06804), [Su et al. (2022)](https://arxiv.org/abs/2209.01975), [Rubin et al. (2022)](https://arxiv.org/abs/2112.08633).



```python
def sentiment(text):
    response = chat_completion(messages=[
        user("You are a sentiment classifier. For each message, give the percentage of positive/netural/negative."),
        user("I liked it"),
        assistant("70% positive 30% neutral 0% negative"),
        user("It could be better"),
        assistant("0% positive 50% neutral 50% negative"),
        user("It's fine"),
        assistant("25% positive 50% neutral 25% negative"),
        user(text),
    ])
    return response

def print_sentiment(text):
    print(f'INPUT: {text}')
    print(sentiment(text))

print_sentiment("I thought it was okay")
# More likely to return a balanced mix of positive, neutral, and negative
print_sentiment("I loved it!")
# More likely to return 100% positive
print_sentiment("Terrible service 0/10")
# More likely to return 100% negative
```

### Role Prompting

Llama 2 will often give more consistent responses when given a role ([Kong et al. (2023)](https://browse.arxiv.org/pdf/2308.07702.pdf)). Roles give context to the LLM on what type of answers are desired.

Let's use Llama 2 to create a more focused, technical response for a question around the pros and cons of using PyTorch.

```python
complete_and_print("Explain the pros and cons of using PyTorch.")
# More likely to explain the pros and cons of PyTorch covers general areas like documentation, the PyTorch community, and mentions a steep learning curve

complete_and_print("Your role is a machine learning expert who gives highly technical advice to senior engineers who work with complicated datasets. Explain the pros and cons of using PyTorch.")
# Often results in more technical benefits and drawbacks that provide more technical details on how model layers
```

### Chain-of-Thought

Simply adding a phrase encouraging step-by-step thinking "significantly improves the ability of large language models to perform complex reasoning" ([Wei et al. (2022)](https://arxiv.org/abs/2201.11903)). This technique is called "CoT" or "Chain-of-Thought" prompting:

```python
complete_and_print("Who lived longer Elvis Presley or Mozart?")
# Often gives incorrect answer of "Mozart"

complete_and_print("Who lived longer Elvis Presley or Mozart? Let's think through this carefully, step by step.")
# Gives the correct answer "Elvis"
```

### Self-Consistency

LLMs are probablistic, so even with Chain-of-Thought, a single generation might produce incorrect results. Self-Consistency ([Wang et al. (2022)](https://arxiv.org/abs/2203.11171)) introduces enhanced accuracy by selecting the most frequent answer from multiple generations (at the cost of higher compute):

```python
import re
from statistics import mode

def gen_answer():
    response = completion(
        "John found that the average of 15 numbers is 40."
        "If 10 is added to each number then the mean of the numbers is?"
        "Report the answer surrounded by three backticks, for example: ```123```",
        model = LLAMA2_70B_CHAT
    )
    match = re.search(r'```(\d+)```', response)
    if match is None:
        return None
    return match.group(1)

answers = [gen_answer() for i in range(5)]

print(
    f"Answers: {answers}\n",
    f"Final answer: {mode(answers)}",
    )

# Sample runs of Llama-2-70B (all correct):
# [50, 50, 750, 50, 50]  -> 50
# [130, 10, 750, 50, 50] -> 50
# [50, None, 10, 50, 50] -> 50
```

### Retrieval-Augmented Generation

You'll probably want to use factual knowledge in your application. You can extract common facts from today's large models out-of-the-box (i.e. using just the model weights):

```python
complete_and_print("What is the capital of the California?", model = LLAMA2_70B_CHAT)
# Gives the correct answer "Sacramento"
```

However, more specific facts, or private information, cannot be reliably retrieved. The model will either declare it does not know or hallucinate an incorrect answer:

```python
complete_and_print("What was the temperature in Menlo Park on December 12th, 2023?")
# "I'm just an AI, I don't have access to real-time weather data or historical weather records."

complete_and_print("What time is my dinner reservation on Saturday and what should I wear?")
# "I'm not able to access your personal information [..] I can provide some general guidance"
```

Retrieval-Augmented Generation, or RAG, describes the practice of including information in the prompt you've retrived from an external database ([Lewis et al. (2020)](https://arxiv.org/abs/2005.11401v4)). It's an effective way to incorporate facts into your LLM application and is more affordable than fine-tuning which may be costly and negatively impact the foundational model's capabilities.

This could be as simple as a lookup table or as sophisticated as a [vector database]([FAISS](https://github.com/facebookresearch/faiss)) containing all of your company's knowledge:

```python
MENLO_PARK_TEMPS = {
    "2023-12-11": "52 degrees Fahrenheit",
    "2023-12-12": "51 degrees Fahrenheit",
    "2023-12-13": "51 degrees Fahrenheit",
}


def prompt_with_rag(retrived_info, question):
    complete_and_print(
        f"Given the following information: '{retrived_info}', respond to: '{question}'"
    )


def ask_for_temperature(day):
    temp_on_day = MENLO_PARK_TEMPS.get(day) or "unknown temperature"
    prompt_with_rag(
        f"The temperature in Menlo Park was {temp_on_day} on {day}'",  # Retrieved fact
        f"What is the temperature in Menlo Park on {day}?",  # User question
    )


ask_for_temperature("2023-12-12")
# "Sure! The temperature in Menlo Park on 2023-12-12 was 51 degrees Fahrenheit."

ask_for_temperature("2023-07-18")
# "I'm not able to provide the temperature in Menlo Park on 2023-07-18 as the information provided states that the temperature was unknown."
```

### Program-Aided Language Models

LLMs, by nature, aren't great at performing calculations. Let's try:

$$
((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))
$$

(The correct answer is 91383.)

```python
complete_and_print("""
Calculate the answer to the following math problem:

((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))
""")
# Gives incorrect answers like 92448, 92648, 95463
```

[Gao et al. (2022)](https://arxiv.org/abs/2211.10435) introduced the concept of "Program-aided Language Models" (PAL). While LLMs are bad at arithmetic, they're great for code generation. PAL leverages this fact by instructing the LLM to write code to solve calculation tasks.

```python
complete_and_print(
    """
    # Python code to calculate: ((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))
    """,
    model="meta/codellama-34b:67942fd0f55b66da802218a19a8f0e1d73095473674061a6ea19f2dc8c053152"
)
```

```python
# The following code was generated by Code Llama 34B:

num1 = (-5 + 93 * 4 - 0)
num2 = (4**4 + -7 + 0 * 5)
answer = num1 * num2
print(answer)
```

### Limiting Extraneous Tokens

A common struggle is getting output without extraneous tokens (ex. "Sure! Here's more information on...").

Check out this improvement that combines a role, rules and restrictions, explicit instructions, and an example:

```python
complete_and_print(
    "Give me the zip code for Menlo Park in JSON format with the field 'zip_code'",
    model = LLAMA2_70B_CHAT,
)
# Likely returns the JSON and also "Sure! Here's the JSON..."

complete_and_print(
    """
    You are a robot that only outputs JSON.
    You reply in JSON format with the field 'zip_code'.
    Example question: What is the zip code of the Empire State Building? Example answer: {'zip_code': 10118}
    Now here is my question: What is the zip code of Menlo Park?
    """,
    model = LLAMA2_70B_CHAT,
)
# "{'zip_code': 94025}"
```

## Additional References
- [PromptingGuide.ai](https://www.promptingguide.ai/)
- [LearnPrompting.org](https://learnprompting.org/)
- [Lil'Log Prompt Engineering Guide](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)


## Author & Contact

Edited by [Dalton Flanagan](https://www.linkedin.com/in/daltonflanagan/) (dalton@meta.com) with contributions from Mohsen Agsen, Bryce Bortree, Ricardo Juan Palma Duran, Kaolin Fire, Thomas Scialom.
