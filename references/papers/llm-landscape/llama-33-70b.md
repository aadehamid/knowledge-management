title: I can now run a GPT-4 class model on my laptop
description: Meta’s new Llama 3.3 70B is a genuinely GPT-4 class Large Language Model that runs on my laptop. Just 20 months ago I was amazed to see something that felt …
author: Simon Willison

## I can now run a GPT-4 class model on my laptop

9th December 2024

Meta’s new [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) is a genuinely GPT-4 class Large Language Model that runs on my laptop.

Just 20 months ago I was amazed to see something that felt GPT-3 class run on that same machine. The quality of models that are accessible on consumer hardware has improved *dramatically* in the past two years.

My laptop is a 64GB MacBook Pro M2, which I got in January 2023—two months after the initial release of ChatGPT. All of my experiments running LLMs on a laptop have used this same machine.

In March 2023 I wrote that [Large language models are having their Stable Diffusion moment](https://simonwillison.net/2023/Mar/11/llama/) after running Meta’s initial LLaMA release (think of that as Llama 1.0) via the then-brand-new [llama.cpp](https://github.com/ggerganov/llama.cpp). I said:

> As my laptop started to spit out text at me I genuinely had a feeling that the world was about to change

I had a moment of déjà vu the day before yesterday, when I ran Llama 3.3 70B on the same laptop for the first time.

Meta [claim that](https://twitter.com/AIatMeta/status/1865079068833780155):

> This model delivers similar performance to Llama 3.1 405B with cost effective inference that’s feasible to run locally on common developer workstations.

Llama 3.1 405B is their *much* larger best-in-class model, which is very much in the same weight class as GPT-4 and friends.

Everything I’ve seen so far from Llama 3.3 70B suggests that it holds up to that standard. I honestly didn’t think this was possible—I assumed that anything as useful as GPT-4 would require many times more resources than are available to me on my consumer-grade laptop.

I’m so excited by the continual efficiency improvements we’re seeing in running these impressively capable models. In the proprietary hosted world it’s giving us incredibly cheap and fast models like [Gemini 1.5 Flash](https://simonwillison.net/search/?q=gemini+flash&sort=date), [GPT-4o mini](https://simonwillison.net/2024/Jul/18/gpt-4o-mini/) and [Amazon Nova](https://simonwillison.net/2024/Dec/4/amazon-nova/). In the openly licensed world it’s giving us increasingly powerful models we can run directly on our own devices.

- [How I ran Llama 3.3 70B on my machine using Ollama](https://simonwillison.net/2024/Dec/9/llama-33-70b/#how-i-ran-llama-3-3-70b-on-my-machine-using-ollama)
- [Putting the model through its paces](https://simonwillison.net/2024/Dec/9/llama-33-70b/#putting-the-model-through-its-paces)
- [How does it score?](https://simonwillison.net/2024/Dec/9/llama-33-70b/#how-does-it-score-)
- [Honorable mentions](https://simonwillison.net/2024/Dec/9/llama-33-70b/#honorable-mentions)
- [Is performance about to plateau?](https://simonwillison.net/2024/Dec/9/llama-33-70b/#is-performance-about-to-plateau-)
- [Bonus: running Llama 3.3 70B with MLX](https://simonwillison.net/2024/Dec/9/llama-33-70b/#bonus-running-llama-3-3-70b-with-mlx)

#### How I ran Llama 3.3 70B on my machine using Ollama {#how-i-ran-llama-3-3-70b-on-my-machine-using-ollama}

I don’t expect that this model would work well with much *less* than my 64GB of RAM. The first time I tried it consumed every remaining bit of available memory and hard-crashed my Mac! For my second attempt I made sure not to have Firefox and VS Code running at the same time and it worked just fine.

I used [Ollama](https://ollama.com/). Install that, then run this command to fetch the model:

```
ollama pull llama3.3
```

This fetched 42GB of data into my `~/.ollama/models/blobs` directory.

I used [my LLM](https://llm.datasette.io/) tool to run prompts against the model, since that way both the prompt and the response would be automatically [logged to SQLite](https://llm.datasette.io/en/stable/logging.html). First I installed the [llm-ollama](https://github.com/taketwo/llm-ollama) plugin:

```
llm install llm-ollama
```

Then with the Ollama local server running I could run this:

```
llm models
```

And see this in the list:

```
Ollama: llama3.3:latest (aliases: llama3.3)
```

#### Putting the model through its paces {#putting-the-model-through-its-paces}

This means I can run prompts like so:

```
llm -m llama3.3 'Write a letter to the mayor of Half Moon Bay CA advocating for the installation of cosy boxes for Pelicans in the harbor'
```

And get back a response that started like this:

> \[Your Name\]\
>  \[Your Address\]\
>  Half Moon Bay, CA \[Zip Code\]\
>  \[Email Address\]\
>  \[Phone Number\]\
>  \[Date\]
> The Honorable Mayor, Deborah Penrose\
>  City of Half Moon Bay\
>  501 Main Street\
>  Half Moon Bay, CA 94019\
> 
> Dear Mayor Penrose,
> I am writing to propose an innovative and unique initiative that would not only enhance the beauty of our picturesque harbor but also provide a safe haven for one of our beloved coastal residents: the California Brown Pelican. I am excited to advocate for the installation of cozy boxes specifically designed for pelicans in the Half Moon Bay harbor. \[...\]

(Deborah Penrose was the mayor of Half Moon Bay for a single year from December 2016 to December 2017—so a hint of some quite finely grained world knowledge there, even if it’s not relevant for the present day.)

Here’s [the full transcript](https://gist.github.com/simonw/34a6eb5e765928b6491a6a59894b272d), including the response I got when I ran the follow-up `llm -c 'translate that to spanish'`.

Next I tried a simple coding challenge. I love [using Claude Artifacts](https://simonwillison.net/2024/Oct/21/claude-artifacts/) to build simple single page applications, can Llama 3.3 on my laptop do that too?

```
llm -m llama3.3 'produce HTML with embedded CSS and JavaScript - no react, no dependencies - for an app that takes a URL to a YouTube video and outputs images along with visible URLs for all known thumbnails for that video'
```

Here’s the code it produced (the [full transcript is here](https://gist.github.com/simonw/924ba4d75528eccfe4756b5ee37ea7de)):

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>YouTube Thumbnail Extractor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        #thumbnails {
            display: flex;
            flex-wrap: wrap;
        }
        .thumbnail {
            margin: 10px;
        }
        img {
            max-width: 150px;
            max-height: 150px;
            object-fit: cover;
        }
    </style>
</head>
<body>

<h2>YouTube Thumbnail Extractor</h2>
<input type="text" id="youtubeUrl" placeholder="Enter YouTube video URL">
<button onclick="extractThumbnails()">Extract Thumbnails</button>
<div id="thumbnails"></div>

<script>
    function extractThumbnails() {
        const youtubeUrl = document.getElementById('youtubeUrl').value;
        const videoId = getYoutubeVideoId(youtubeUrl);
        
        if (!videoId) {
            alert("Could not parse YouTube video ID from the URL.");
            return;
        }
        
        const thumbnailSizes = ['default', 'mqdefault', 'hqdefault', 'sddefault', 'maxresdefault'];
        const thumbnailsDiv = document.getElementById('thumbnails');
        thumbnailsDiv.innerHTML = ''; // Clear previous results
        
        thumbnailSizes.forEach(size => {
            const thumbnailUrl = `https://img.youtube.com/vi/${videoId}/${size}.jpg`;
            const img = document.createElement('img');
            img.src = thumbnailUrl;
            img.alt = `${size} thumbnail`;
            img.className = 'thumbnail';
            
            const link = document.createElement('a');
            link.href = thumbnailUrl;
            link.text = size;
            link.style.display = 'block';
            
            const div = document.createElement('div');
            div.appendChild(img);
            div.appendChild(link);
            thumbnailsDiv.appendChild(div);
        });
    }
    
    function getYoutubeVideoId(url) {
        const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
        const match = url.match(regExp);
        return (match && match[7].length == 11) ? match[7] : false;
    }
</script>

</body>
</html>
```

Again, we’re illustrating some world knowledge here: it knows that YouTube videos are generally available as `default`, `mqdefault`, `hqdefault`, `sddefault` and `maxresdefault`.

Here’s [the hosted page](https://static.simonwillison.net/static/2024/youtube-thumbnails.html), and this is what it looks like when run against [this YouTube URL](https://www.youtube.com/watch?v=OziYd7xcGzc):

![YouTube Thumbnail EXtractor. The URL has been entered and a Extract Thumbnails button clicked. It shows five thumbnails, for default and mqdefault and hqdefault and sddefault and maxresdefault - they are each the same size with a hyperlink to the full version.](https://static.simonwillison.net/static/2024/youtube-thumbnails.jpg)

It’s not as good as the [version I iterated on with Claude](https://simonwillison.net/2024/Sep/20/youtube-thumbnail-viewer/), but this still shows that Llama 3.3 can one-shot a full interactive application while *running on my Mac*.

#### How does it score? {#how-does-it-score-}

It’s always useful to check independent benchmarks for this kind of model.

One of my current favorites for that is [LiveBench](https://livebench.ai/), which calls itself “a challenging, contamination-free LLM benchmark” and tests a large array of models with a comprehensive set of different tasks.

`llama-3.3-70b-instruct-turbo` currently sits in position 19 on their table, a place ahead of Claude 3 Opus (my favorite model for several months after its release in March 2024) and just behind April’s GPT-4 Turbo and September’s GPT-4o.

![Data table showing AI model performance metrics with column headers for Model, Provider, Global Average plus several other performance categories. Visible entries are: gemini-1.5-pro-exp-0827 (Google, 52.38), meta-llama-3.1-405b-instruct-turbo (Meta, 52.04), gpt-4o-2024-11-20 (OpenAI, 50.64), qwen2.5-72b-instruct-turbo (Alibaba, 50.63), dracarys-72b-instruct (AbacusAI, 50.15), chatgpt-4o-latest-0903 (OpenAI, 50.07), gpt-4-turbo-2024-04-09 (OpenAI, 49.83), llama-3.3-70b-instruct-turbo (Meta, 49.78), and claude-3-opus-20240229 (Anthropic, 48.51).](https://static.simonwillison.net/static/2024/livebench-llama.jpg)

LiveBench here is sorted by the average across multiple evals, and Llama 3.3 70B somehow currently scores top of the table for the “IF” (Instruction Following) eval which likely skews that average. Here’s the [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) paper describing that particular test.

It’s worth noting that the benchmarks listed here run against the full-sized Llama 3.3 release. The versions I’ve been running on my laptop are quantized (Ollama’s is Q4\_K\_M), so they aren’t exactly the same model and likely have different (lower) benchmark scores.

#### Honorable mentions {#honorable-mentions}

Llama 3.3 is currently the model that has impressed me the most that I’ve managed to run on my own hardware, but I’ve had several other positive experiences recently.

Last month [I wrote about Qwen2.5-Coder-32B](https://simonwillison.net/2024/Nov/12/qwen25-coder/), an Apache 2.0 licensed model from Alibaba’s Qwen research team that also gave me impressive results with code.

A couple of weeks ago [I tried another Qwen model, QwQ](https://simonwillison.net/2024/Nov/27/qwq/), which implements a similar chain-of-thought pattern to OpenAI’s o1 series but again runs comfortably on my own device.

Meta’s Llama 3.2 family of models are interesting as well: tiny 1B and 3B models (those should run even on a Raspberry Pi) that are way more capable than I would have expected—plus Meta’s first multi-modal vision models at 11B and 90B sizes. [I wrote about those in September](https://simonwillison.net/2024/Sep/25/llama-32/).

#### Is performance about to plateau? {#is-performance-about-to-plateau-}

I’ve been mostly unconvinced by the ongoing discourse around LLMs hitting a plateau. The areas I’m personally most excited about are multi-modality (images, audio and video as input) and model efficiency. Both of those have had enormous leaps forward in the past year.

I don’t particularly care about “AGI”. I want models that can do useful things that I tell them to, quickly and inexpensively—and that’s exactly what I’ve been getting more of over the past twelve months.

Even if progress on these tools entirely stopped right now, the amount I could get done with just the models I’ve downloaded and stashed on a USB drive would keep me busy and productive for years.

#### Bonus: running Llama 3.3 70B with MLX {#bonus-running-llama-3-3-70b-with-mlx}

I focused on Ollama in this article because it’s the easiest option, but I also managed to run a version of Llama 3.3 using Apple’s excellent [MLX library](https://github.com/ml-explore/mlx), which just celebrated its first birthday.

Here’s how I ran the model with MLX, using [uv](https://github.com/astral-sh/uv) to fire up a temporary virtual environment:

```
uv run --with mlx-lm --python 3.12 python
```

This gave me a Python interpreter with [mlx-lm](https://pypi.org/project/mlx-lm/#description) available. Then I ran this:

```
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Llama-3.3-70B-Instruct-4bit")
```

This downloaded 37G from [mlx-community/Llama-3.3-70B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit) to `~/.cache/huggingface/hub/models--mlx-community--Llama-3.3-70B-Instruct-4bit`.

Then:

```
prompt = "Generate an SVG of a pelican riding a bicycle, start with <svg>"
messages = [{"role": "user", "content": prompt}]
prompt2 = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = generate(
    model, tokenizer, prompt=prompt2, verbose=True
)
```

Which gave me:

```
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
  <!-- Sky -->
  <rect x="0" y="0" width="400" height="300" fill="#87CEEB" rx="0" />
  
  <!-- Ground -->
  <rect x="0" y="250" width="400" height="50" fill="#964B00" rx="0" />
  
  <!-- Pelican -->
  <g transform="translate(150, 150)">
    <!-- Body -->
    <ellipse cx="0" cy="0" rx="30" ry="40" fill="#FFFFFF" stroke="#000000" stroke-width="2" />
    
    <!-- Beak -->
    <path d="M 0 -40 L 20 -60 L 40 -40 Z" fill="#FFFF00" stroke="#000000" stroke-width="2" />
    
    <!-- Legs -->
    <line x1="0" y1="40" x2="-20" y2="80" stroke="#000000" stroke-width="5" />
    <line x1="0" y1="40" x2="20" y2="80" stroke="#000000" stroke-width="5" />
    
    <!-- Eyes -->
    <circle cx="-15" cy="-20" r="5" fill="#000000" />
    <circle cx="15" cy="-20" r="5" fill="#000000" />
  </g>
  
  <!-- Bicycle -->
  <g transform="translate(200, 220) scale(0.7, 0.7)">
    <!-- Frame -->
    <rect x="-50" y="-20" width="100" height="40" fill="#808080" rx="10" />
    
    <!-- Wheels -->
    <circle cx="-35" cy="20" r="30" fill="#808080" />
    <circle cx="35" cy="20" r="30" fill="#808080" />
    
    <!-- Pedals -->
    <rect x="-45" y="40" width="10" height="20" fill="#808080" rx="5" />
    <rect x="35" y="40" width="10" height="20" fill="#808080" rx="5" />
    
    <!-- Seat -->
    <rect x="-20" y="-40" width="40" height="20" fill="#808080" rx="5" />
  </g>
  
  <!-- Pelican on Bicycle -->
  <g transform="translate(200, 150)">
    <!-- Pelican's hands on handlebars -->
    <line x1="-20" y1="0" x2="-40" y2="-20" stroke="#000000" stroke-width="5" />
    <line x1="20" y1="0" x2="40" y2="-20" stroke="#000000" stroke-width="5" />
    
    <!-- Pelican's feet on pedals -->
    <line x1="0" y1="40" x2="-20" y2="60" stroke="#000000" stroke-width="5" />
    <line x1="0" y1="40" x2="20" y2="60" stroke="#000000" stroke-width="5" />
  </g>
</svg>
```

Followed by:

```
Prompt: 52 tokens, 49.196 tokens-per-sec
Generation: 723 tokens, 8.733 tokens-per-sec
Peak memory: 40.042 GB
```

Here’s what that looks like:

Honestly, [I’ve seen worse](https://github.com/simonw/pelican-bicycle?tab=readme-ov-file#pelicans-on-a-bicycle).
