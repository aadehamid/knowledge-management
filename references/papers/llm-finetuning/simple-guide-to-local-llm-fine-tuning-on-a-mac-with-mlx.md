title: A simple guide to local LLM fine-tuning on a Mac with MLX
description: Hello there! Are you looking to fine-tune a large language model \(LLM\) on your Apple silicon Mac? If so, you’re in the right place. Let’s walk through the process of…

# A simple guide to local LLM fine-tuning on a Mac with MLX

January 8, 2024 • 2 min read

Hello there! Are you looking to fine-tune a large language model (LLM) on your Apple silicon Mac? If so, you’re in the right place.

Let’s walk through the process of fine-tuning step-by-step. It won’t cost you a penny because we’re going to do it all on your own hardware using [Apple’s MLX framework](https://github.com/ml-explore/mlx). 

Once we’re done you’ll have a fully fine-tuned LLM you can prompt, all from the comfort of your own device.

I’ve broken this guide down into multiple sections. Each part is self contained, so feel free to skip to the part that’s most relevant to you:

**Let me add a disclaimer here**. Everything in this space moves really fast, so within weeks some of this is going to be out of date! I’m also learning this myself, so I expect to read this back in a few months and feel slightly embarrassed.

That said, what I’m writing looks to be a good approach as of January 2024. I’ll try and update parts when major changes happen or I figure out a better way.

I hope you find this guide helpful! If you have any feedback, questions, or suggestions please drop them [on the Twitter/X thread](https://twitter.com/apeatling/status/1744362343876682072), or on the [Mastodon thread](https://mastodon.social/@apeatling/111720834625871876).

---

---

*P.S. [Happy birthday Matt](https://ma.tt/2024/01/birthday-gift/)! Thanks for the prompt to write a blog post*.
