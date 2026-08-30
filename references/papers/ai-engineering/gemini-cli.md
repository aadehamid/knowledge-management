title: Gemini CLI
description: First there was Claude Code in February, then OpenAI Codex \(CLI\) in April, and now Gemini CLI in June. All three of the largest AI labs now have their own …
author: Simon Willison

# Gemini CLI

25th June 2025 - Link Blog

**[Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/)**. First there was [Claude Code](https://simonwillison.net/2025/Feb/24/claude-37-sonnet-and-claude-code/) in February, then [OpenAI Codex (CLI)](https://simonwillison.net/2025/Apr/16/) in April, and now Gemini CLI in June. All three of the largest AI labs now have their own version of what I am calling a "terminal agent" - a CLI tool that can read and write files and execute commands on your behalf in the terminal.

I'm honestly a little surprised at how significant this category has become: I had assumed that terminal tools like this would always be something of a niche interest, but given the number of people I've heard from spending hundreds of dollars a month on Claude Code this niche is clearly larger and more important than I had thought!

I had a few days of early access to the Gemini one. It's very good - it takes advantage of Gemini's million token context and has good taste in things like when to read a file and when to run a command.

Like OpenAI Codex and unlike Claude Code it's open source (Apache 2) - the full source code can be found in [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) on GitHub. The core system prompt [lives in core/src/core/prompts.ts](https://github.com/google-gemini/gemini-cli/blob/0915bf7d677504c28b079693a0fe1c853adc456e/packages/core/src/core/prompts.ts#L40-L109) - I've extracted that out as [a rendered Markdown Gist](https://gist.github.com/simonw/9e5f13665b3112cea00035df7da696c6).

As usual, the system prompt doubles as extremely accurate and concise documentation of what the tool can do! Here's what it has to say about comments, for example:

> - **Comments:** Add code comments sparingly. Focus on *why* something is done, especially for complex logic, rather than *what* is done. Only add high-value comments if necessary for clarity or if requested by the user. Do not edit comments that are seperate from the code you are changing. *NEVER* talk to the user or describe your changes through comments.

The list of preferred technologies is interesting too:

> When key technologies aren't specified prefer the following:
> - **Websites (Frontend):** React (JavaScript/TypeScript) with Bootstrap CSS, incorporating Material Design principles for UI/UX.
> - **Back-End APIs:** Node.js with Express.js (JavaScript/TypeScript) or Python with FastAPI.
> - **Full-stack:** Next.js (React/Node.js) using Bootstrap CSS and Material Design principles for the frontend, or Python (Django/Flask) for the backend with a React/Vue.js frontend styled with Bootstrap CSS and Material Design principles.
> - **CLIs:** Python or Go.
> - **Mobile App:** Compose Multiplatform (Kotlin Multiplatform) or Flutter (Dart) using Material Design libraries and principles, when sharing code between Android and iOS. Jetpack Compose (Kotlin JVM) with Material Design principles or SwiftUI (Swift) for native apps targeted at either Android or iOS, respectively.
> - **3d Games:** HTML/CSS/JavaScript with Three.js.
> - **2d Games:** HTML/CSS/JavaScript.

As far as I can tell Gemini CLI only defines a small selection of tools:

- `edit`: To modify files programmatically.
- `glob`: To find files by pattern.
- `grep`: To search for content within files.
- `ls`: To list directory contents.
- `shell`: To execute a command in the shell
- `memoryTool`: To remember user-specific facts.
- `read-file`: To read a single file
- `write-file`: To write a single file
- `read-many-files`: To read multiple files at once.
- `web-fetch`: To get content from URLs.
- `web-search`: To perform a web search (using [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) via the Gemini API).

I found most of those by having Gemini CLI inspect its own code for me! Here's [that full transcript](https://gist.github.com/simonw/12c7b072e8e21ef1e040fb3b69c1da28), which used just over 300,000 tokens total.

How much does it cost? The announcement describes a generous free tier:

> To use Gemini CLI free-of-charge, simply login with a personal Google account to get a free Gemini Code Assist license. That free license gets you access to Gemini 2.5 Pro and its massive 1 million token context window. To ensure you rarely, if ever, hit a limit during this preview, we offer the industry’s largest allowance: 60 model requests per minute and 1,000 requests per day at no charge.

It's not yet clear to me if your inputs can be used to improve Google's models if you are using the free tier - that's been the situation with free prompt inference they have offered in the past.

You can also drop in your own paid API key, at which point your data will not be used for model improvements and you'll be billed based on your token usage.
