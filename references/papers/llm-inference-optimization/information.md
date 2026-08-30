✨ New Course! Enroll in [Building Adaptive AI Agents](https://bit.ly/3SWAsIy)

[![DeepLearning.AI](/_next/image?url=%2Fdlai%2Fassets%2Fdlai-logo.png&w=640&q=75&dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](/)

* [Courses](/courses)
* News

  + [The Batch](/the-batch)
  + [Andrew's Letter](/the-batch/tag/letters)
  + [Data Points](/the-batch/tag/data-points)
  + [ML Research](/the-batch/tag/research)
  + [Blog](/blog)
* Community

  + [Forum](https://community.deeplearning.ai/)
  + [Events](/events)
  + [Ambassadors](/ambassador)
  + [Ambassador Spotlight](/blog/category/ambassador-spotlight)
  + [Resources](/resources)
* [Membership](/membership)
* For Business

  + [Overview](/business)
  + [Plans & Pricing](/business/plans-pricing)
  + [Learning Tracks](/business/learningtracks)
  + [Contact Us](/business/contact-us)
* + [Membership](/membership)
  + For Business
    - [Overview](/business)
    - [Plans & Pricing](/business/plans-pricing)
    - [Learning Tracks](/business/learningtracks)
    - [Contact Us](/business/contact-us)

* [Courses](/courses)
* News
  + [The Batch](/the-batch)
  + [Andrew's Letter](/the-batch/tag/letters)
  + [Data Points](/the-batch/tag/data-points)
  + [ML Research](/the-batch/tag/research)
  + [Blog](/blog)
* Community
  + [Forum](https://community.deeplearning.ai/)
  + [Events](/events)
  + [Ambassadors](/ambassador)
  + [Ambassador Spotlight](/blog/category/ambassador-spotlight)
  + [Resources](/resources)
* [Membership](/membership)
* For Business
  + [Overview](/business)
  + [Plans & Pricing](/business/plans-pricing)
  + [Learning Tracks](/business/learningtracks)
  + [Contact Us](/business/contact-us)

* [Overview](#overview)
* [Course Outline](#course-outline)
* [Instructors](#instructors)

![](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2024%2F10%2FYour-paragraph-text.png&w=1080&q=75)

1. [All Courses](/courses)
3. [Short Course](/courses?types=short_course)
5. Efficient Inference with SGLang: Text and Image Generation

1. [All Courses](/courses)
3. [Short Course](/courses?types=short_course)
5. Efficient Inference with SGLang: Text and Image Generation

Short CourseIntermediate1h19m

# Efficient Inference with SGLang: Text and Image Generation

Instructor: Richard Chen

[![RadixArk logo](https://home-wordpress.deeplearning.ai/wp-content/uploads/2026/04/Radixark-logo.svg)](https://www.radixark.ai/)[![SGLang logo](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2Fsglang-logo.png&w=384&q=75)](https://www.lmsys.org/)

Earn an accomplishment with [PRO](https://learn.deeplearning.ai/membership)

[Enroll Now](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation?utm_source=home&utm_medium=course-landing-page&utm_campaign=summary-cta-button)

![](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FYouTube-Thumbnails-2026-03-31T104417.455.png&w=3840&q=75)

* Intermediate
* 1h19m
* 7 Video Lessons
* 3 Code Examples
* 1 Graded Assignment PRO
* Earn an accomplishment with PRO
* Instructor: Richard Chen
* ![RadixArk](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FRadixark-square.jpg&w=48&q=75)RadixArk![SGLang](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FLMSys-logo-square.jpg&w=48&q=75)SGLang
* Learn more about[Membership PRO Plan](https://learn.deeplearning.ai/membership)

[Enroll Now](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation?utm_source=home&utm_medium=course-landing-page&utm_campaign=summary-cta-button)

## What you'll learn

* Understand how LLM inference works token by token, why it gets expensive at scale, and how the KV cache eliminates redundant computation by storing and reusing intermediate values.
* Implement SGLang’s RadixAttention to extend caching across users and requests, and measure the real speedups it delivers.
* Apply SGLang’s caching and parallelism strategies to diffusion models, accelerating image generation using the same principles as text.

## About this course

Introducing **Efficient Inference with SGLang: Text and Image Generation**, built in partnership with LMSys and RadixArk, and taught by Richard Chen a Member of Technical Staff at RadixArk.

Running LLMs in production is expensive. Much of that cost comes from redundant computation: every new request forces the model to reprocess the same system prompt and shared context from scratch. SGLang is an open-source inference framework that eliminates that waste by caching computation that’s already been done and reusing it across future requests.

In this course, you’ll build a clear mental model of how inference works (from input tokens to generated output) and learn why the memory bottleneck exists. From there, you’ll implement the KV cache from scratch to store and reuse intermediate attention values within a single request. Then you’ll go further with RadixAttention, SGLang’s approach to sharing KV cache across requests by identifying common prefixes using a radix tree. Finally, you’ll apply these same optimization principles to image generation using diffusion models.

**In detail, you’ll:**

* Build a mental model of LLM inference: how a model processes input tokens, generates output token by token, and where the computational cost accumulates.
* Implement the attention mechanism from scratch and build a KV cache to store and reuse intermediate key-value tensors, cutting redundant computation within a single request.
* Extend caching across requests using SGLang’s RadixAttention, which uses a radix tree to identify shared prefixes across users and skip repeated processing.
* Apply SGLang’s caching strategies to diffusion models for faster image generation, and explore multi-GPU parallelism for further acceleration.
* Survey where the inference field is heading, including emerging techniques and how the optimization principles from this course apply to future developments.

By the end, you’ll have hands-on experience with the caching strategies powering today’s most efficient AI systems and the tools to implement these optimizations in your own models at scale.

## Who should join?

Developers and ML practitioners who want to better understand and optimize LLM inference in production. Familiarity with Python and basic language model concepts is recommended.

## Course Outline

7 Lessons・3 Code Examples

* [Introduction

  Video

  ・

  3m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/fwapdx/introduction)
* [Overview of Inference

  Video

  ・

  10m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/uzfnvt/overview-of-inference)
* [LLM Inference Fundamentals

  Video with Code Example

  ・

  11m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/rm6zga/llm-inference-fundamentals)
* [Advanced LLM Inference Optimization

  Video with Code Example

  ・

  18m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/napkiw/advanced-llm-inference-optimization)
* [SGLang Diffusion

  Video with Code Example

  ・

  19m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/de1bn1/sglang-diffusion)
* [The future of inference– where do we go from here?

  Video

  ・

  6m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/qjr4xu/the-future-of-inference%E2%80%93-where-do-we-go-from-here%3F)
* [Conclusion

  Video

  ・

  1m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/0kjrst/conclusion)
* [Quiz

  Graded・Quiz

  ・

  10m](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation/lesson/de1bnb/quiz)

![Unlock certificates](/_next/image?url=%2Fdlai%2Fassets%2Funlock-certificate.png&w=3840&q=75&dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)![Unlock certificates](/_next/image?url=%2Fdlai%2Fassets%2Funlock-certificate-narrow.png&w=3840&q=75&dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)

#### Elevate your learning experience with Pro

Upgrade to Pro and gain unlimited accomplishments on your resume

[Learn More](https://learn.deeplearning.ai/membership)

## Instructor

![Richard Chen](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FInstructors-profile-picture-97.png&w=256&q=75)

### Richard Chen

Member of Technical Staff, [RadixArk](https://www.radixark.ai/)

## Efficient Inference with SGLang: Text and Image Generation

* Intermediate
* 1h19m
* 7 Video Lessons
* 3 Code Examples
* 1 Graded Assignment PRO
* Earn an accomplishment with PRO
* Instructor: Richard Chen
* ![RadixArk](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FRadixark-square.jpg&w=48&q=75)RadixArk![SGLang](/_next/image?url=https%3A%2F%2Fhome-wordpress.deeplearning.ai%2Fwp-content%2Fuploads%2F2026%2F04%2FLMSys-logo-square.jpg&w=48&q=75)SGLang
* Learn more about[Membership PRO Plan](https://learn.deeplearning.ai/membership)

[Enroll Now](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation?utm_source=home&utm_medium=course-landing-page&utm_campaign=summary-cta-button)

![](/dlai/assets/course-information/hero-blocks.svg?dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)

Additional learning features, such as quizzes and projects, are included with DeepLearning.AI Pro. Explore it today

[Enroll Now](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation?utm_source=home&utm_medium=course-landing-page&utm_campaign=summary-cta-button)

## Want to learn more about Generative AI?

Keep learning with updates on curated AI news, courses, and events, as well as Andrew’s thoughts from DeepLearning.AI!

[Enroll Now](https://learn.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation?utm_source=home&utm_medium=course-landing-page&utm_campaign=summary-cta-button)

[![DeepLearning.AI](/_next/image?url=%2Fdlai%2Fassets%2Fdlai-logo.png&w=3840&q=75&dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](/)

* [Courses](/courses)

* [The Batch](/the-batch)
* [Community](/community)

* [Careers](/careers)
* [About](/about)
* [Contact](/contact)

* [Help](https://info.deeplearning.ai/knowledge-base)

Get the mobile app

[![](/dlai/assets/home/badge-app-store.svg?dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](https://apps.apple.com/us/app/deeplearning-ai/id6761054329)[![](/dlai/assets/home/badge-google-play.svg?dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](https://play.google.com/store/apps/details?id=ai.deeplearning.apprn)

[![](/dlai/assets/home/badge-app-store.svg?dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](https://dlai.onelink.me/kfDj/footer)[![](/dlai/assets/home/badge-google-play.svg?dpl=dpl_6N1duVh95NbxAdbPZGHRUuNykCCi)](https://dlai.onelink.me/kfDj/footer)

[Terms of Use](/terms-of-use)[Privacy Policy](/privacy)

## Choose Your Plan

Planning for more users?

Team Plan

[Learn More](/membership)Keep Current Plan

close