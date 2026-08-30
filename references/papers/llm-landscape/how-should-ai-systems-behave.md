title: How should AI systems behave, and who should decide?
description: We’re clarifying how ChatGPT’s behavior is shaped and our plans for improving that behavior, allowing more user customization, and getting more public input into our decision-making in these areas.

# How should AI systems behave, and who should decide?

We’re clarifying how ChatGPT’s behavior is shaped and our plans for improving that behavior, allowing more user customization, and getting more public input into our decision-making in these areas.

![How Should AI Systems Behave](https://images.ctfassets.net/kftzwdyauwt9/b9ae2cb3-b7df-4636-a7faacfc627a/b851f1ad4ae087b5c45d646c1784b87b/how-should-ai-systems-behave.png?w=3840&q=90&fm=webp)

Illustration: Justin Jay Wang

OpenAI’s [mission⁠](https://openai.com/charter/)  is to ensure that artificial general intelligence (AGI) ^^ benefits all of humanity. We therefore think a lot about the behavior of AI systems we build in the run-up to AGI, and the way in which that behavior is determined. Since our launch of [ChatGPT⁠](https://openai.com/index/chatgpt/) , users have shared outputs that they consider politically biased, offensive, or otherwise objectionable. In many cases, we think that the concerns raised have been valid and have uncovered real limitations of our systems which we want to address. We’ve also seen a few misconceptions about how our systems and policies work together to shape the outputs you get from ChatGPT.

Below, we summarize:

- How ChatGPT’s behavior is shaped;
- How we plan to improve ChatGPT’s default behavior;
- Our intent to allow more system customization; and
- Our efforts to get more public input on our decision-making.

Unlike ordinary software, our models are massive neural networks. Their behaviors are learned from a broad range of data, not programmed explicitly. Though not a perfect analogy, the process is more similar to training a dog than to ordinary programming. An initial “pre-training” phase comes first, in which the model learns to predict the next word in a sentence, informed by its exposure to lots of Internet text (and to a vast array of perspectives). This is followed by a second phase in which we “fine-tune” our models to narrow down system behavior.

As of today, this process is imperfect. Sometimes the fine-tuning process falls short of our intent (producing a safe and useful tool) and the user’s intent (getting a helpful output in response to a given input). Improving our methods for aligning AI systems with human values is a top [priority⁠](https://openai.com/alignment/)  for our company, particularly as AI systems become more capable.

The two main steps involved in building ChatGPT work as follows:

![Building ChatGPT diagram](https://images.ctfassets.net/kftzwdyauwt9/3f67394d-34ab-4ffe-3d0b425ab328/60b0c22f045d78d0b2190263b581293d/building-chatgpt.svg?w=3840&q=90){width=660 height=480}

First, we “ **pre-train** ” models by having them predict what comes next in a big dataset that contains parts of the Internet. They might learn to complete the sentence “instead of turning left, she turned \_\_\_.” By learning from billions of sentences, our models learn grammar, many facts about the world, and some reasoning abilities. They also learn some of the biases present in those billions of sentences.

Then, we “ **fine-tune** ” these models on a more narrow dataset that we carefully generate with human reviewers who follow guidelines that we provide them. Since we cannot predict all the possible inputs that future users may put into our system, we do not write detailed instructions for every input that ChatGPT will encounter. Instead, we outline a few categories in the guidelines that our reviewers use to review and rate possible model outputs for a range of example inputs. Then, while they are in use, the models generalize from this reviewer feedback in order to respond to a wide array of specific inputs provided by a given user.

In some cases, we may give guidance to our reviewers on a certain kind of output (for example, “do not complete requests for illegal content”). In other cases, the guidance we share with reviewers is more high-level (for example, “avoid taking a position on controversial topics”). Importantly, our collaboration with reviewers is not one-and-done—it’s an ongoing relationship, in which we learn a lot from their expertise.

A large part of the fine-tuning process is maintaining a strong feedback loop with our reviewers, which involves weekly meetings to address questions they may have, or provide clarifications on our guidance. This iterative feedback process is how we train the model to be better and better over time.

Many are rightly worried about biases in the design and impact of AI systems. We are committed to robustly addressing this issue and being transparent about both our intentions and our progress. Towards that end, we are sharing a [portion of our guidelines⁠](https://cdn.openai.com/snapshot-of-chatgpt-model-behavior-guidelines.pdf)  that pertain to political and controversial topics. Our guidelines are explicit that reviewers should not favor any political group. Biases that nevertheless may emerge from the process described above are bugs, not features.

While disagreements will always exist, we hope sharing this blog post and these instructions will give more insight into how we view this critical aspect of such a foundational technology. It’s our belief that technology companies must be accountable for producing policies that stand up to scrutiny.

We’re always working to improve the clarity of these guidelines—and based on what we’ve learned from the ChatGPT launch so far, we’re going to provide clearer instructions to reviewers about potential pitfalls and challenges tied to bias, as well as controversial figures and themes. Additionally, as part of ongoing transparency initiatives, we are working to share aggregated demographic information about our reviewers in a way that doesn’t violate privacy rules and norms, since this is an additional source of potential bias in system outputs.

We are currently researching how to make the [fine-tuning process⁠](https://openai.com/index/instruction-following/)  more understandable and controllable, and are building on external advances such as [rule based rewards⁠](https://arxiv.org/abs/2209.14375)  and [Constitutional AI⁠](https://arxiv.org/abs/2212.08073) .

In pursuit of our mission, we’re committed to ensuring that access to, benefits from, and influence over AI and AGI are widespread. We believe there are at least three building blocks required in order to achieve these goals in the context of AI system behavior. ^^

**1. Improve default behavior** . We want as many users as possible to find our AI systems useful to them “out of the box” and to feel that our technology understands and respects their values.

Towards that end, we are investing in research and engineering to reduce both glaring and subtle biases in how ChatGPT responds to different inputs. In some cases ChatGPT currently refuses outputs that it shouldn’t, and in some cases, it doesn’t refuse when it should. We believe that improvement in both respects is possible.

Additionally, we have room for improvement in other dimensions of system behavior such as the system “making things up.” Feedback from users is invaluable for making these improvements.

**2. Define your AI’s values, within broad bounds** . We believe that AI should be a useful tool for individual people, and thus customizable by each user up to limits defined by society. Therefore, we are developing an upgrade to ChatGPT to allow users to easily customize its behavior.

This will mean allowing system outputs that other people (ourselves included) may strongly disagree with. Striking the right balance here will be challenging–taking customization to the extreme would risk enabling [malicious uses⁠](https://openai.com/index/forecasting-misuse/)  of our technology and sycophantic AIs that mindlessly amplify people’s existing beliefs.

There will therefore always be some bounds on system behavior. The challenge is defining what those bounds are. If we try to make all of these determinations on our own, or if we try to develop a single, monolithic AI system, we will be failing in the commitment we make in our Charter to “avoid undue concentration of power.”

**3. Public input on defaults and hard bounds** . One way to avoid undue concentration of power is to give people who use or are affected by systems like ChatGPT the ability to influence those systems’ rules.

We believe that many decisions about our defaults and hard bounds should be made collectively, and while practical implementation is a challenge, we aim to include as many perspectives as possible. As a starting point, we’ve sought external input on our technology in the form of [red teaming⁠](https://github.com/openai/dalle-2-preview/blob/main/system-card.md) . We also recently began [soliciting public input⁠](https://platform.openai.com/docs/chatgpt-education/educator-input)  on AI in education (one particularly important context in which our technology is being deployed).

We are in the early stages of piloting efforts to solicit public input on topics like system behavior, disclosure mechanisms (such as watermarking), and our deployment policies more broadly. We are also exploring partnerships with external organizations to conduct third-party audits of our safety and policy efforts.

Combining the three building blocks above gives the following picture of where we’re headed:

![Diagram of where we’re headed building ChatGPT](https://images.ctfassets.net/kftzwdyauwt9/65b50211-85c0-4f5f-1620b6da4303/8a139c6b87eb50d8abd819782948d9d9/building-chatgpt-future.svg?w=3840&q=90){width=660 height=650}

Sometimes we will make mistakes. When we do, we will learn from them and [iterate⁠](https://openai.com/index/language-model-safety-and-misuse/)  on our models and systems.

We appreciate the ChatGPT user community as well as the wider public’s vigilance in holding us accountable, and are excited to share more about our work in the three areas above in the coming months.

*If you are interested in doing research to help achieve this vision, including but not limited to research on fairness and representation, alignment, and sociotechnical research to understand the impact of AI on society, please apply for subsidized access to our API via the* [*Researcher Access Program*⁠](https://share.hsforms.com/1b-BEAq_qQpKcfFGKwwuhxA4sk30) *.*

*We are also* [*hiring*⁠](https://openai.com/careers/#open) *for positions across Research, Alignment, Engineering, and more.*

## Author
