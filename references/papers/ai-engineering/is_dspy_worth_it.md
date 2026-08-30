title: Maxime Rivest – DSPy sounds neat, but is it reliable?
description: Someone asked: DSPy sounds neat, but is it reliable? Many people think they should not use DSPy if it’s optimization does not work for you in practice, but that’s not true. Here’s why.
author: Maxime Rivest

# DSPy sounds neat, but is it reliable?

Maxime Rivest

2025-09-26

![](https://maximerivest.com/posts/images/worthit.png)

Someone just asked/told me:

> DSPy sounds like a neat idea but is it practical?

TL;DR: yes.

My take on reliability and practicality is that, yes, it’s reliable—but that’s the wrong question to ask. The reason I reach for DSPy is first and foremost because it’s the simplest and most ergonomic way I know to call an LLM from Python. It’s the most lightweight way to send a prompt to an LLM, and it also happens to provide the richest tooling and possibilities: optimization of instructions and few-shots, agents, fine-tuning, structured output with validation and retries, and signatures.

Say you just want to send a string to an LLM and be productive. Without much setup or framework, you can just do this:

```
import dspy
lm = dspy.LM("gemini/gemini-2.5-flash") # api_key="my_api_key"
lm("Count all the R's in strawberry")
```

```
['There are **3** R\'s in "strawberry".']
```

Nothing more, this is extremely reliable and much more ergonomic and productive than any other AI SDK I have tried (special mention to Claudette, which is also very nice).

But then you want to go from prompt to workflow or AI system. You just move to:

```
dspy.configure(lm=lm)
prg = dspy.Predict("text_input, letter -> letter_occurence: int")
prg(text_input="snowboarding is cool", letter="o")
```

```
Prediction(
    letter_occurence=3
)
```

That, in my experience, is as good as writing the prompt myself, but already much more general, and much more productive and ergonomic! But then, say you want to optimize it. You add this:

```
examples = [dspy.Example(
    text_input="snowboarding is cool", letter="o",
    letter_occurence=4

dspy.Example(
    text_input="writing the prompt", letter="i",
    letter_occurence=2

dspy.Example(
    text_input="that is extremely reliable", letter="e",
    letter_occurence=5


# mark input fields
trainset = [i.with_inputs("text_input", "letter") for i in examples]

def is_equal(gold, pred, _=None):
    return gold.letter_occurence == pred.letter_occurence

optimizer = dspy.MIPROv2(metric=is_equal)
prg_opt = optimizer.compile(prg, trainset=trainset)
```

… 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: \=\=\=\=\= Trial 10 / 10 \=\=\=\=\=

Average Metric: 1.00 / 2 (50.0%): 100%|██████████| 2/2 \[00:00\<00:00, 293.70it/s\] 2025/09/26 21:02:37 INFO dspy.evaluate.evaluate: Average Metric: 1 / 2 (50.0%) 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Score: 50.0 with parameters \[‘Predictor 0: Instruction 2’, ‘Predictor 0: Few-Shot Set 5’\]. 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Scores so far: \[50.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 50.0, 100.0, 50.0\] 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Best score so far: 100.0 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: \=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=

2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: \=\=\=\=\= Trial 11 / 10 \=\=\=\=\=

Average Metric: 2.00 / 2 (100.0%): 100%|██████████| 2/2 \[00:00\<00:00, 178.00it/s\] 2025/09/26 21:02:37 INFO dspy.evaluate.evaluate: Average Metric: 2 / 2 (100.0%) 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Score: 100.0 with parameters \[‘Predictor 0: Instruction 1’, ‘Predictor 0: Few-Shot Set 3’\]. 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Scores so far: \[50.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 50.0, 100.0, 50.0, 100.0\] 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Best score so far: 100.0 2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: \=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=\=

2025/09/26 21:02:37 INFO dspy.teleprompt.mipro\_optimizer\_v2: Returning best identified program with score 100.0!

Now we can use the optimized program:

```
prg_opt(text_input="Returning best identified program", letter="e")
```

```
Prediction(
    letter_occurence=4
)
```

That is what DSPy sent to the LLM. For us:

```

  "messages": [

      "role": "system",
      "content": [
        "Your input fields are:",
        "1. `text_input` (str):",
        "2. `letter` (str):",
        "Your output fields are:",
        "1. `letter_occurence` (int):",
        "All interactions will be structured in the following way, with the appropriate values filled in.",
        "",
        "[[ ## text_input ## ]]",
        "{text_input}",
        "",
        "[[ ## letter ## ]]",
        "{letter}",
        "",
        "[[ ## letter_occurence ## ]]",
        "{letter_occurence}        # note: the value you produce must be a single int value",
        "",
        "[[ ## completed ## ]]",
        "In adhering to this structure, your objective is:",
        "Count the occurrences of the specified `letter` within the `text_input` string, and provide this total as `letter_occurence`."



      "role": "user",
      "content": [
        "[[ ## text_input ## ]]",
        "snowboarding is cool",
        "",
        "[[ ## letter ## ]]",
        "o"



      "role": "assistant",
      "content": [
        "[[ ## letter_occurence ## ]]",
        "4",
        "",
        "[[ ## completed ## ]]"



      "role": "user",
      "content": [
        "[[ ## text_input ## ]]",
        "Returning best identified program",
        "",
        "[[ ## letter ## ]]",
        "e",
        "",
        "Respond with the corresponding output fields, starting with the field `[[ ## letter_occurence ## ]]` (must be formatted as a valid Python int), and then ending with the marker for `[[ ## completed ## ]]`."



      "role": "assistant",
      "content": [
        "[[ ## letter_occurence ## ]]",
        "4",
        "",
        "[[ ## completed ## ]]"



```

The final instruction was:

> In adhering to this structure, your objective is: Count the occurrences of the specified `letter` within the `text_input` string, and provide this total as `letter_occurence`.

And the selected few-shot example was this one:

```
...{"role": "user", 
  "content": [
         "[[ ## text_input ## ]] snowboarding is cool
           [[ ## letter ## ]] o"

    "role": "assistant",      
    "content": [
         "[[ ## letter_occurence ## ]] 4
          [[ ## completed ## ]]"       
```

As you can see, dspy can be used gradually and is a delight at each step of the way :)
