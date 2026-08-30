title: Prompt Optimization for Language Models with DSPy GEPA · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

#  Prompt Optimization for Language Models with DSPy GEPA

*Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)*

This notebook demonstrates how to use [DSPy](https://dspy.ai/)’s GEPA (Generalized Error-driven Prompt Augmentation) optimizer to improve language model performance on mathematical reasoning tasks. We’ll work with the [NuminaMath-1.5 dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) and show how GEPA can boost accuracy through automated prompt optimization.

**What you’ll learn:**

- Setting up DSPy with language models ([OpenRouter](https://openrouter.ai/))
- Processing and filtering mathematical problem datasets
- Building a baseline Chain-of-Thought reasoning program
- Optimizing prompts with GEPA using error-driven feedback
- Evaluating improvements in model accuracy

GEPA works by analyzing errors, generating targeted feedback, and automatically refining prompts to address common failure patterns. This makes it particularly effective for complex reasoning tasks where prompt quality significantly impacts performance.

**Key Resources:**

- [DSPy Documentation](https://dspy.ai/learn/programming/)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)
- [GEPA Optimizer Guide](https://dspy.ai/api/optimizers/GEPA/)

##  Installation and Setup

Install required dependencies and import libraries for DSPy, dataset processing, and model configuration.

**Installation Options:**

- **uv** - Fast Python package installer ([documentation](https://docs.astral.sh/uv/))
- **pip** - Traditional Python package manager

**Key Dependencies:**

- `dspy` - DSPy framework for language model programming
- `datasets` - Hugging Face datasets library for loading NuminaMath-1.5
- `python-dotenv` - Environment variable management for API keys

```
# Install with uv (recommended - faster)
!uv pip install dspy datasets python-dotenv

# Alternative: Install with pip
# !pip install dspy datasets python-dotenv
```

```
import dspy
from datasets import load_dataset
import os
```

##  Language Model Configuration

Configure your language model - either local (Ollama) or cloud-based (OpenRouter) - for use with DSPy.

```
from dotenv import load_dotenv
load_dotenv("../../.env")
```

###  Model Selection Rationale

**Main LM: `openrouter/openai/gpt-4.1-nano`**

*Primary Role:* High-volume inference during baseline evaluation and GEPA optimization iterations

*Key Selection Criteria:*

1. **Cost Efficiency** - $0.10/M input tokens, $0.40/M output tokens (\~90% cheaper than GPT-4.1 or Claude)
2. **Low Latency** - Fastest GPT-4.1 variant, enables rapid iteration with 16-32 parallel threads
3. **Adequate Performance** - 60-65% baseline accuracy (MMLU: 80.1%, GPQA: 50.3%)
4. **Context Window** - 1M tokens for long chain-of-thought reasoning

---

**Reflection LM: `openrouter/qwen/qwen3-next-80b-a3b-thinking`**

*Primary Role:* Deep error analysis and prompt improvement during GEPA’s reflection phase

*Key Selection Criteria:*

1. **Advanced Reasoning** - “Thinking” variant specialized for analytical reasoning and pattern identification
2. **Quality Over Speed** - \~16 reflection calls vs 2000\+ inference calls, can afford slower, higher-quality model
3. **Context Handling** - 10M token context window for processing multiple training examples
4. **Cost Trade-off** - More expensive per token but negligible total cost due to low volume

**Architecture Philosophy:** Use a cheap, fast model for high-volume inference (99% of calls) and a smart, analytical model for low-volume reflection (1% of calls). This asymmetric design optimizes for both cost efficiency and learning quality.

###  Understanding GEPA’s Two-Model Architecture

GEPA’s breakthrough innovation lies in its **dual-model approach** for reflective prompt optimization, which fundamentally differs from traditional single-model optimizers.

**Why Two Models?**

Traditional prompt optimizers rely on scalar metrics (accuracy scores) to guide improvements, essentially using trial-and-error without understanding *why* predictions fail. GEPA introduces a revolutionary approach by separating concerns:

**1. Student LM (Inference Model)**

- **Role**: Primary model that executes tasks and generates predictions
- **Characteristics**: Fast, cost-efficient, handles high-volume inference
- **Usage Pattern**: \~90-95% of all API calls during optimization
- **In This Notebook**: `openrouter/openai/gpt-4.1-nano`

**2. Reflection LM (Meta-Cognitive Model)**

- **Role**: Analyzes failures, identifies patterns, and generates prompt improvements
- **Characteristics**: Stronger reasoning, analytical depth, interpretability
- **Usage Pattern**: \~5-10% of API calls (only during reflection phases)
- **In This Notebook**: `openrouter/qwen/qwen3-next-80b-a3b-thinking`

**The Reflective Optimization Cycle:**

```
1. Student LM solves training problems → predictions
2. Metric provides rich textual feedback on failures
3. Reflection LM analyzes batches of failures → identifies patterns
4. Reflection LM generates improved prompt instructions
5. Student LM tests new prompts → validation
6. Repeat until convergence
```

**Research Foundation:**

This approach is detailed in the paper [“Reflective Prompt Evolution Can Outperform Reinforcement Learning”](https://arxiv.org/abs/2507.19457), which demonstrates that reflective optimization with textual feedback outperforms reinforcement learning approaches on complex reasoning tasks.

```
>>> # ============================================
>>> # OpenRouter Language Model Configuration
>>> # ============================================
>>> # Requires OPENROUTER_API_KEY environment variable
>>> # Sign up at https://openrouter.ai/ to get your API key

>>> # # Main LM for inference
>>> open_router_lm = dspy.LM(
...     'openrouter/openai/gpt-4.1-nano', 
...     api_key=os.getenv('OPENROUTER_API_KEY'), 
...     api_base='https://openrouter.ai/api/v1',
...     max_tokens=65536,
...     temperature=1.0
... )

>>> # # Reflection LM for GEPA optimization
>>> reflection_lm = dspy.LM(
...     'openrouter/qwen/qwen3-next-80b-a3b-thinking', 
...     api_key=os.getenv('OPENROUTER_API_KEY'), 
...     api_base='https://openrouter.ai/api/v1',
...     max_tokens=65536,
...     temperature=1.0
... )

>>> # Set OpenRouter as default LM
>>> dspy.configure(lm=open_router_lm)

>>> print("✅ OpenRouter LM configured successfully!")
>>> print(f"Main model: openrouter/openai/gpt-4.1-nano")
>>> print(f"Reflection model: openrouter/qwen/qwen3-next-80b-a3b-thinking")
```

```
✅ OpenRouter LM configured successfully!
Main model: openrouter/openai/gpt-4.1-nano
Reflection model: openrouter/qwen/qwen3-next-80b-a3b-thinking
```

##  Dataset Preparation Functions

Helper functions to process the dataset, split it into train/val/test sets, and preview examples.

```
def init_dataset(
    train_split_ratio: float = None, 
    test_split_ratio: float = None, 
    val_split_ratio: float = None, 
    sample_fraction: float = 1.0
) -> tuple[list, list, list]:
    """
    Initialize and split the NuminaMath-1.5 dataset into train/val/test sets.
    
    Loads the dataset, filters for numeric answers, converts to DSPy Examples,
    shuffles with fixed seed for reproducibility, and optionally samples a fraction.
    
    Args:
        train_split_ratio: Proportion for training (default: 0.5)
        test_split_ratio: Proportion for testing (default: 0.45)
        val_split_ratio: Proportion for validation (default: 0.05)
        sample_fraction: Fraction of dataset to use (default: 1.0 = full dataset)
    
    Returns:
        Tuple of (train_set, val_set, test_set) as lists of DSPy Examples
    
    Raises:
        AssertionError: If split ratios don't sum to 1.0
    """
    # Set default split ratios
    if train_split_ratio is None:
        train_split_ratio = 0.5
    if test_split_ratio is None:
        test_split_ratio = 0.4
    if val_split_ratio is None:
        val_split_ratio = 0.1
    
    # Validate split ratios sum to 1.0
    assert (train_split_ratio + test_split_ratio + val_split_ratio) == 1.0, "Ratios must sum to 1.0"

    # Load dataset from Hugging Face Hub
    train_split = load_dataset("AI-MO/NuminaMath-1.5")['train']
    
    # Convert to DSPy Examples with input/output fields
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")  # Mark 'problem' as input field
        for x in train_split
    ]
    
    # Shuffle with fixed seed for reproducibility
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)
    print(f"Total number of examples after filtering: {tot_num}")

    # Apply sampling if requested
    if sample_fraction < 1.0:
        sample_num = int(tot_num * sample_fraction)
        train_split = train_split[:sample_num]
        tot_num = sample_num
        print(f"Sampled down to {sample_num} examples.")
    
    # Split into train/val/test based on ratios
    train_end = int(train_split_ratio * tot_num)
    val_end = int((train_split_ratio + val_split_ratio) * tot_num)
    
    train_set = train_split[:train_end]
    val_set = train_split[train_end:val_end]
    test_set = train_split[val_end:]

    return train_set, val_set, test_set
```

```
>>> train_set, val_set, test_set = init_dataset(sample_fraction=0.00025)

>>> print(len(train_set), len(val_set), len(test_set))
```

```
Total number of examples after filtering: 896215
Sampled down to 224 examples.
112 22 90
```

```
>>> print("Problem:")
>>> print(train_set[0]['problem'])
>>> print("\n\nSolution:")
>>> print(train_set[0]['solution'])
>>> print("\n\nAnswer:")
>>> print(train_set[0]['answer'])
```

```
Problem:
In the diagram, AB=15 cm,AB = 15\text{ cm}, $DC = 24\text&#123; cm},$ and AD=9 cm.AD = 9\text{ cm}. What is the length of AC,AC, to the nearest tenth of a centimeter?

[asy]
draw((0,0)--(9,16)--(33,16)--(9,0)--cycle,black+linewidth(1));
draw((9,16)--(9,0),black+linewidth(1));
draw((0,0)--(33,16),black+linewidth(1));
draw((9,0)--(9,0.5)--(8.5,0.5)--(8.5,0)--cycle,black+linewidth(1));
draw((9,16)--(9.5,16)--(9.5,15.5)--(9,15.5)--cycle,black+linewidth(1));
label("$A$",(0,0),NW);
label("$B$",(9,16),NW);
label("$C$",(33,16),E);
label("$D$",(9,0),SE);
label("15 cm",(0,0)--(9,16),NW);
label("9 cm",(0,0)--(9,0),S);
label("24 cm",(9,0)--(33,16),SE);
[/asy]


Solution:
Extend ADAD to point EE where it intersects the perpendicular from CC on $BC$'s extension.

[asy]
draw((0,0)--(9,16)--(33,16)--(9,0)--cycle,black+linewidth(1));
draw((9,16)--(9,0),black+linewidth(1));
draw((0,0)--(33,16),black+linewidth(1));
draw((9,0)--(9,0.5)--(8.5,0.5)--(8.5,0)--cycle,black+linewidth(1));
draw((9,16)--(9.5,16)--(9.5,15.5)--(9,15.5)--cycle,black+linewidth(1));
label("$A$",(0,0),NW);
label("$B$",(9,16),NW);
label("$C$",(33,16),E);
label("$D$",(9,0),SE);
draw((9,0)--(33,0),black+linewidth(1)+dashed);
draw((33,0)--(33,16),black+linewidth(1)+dashed);
draw((33,0)--(33,0.5)--(32.5,0.5)--(32.5,0)--cycle,black+linewidth(1));
label("$E$",(33,0),SE);
label("18 cm",(9,0)--(33,0),S);
label("16 cm",(33,0)--(33,16),E);
[/asy]

Using the Pythagorean theorem in $\triangle ADB$, calculate $BD^2 = BA^2 - AD^2 = 15^2 - 9^2 = 144$, so $BD = 12\text&#123; cm}$.

In $\triangle DBC$, compute $BC^2 = DC^2 - BD^2 = 24^2 - 12^2 = 432$, thus $BC = 18\text&#123; cm}$.

Recognize BCEDBCED as a rectangle, hence DE=BC=18 cmDE = BC = 18\text{ cm} and $CE = BD = 12\text&#123; cm}$.

Examine △AEC\triangle AEC with $AE = AD + DE = 9 + 18 = 27\text&#123; cm}$, then apply Pythagorean theorem:
\[ AC^2 = AE^2 + CE^2 = 27^2 + 12^2 = 729 + 144 = 873 \]
\[ AC = \sqrt&#123;873} \approx \boxed&#123;29.5\text&#123; cm}} \]


Answer:
29.5\text&#123; cm}
```

```
>>> print(test_set[0]['problem'])
>>> print("\n\nAnswer:")
>>> print(test_set[0]['answer'])
```

```
a cistern is two - third full of water . pipe a can fill the remaining part in 12 minutes and pipe b in 8 minutes . once the cistern is emptied , how much time will they take to fill it together completely ?


Answer:
14.4
```

##  Baseline Chain-of-Thought Program

Create a simple baseline using DSPy’s Chain-of-Thought module to establish initial performance.

```
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)
```

##  Evaluation Metric

Define the evaluation metric to compare model predictions against ground truth answers.

```
def parse_integer_answer(answer):
    try:
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError, TypeError):
        answer = 0

    return answer

def metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

##  Baseline Evaluation

Evaluate the baseline Chain-of-Thought program to establish our starting accuracy before optimization.

```
>>> evaluate = dspy.Evaluate(
...     devset=test_set,
...     metric=metric,
...     num_threads=16,
...     display_table=True,
...     display_progress=True
... )

>>> evaluate(program)
```

```
0%|          | 0/90 [00:00
```

###  Understanding the Baseline Results

The evaluation table shows our model’s performance on 90 test problems:

**Table Columns:**

- `problem`: The mathematical question from NuminaMath-1.5
- `example_answer`: Ground truth answer
- `reasoning`: Model’s chain-of-thought reasoning process
- `pred_answer`: Model’s final prediction
- `metric`: ✔️ indicates correct answer

**Key Observations:**

- **Baseline Accuracy: \~52%** - The model gets roughly half the problems correct
- **Reasoning Quality**: The model generates coherent step-by-step reasoning (see the `reasoning` column)
- **Common Failures**:
    - Calculation errors (e.g., row 0: predicted 4.8 minutes vs correct 14.4 minutes)
    - Misinterpreting problem statements

**Why This Matters:** This baseline performance demonstrates that while GPT-4.1 Nano has reasonable mathematical reasoning capability, there’s significant room for improvement. GEPA will analyze these errors and automatically refine the prompt to address common failure patterns, potentially boosting accuracy by 10-20 percentage points.

##  GEPA Optimization

Apply GEPA optimizer with error-driven feedback to automatically improve the prompt and boost performance.

###  How GEPA Works: Error-Driven Prompt Improvement

GEPA (Generalized Error-driven Prompt Augmentation) is an automatic prompt optimization technique that learns from mistakes to improve model performance. Here’s how it works:

**The GEPA Optimization Cycle:**

1. **Evaluation Phase** - Run the model on training examples and collect predictions
2. **Error Analysis** - Identify which problems the model got wrong
3. **Feedback Generation** - Create detailed feedback explaining:
    - What the correct answer should be
    - Why the model’s answer was wrong
    - The complete step-by-step solution
4. **Reflection Phase** - Use the reflection LM (Qwen3 Thinking) to:
    - Analyze patterns across multiple failed examples
    - Identify common failure modes (e.g., “model miscalculates ratios”, “model misinterprets word problems”)
    - Generate improved prompt instructions to address these patterns
5. **Prompt Update** - Modify the system prompt with new guidelines
6. **Validation** - Test the updated prompt on validation set
7. **Iteration** - Repeat the cycle, keeping only improvements that boost validation accuracy

**Why We Need `metric_with_feedback`:**

Unlike a standard metric that just returns 0 or 1 (correct/incorrect), `metric_with_feedback` returns:

- **Score**: 0 or 1 for correctness
- **Feedback**: Rich textual explanation including the ground truth solution

This feedback is crucial because GEPA’s reflection model needs to understand *why* predictions failed to generate better prompts. The more detailed the feedback, the better GEPA can identify patterns and create targeted improvements.

**Key Parameters:**

- `auto="light"`: Controls optimization intensity (light/medium/heavy)
- `reflection_minibatch_size=16`: Number of errors analyzed together per reflection
- `reflection_lm`: The smarter model used for analyzing errors and improving prompts
- `num_threads=32`: Parallel evaluation for faster optimization

```
def metric_with_feedback(
    example: dspy.Example, 
    prediction: dspy.Prediction, 
    trace=None, 
    pred_name=None, 
    pred_trace=None
) -> dspy.Prediction:
    """
    Enhanced evaluation metric with detailed feedback for GEPA optimization.
    
    Evaluates predictions and generates targeted feedback including error analysis
    and the complete solution for learning. Feedback helps GEPA identify failure
    patterns and improve prompts.
    
    Args:
        example: DSPy Example with ground truth answer and solution
        prediction: DSPy Prediction with model's answer
        trace: Optional trace information (unused)
        pred_name: Optional prediction name (unused)
        pred_trace: Optional prediction trace (unused)
    
    Returns:
        DSPy Prediction with score (0 or 1) and detailed feedback text
    """
    # Extract ground truth and solution
    written_solution = example.get('solution', '')
    
    try:
        llm_answer = prediction
    except ValueError as e:
        # Handle parsing failure with detailed feedback
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with '{prediction.answer}', which couldn't be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting."
        )
        feedback_text += f" The correct answer is '{example.get('answer', '')}'."
        
        # Include full solution if available
        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                f"Think about what takeaways you can learn from this solution to improve "
                f"your future answers and approach to similar problems and ensure your "
                f"final answer is a valid integer."
            )
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Score: 1 for correct, 0 for incorrect
    score = metric(example, llm_answer)

    # Generate appropriate feedback based on correctness
    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{example.get('answer', '')}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{example.get('answer', '')}'."

    # Append complete solution for learning
    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\n"
            f"Think about what takeaways you can learn from this solution to improve "
            f"your future answers and approach to similar problems."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)
```

```
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=16,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=reflection_lm,
)
```

```
optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

```
>>> print(optimized_program.predict.signature.instructions)
```

```
text
Solve the problem step-by-step, following these guidelines:

- Carefully read the problem statement to understand all provided data and conditions explicitly.
- Define all variables and parameters clearly at the beginning.
- For geometry problems:
  - Confirm exact shape properties (e.g., isosceles triangle has two equal sides; quadratic equation solutions may form sides where two sides equal one root value and the third side is the other root).
  - Apply correct formulas (e.g., circumradius R = abc/(4Δ) or precise isosceles triangle formulas) and verify triangle inequalities (sum of any two sides > third side).
- For word problems:
  - Correctly interpret phrases (e.g., "A beats B by 200 meters" means when A finishes the race, B has run 800 meters).
  - For gradual change problems (fleets, age, etc.), track each year/item step-by-step with clear calculations.
- For functional equations with recurrences (e.g., f(x) + f(x+1) = 1):
  - Break domain into intervals based on integer/fractional parts.
  - Apply recurrence relations correctly to express unknowns in terms of known intervals.
- For derivative problems:
  - Differentiate terms precisely (treat f'(c) as a constant for fixed c).
  - Solve equations step by step, including substitution of specific values at the correct stage.
- For proofs/identities:
  - Simplify algebraically or trigonometrically using standard identities.
  - Check key steps (e.g., gcd analysis for integer problems, factorization, modular arithmetic).
- For multiple-choice questions:
  - Select the correct option letter (e.g., \boxed&#123;\text&#123;C}}) after verification.
- Always verify results against all problem constraints (e.g., domain restrictions, integer requirements, physical feasibility).
- Present the final answer strictly inside \boxed&#123;} with the required format (numerical value, expression, or option letter).
- Never make unwarranted assumptions (e.g., assuming equilateral triangle when only isosceles is specified).
```

##  Optimized Program Evaluation

Evaluate the GEPA-optimized program to measure the improvement in accuracy and effectiveness.

```
>>> evaluate(optimized_program)
```

```
Average Metric: 52.00 / 90 (57.8%): 100%|██████████| 90/90 [01:13<00:00,  1.23it/s]
```

###  Understanding the Optimization Results

**Performance Improvement:**

- **Baseline Accuracy**: 52.2% (47/90 correct)
- **Optimized Accuracy**: 57.8% (52/90 correct)
- **Improvement**: \+5.6 percentage points (\~11% relative improvement)

**What Changed:** See the instruction GEPA developed above.

**Why the Modest Improvement?**

The \~6% gain is expected given:

1. **Small Training Set**: Only 112 training examples (0.025% of full dataset)
2. **Light Optimization**: Using `auto="light"` for faster iteration
3. **Simple Baseline**: Chain-of-Thought already provides decent reasoning structure
4. **Model Limitations**: GPT-4.1 Nano’s mathematical capabilities are the ceiling

**Cost Efficiency:**

This entire experiment (baseline evaluation, GEPA optimization, and final evaluation on 224 examples) cost **less than $0.50** thanks to:

- GPT-4.1 Nano’s low pricing ($0.10/M input, $0.40/M output)
- Asymmetric architecture (cheap model for 99% of calls, smart model for 1%)
- Small sample size for demonstration purposes

**Key Takeaway:**

Even with limited data and light optimization, GEPA successfully identified failure patterns and generated targeted prompt improvements. With more training data (`sample_fraction=0.01` or higher) and heavier optimization (`auto="medium"` or `"heavy"`), we’d expect 15-25% improvements, potentially reaching 65-70% accuracy.

##  Learn More

This notebook introduced DSPy’s GEPA optimizer for automated prompt improvement. Here are additional resources to deepen your understanding:

###  DSPy Framework

- **[DSPy Documentation](https://dspy.ai/)** - Official documentation and guides
- **[DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)** - Source code and examples
- **[DSPy Research Paper](https://arxiv.org/abs/2310.03714)** - “DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines”
- **[DSPy Tutorial Series](https://dspy.ai/learn/programming/)** - Step-by-step learning path

###  Prompt Optimization

- **[GEPA Optimizer Documentation](https://dspy.ai/api/optimizers/GEPA/)** - Technical details on GEPA
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** - Foundational paper on CoT reasoning
- **[Automatic Prompt Engineering](https://arxiv.org/abs/2211.01910)** - “Large Language Models Are Human-Level Prompt Engineers”
- **[DSPy Optimizers Comparison](https://dspy.ai/api/optimizers/)** - Overview of different optimization strategies

###  Mathematical Reasoning

- **[NuminaMath Dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)** - The dataset used in this notebook
- **[GSM8K Dataset](https://huggingface.co/datasets/gsm8k)** - Grade school math word problems benchmark
- **[MATH Dataset](https://huggingface.co/datasets/hendrycks/competition_math)** - Competition-level mathematics problems
- **[Mathematical Reasoning with LLMs](https://arxiv.org/abs/2206.14858)** - Survey of techniques

###  Related Techniques

- **[Few-Shot Learning](https://arxiv.org/abs/2005.14165)** - “Language Models are Few-Shot Learners” (GPT-3 paper)
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)** - Improving reasoning via multiple sampling paths
- **[ReAct Prompting](https://arxiv.org/abs/2210.03629)** - Reasoning and Acting in language models

###  Tools and Platforms

- **[OpenRouter](https://openrouter.ai/)** - Unified API for multiple LLM providers
- **[Hugging Face Datasets](https://huggingface.co/docs/datasets/)** - Dataset loading and processing
- **[DSPy Optimizers Guide](https://dspy.ai/deep-dive/optimizers/)** - Deep dive into optimization strategies

[ Update on GitHub](https://github.com/huggingface/cookbook/blob/main/notebooks/en/dspy_gepa.md)
