<table style="width:100%">
<tr>
<td style="vertical-align:middle; text-align:left;">
<font size="2">
Supplementary code for the <a href="https://mng.bz/lZ5B">Build a Reasoning Model (From Scratch)</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
<br>Code repository: <a href="https://github.com/rasbt/reasoning-from-scratch">https://github.com/rasbt/reasoning-from-scratch</a>
</font>
</td>
<td style="vertical-align:middle; text-align:left;">
<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover-small.webp" width="100px"></a>
</td>
</tr>
</table>


# Chapter 3: Evaluating Reasoning Models

```python
from importlib.metadata import version

used_libraries = [
    "reasoning_from_scratch",
    "torch",
    "sympy",
    "tokenizers"  # Used by reasoning_from_scratch
]

for lib in used_libraries:
    print(f"{lib} version: {version(lib)}")
```

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F01_raschka.webp?1" width="500px">

&nbsp;
## 3.1 Building a math verifier

- No code in this section

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F02_raschka.webp?1" width="500px">

<br>
<br>
<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F03_raschka.webp?1" width="500px">

&nbsp;
## 3.2 Loading a pre-trained model to generate text

- In this section, we load the model (recap of chapter 2) that we want to evaluate

```python
from pathlib import Path
import torch


from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)


def load_model_and_tokenizer(
    which_model, device, use_compile, local_dir="qwen3"
):
    if which_model == "base":

        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":

        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer
```

- Note that we use the base model here; once you have completed this chapter, you can rerun the notebook after changing `which_model="base"` to `which_model="reasoning"` to evaluate an already trained reasoning model

```python
from reasoning_from_scratch.ch02 import (
    get_device
)

WHICH_MODEL = "base"
device = get_device()

# If you have compatibility issues, try to
# uncomment the line below and rerun the notebook
# device = torch.device("cpu")

model, tokenizer = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=device,
    use_compile=False
)
```

```python
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)

prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)

# Similar to chapter 2 exercise solution:
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
    ).unsqueeze(0)

all_token_ids = []
for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0)
    decoded_id = tokenizer.decode(token_id.tolist())
    print(
        decoded_id,
        end="",
        flush=True
    )
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids)
```

- If you are unfamiliar with LaTeX syntax, the response above can be very hard to read
- You can use the `Latex` class to render the LaTeX syntax to improve readability, as shown below

```python
from IPython.display import Latex, display

display(Latex(all_tokens))
```

- If you only want to render specific math expressions, you can also use the `Math` class:

```python
from IPython.display import Math

display(Math(r"\dfrac{14}{3}"))
```

&nbsp;
## 3.3 Implementing a wrapper for easier text generation

- Above, we loaded the pre-trained LLM and set up the text generation functionality (as illustrated in the figure below), which are the first two steps of the evaluation process covered in the remainder of this chapter

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F05_raschka.webp?1" width="500px">

- For additional convenience, we create a wrapper function for the text generation function so that we only have to pass in the model, tokenizer, and prompt, along with some additional settings

```python
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


skip_portion = False

if not skip_portion:
    generated_text = generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=2048,
        verbose=True
    )
```

&nbsp;
## 3.4 Extracting the final answer box

- In this section, we extract the answer box (step 3); later, in the next section will take the extracted answer and normalize it (step 4)

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F06_raschka.webp?1" width="500px">

```python
model_answer = (
r"""... some explanation...
**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
""")
```

```python
def get_last_boxed(text):
    # Find the last occurrence of "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Get position after "\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip any whitespace after "\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Expect an opening brace "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse the braces with nesting
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Account for unbalanced braces
    if brace_depth != 0:
        return None

    # Extract content inside the outermost braces
    return text[content_start_idx:current_idx-1]
```

```python
extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)
```

```python
import re

RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def extract_final_candidate(text, fallback="number_then_full"):
    # Default return value if nothing matches
    result = ""

    if text:
        # Prefer the last boxed expression if present
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # Use last number
                result = m[-1]
            elif fallback == "number_then_full":
                # Else return full text if no number found
                result = text
    return result
```

- fallback settings if no boxed content is found:
    - "number_then_full": pick the last simple number, else the whole text
    - "number_only": pick the last simple number, else return an empty string `""`
    - "none": extract only boxed content, else return empty string `""`

```python
print(extract_final_candidate(model_answer))
```

```python
print(extract_final_candidate(r"\boxed{ 14/3. }"))
```

```python
print(extract_final_candidate("abc < > 14/3 abc"))
```

```python
print(extract_final_candidate("Text without numbers"))
```

&nbsp;
## 3.5 Normalizing the extracted answer

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F07_raschka.webp?1" width="500px">

- In the previous section, we extracted the answer (step 3), now we are normalizing it (step 4 in the previous figure)

```python
LATEX_FIXES = [  # Latex formatting to be replaced
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # strip chat special tokens like <|assistant|>

def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
    SUPERSCRIPT_MAP = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
    }

    # Strip leading multiple-choice labels
    # E.g., like "c. 3" -> 3, or "b: 2" -> 2
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
        
    # Remove angle-degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)   # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)             # ^\circ
    text = text.replace("°", "")                        # Unicode degree

    # unwrap \text{...} if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # strip inline/display math wrappers \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # light LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    # convert unicode superscripts into exponent form (e.g., 2² -> 2**2)m
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)
    
    # numbers/roots
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # fractions
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # exponent and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()
```

```python
print(normalize_text(extract_final_candidate(model_answer)))
```

```python
print(normalize_text(r"$\dfrac{14}{3.}$"))
```

```python
print(normalize_text(r"\text{\[\frac{14}{3}\]}"))
```

```python
print(normalize_text("4/3"))
```

&nbsp;
## 3.6 Verifying mathematical equivalence

- In this section, we implement the basic functionality to check if the extracted answer (generated by the model) is equivalent to the correct answer (ground truth) provided in the dataset; this is step 5
- In the next section (step 6), we make this process a bit more robust to grade the answer correctness

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F08_raschka.webp?1" width="500px">

```python
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from tokenize import TokenError
from sympy.polys.polyerrors import PolynomialError

def sympy_parser(expr):
    # To avoid crashing on long garbage responses 
    # that some badly trained models (chapter 6) may emit
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Standard transformations like handling parentheses
                *spp.standard_transformations,

                # Allow omitted multiplication symbols (e.g., "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),

            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None
```

- Note that this appears to be an excessive amount of error handling, but these are all errors that I encountered when evaluating the model on all 500 MATH-500 problems as the model does not always generate perfectly formatted outputs

```python
print(sympy_parser(normalize_text(
    extract_final_candidate(model_answer)
)))
```

```python
print(sympy_parser("28/6"))
```

```python
from sympy import simplify

def equality_check(expr_gtruth, expr_pred):
    # First, check if the two expressions are exactly the same string
    if expr_gtruth == expr_pred:
        return True

    # Parse both expressions into SymPy objects (returns None if parsing fails)
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # If both expressions were parsed successfully, try symbolic comparison
    if gtruth is not None and pred is not None:
        try:
            # If the difference is 0, they are equivalent
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False
```

```python
print(equality_check(
    normalize_text("13/4."),
    normalize_text(r"(13)/(4)")
))
```

```python
print(equality_check(
    normalize_text("0.5"),
    normalize_text(r"(1)/(2)")
))
```

```python
print(equality_check(
    normalize_text("14/3"),
    normalize_text("15/3")
))
```

```python
print(equality_check(
    normalize_text("(14/3, 2/3)"),
    normalize_text("(14/3, 4/6)")
))
```

&nbsp;
## 3.7 Grading answers

```python
def split_into_parts(text):
    result = [text]

    if text:
        # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # If text is empty, return an empty list
        result = []

    return result
```

```python
split_into_parts(normalize_text(r"(14/3, 2/3)"))
```

```python
def grade_answer(pred_text, gt_text):
    result = False  # Default outcome if checks fail

    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # Break ground truth into comparable parts

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # Break prediction into comparable parts

        # Ensure both sides have same number of valid parts
        if (gt_parts and pred_parts
           and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  # Check each part for mathematical equivalence

    return result  # True only if all checks passed
```

```python
grade_answer("14/3", r"\frac{14}{3}")
```

```python
grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)")
```

```python
# Define test cases: (name, prediction, ground truth, expected result)
tests = [
        ("check_1", "3/4", r"\frac{3}{4}", True),
        ("check_2", "(3)/(4)", r"3/4", True),
        ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
        ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
        ("check_5", "(1, 2)", r"(1,2)", True),
        ("check_6", "(2, 1)", "(1, 2)", False),
        ("check_7", "(1, 2, 3)", "(1, 2)", False),
        ("check_8", "0.5", "1/2", True),
        ("check_9", "0.3333333333", "1/3", False),
        ("check_10", "1,234/2", "617", True),
        ("check_11", r"\text{2/3}", "2/3", True),
        ("check_12", "50%", "1/2", False),
        ("check_13", r"2\cdot 3/4", "3/2", True),
        ("check_14", r"90^\circ", "90", True),
        ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
        ("check_16", r"2²", "2**2", True),
    ]


def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  # Run equality check
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows
    
    # Compute max width for each column to align table nicely
    col_widths = [
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    # Print table row by row
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    # Print summary of passed tests
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")
```

```python
run_demos_table(tests)
```

&nbsp;
## 3.8 Loading the evaluation dataset

- The previous section implemented the basic evaluation pipeline
- In this section, we load the dataset (step 7) to which we will apply this pipeline in order to evaluate the model (step 8, next section).

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F09_raschka.webp?1" width="500px">

- The dataset was downloaded and prepared via the following code from the [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) repository, which requires the [`datasets`](https://huggingface.co/docs/datasets/en/index) package depencency (you don't need to execute this, it's only included for reference):

```python
from datasets import load_dataset
import json

dset = load_dataset("HuggingFaceH4/MATH-500", split="test")

math_data = dset.to_list()
with open("math500_test.json", "w", encoding="utf-8") as f:
    json.dump(math_data, f, ensure_ascii=False, indent=2)
```

```python
import json
import requests

def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:  # Saves a local copy
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data

math_data = load_math500_test()
print("Number of entries:", len(math_data))
```

```python
from pprint import pprint
pprint(math_data[0])
```

&nbsp;
## 3.9 Evaluating the model

- In the previous section, we loaded the dataset; now we can apply the evaluation pipeline to evaluate the model on this dataset (step 7)

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F10_raschka.webp?1" width="500px">

```python
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

```python
prompt = (  # Same prompt we used at the beginning of the chapter
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)
```

```python
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt_fmt, device,
    max_new_tokens=2048,
    verbose=True
)
```

```python
# Below is an alternative prompt template
# which swaps "Question" with "Problem"

"""
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Solve the problem and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Problem:\n{prompt}\n\nAnswer:"
    )
    return template
"""

# This can noticeably affect the MATH-500 results:
# Base model on mps: improves accuracy 20% -> 40%
# Reasoning model on mps: worsens accuracy 90% -> 60%
```

```python
# Alternatively, we may use no prompt template

"""
def render_prompt(prompt):
    return prompt
"""

# This can noticeably affect the MATH-500 results:
# Base model on mps: improves accuracy 20% -> 70%
# Reasoning model on mps: worsens accuracy 90% -> 50%
```

```python
def mini_eval_demo(model, tokenizer, device):
    ex = {  # Test example with "problem" and "answer" fields
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     # 1. Apply prompt template
    gen_text = generate_text_stream_concat(   # 2. Generate response
        model, tokenizer, prompt, device,
        max_new_tokens=64,
    )
    pred_answer = extract_final_candidate(gen_text)  # 3. Extract and normalize answer
    is_correct = grade_answer(                       # 4. Grade answer
        pred_answer, ex["answer"]
    )
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")
```

```python
mini_eval_demo(model, tokenizer, device)
```

```python
import time


# Helper function to calculate remaining time
def eta_progress_message(
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)

def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")  # Make filename compatible with Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:  # Save results for inspection
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])    # 1. Apply prompt template
            gen_text = generate_text_stream_concat(   # 2. Generate response
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

            extracted = extract_final_candidate(  # 3. Extract and normalize answer
                gen_text
            )
            is_correct = grade_answer(            # 4. Grade answer
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {  # Record to be saved for inspection
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # Print responses during the generation process
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc
```

- We only evaluate on 10 examples for demo purposes (to keep the runtime reasonable)

```python
print("Model:", WHICH_MODEL)
print("Device:", device)
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device, 
    math_data=math_data[:10],
    max_new_tokens=2048,
    verbose=False
)
```

| Mode      | Device | Accuracy | MATH-500 size | Time                  |
|-----------|--------|----------|---------------|-----------------------|
| Base      | CPU    | 30%      | 10            | 0.7 min (Mac Mini M4) |
| Base      | MPS    | 20%      | 10            | 0.4 min (Mac Mini M4) |
| Base      | CUDA   | 30%      | 10            | 0.2 min (DGX Spark)   |
| Base      | XPU    | 30%      | 10            | 1.2 min (Intel)       |
| Reasoning | CPU    | 90%      | 10            | 9.5 min (Mac Mini M4) |
| Reasoning | MPS    | 80%      | 10            | 3.8 min (Mac Mini M4) |
| Reasoning | CUDA   | 90%      | 10            | 3.7 min (DGX Spark)   |
| Reasoning | XPU    | 70%      | 10            | 8.5 min (Intel)       |


| Mode      | Device | Accuracy | MATH-500 size    | Time                   |
|-----------|--------|----------|------------------|------------------------|
| Base      | CUDA   | 15.6%    | 500              | 10.0 min (DGX Spark)   |
| Reasoning | CUDA   | 50.8%    | 500              | 182.2 min (DGX Spark)  |

- Note that these values were obtained in PyTorch 2.8 and can differ in different versions of PyTorch

- For reference, above are the different accuracy values 
- Note that "GPU" here refers to a NVIDIA ("cuda") GPU; MPS refers to an Apple Silicon M4 chip
- The reasoning model is much slower because it produces much longer responses
- While Qwen3-Base is a pre-trained base model and the Qwen3 recommends using it without chat template, changing `tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)` to `tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path, apply_chat_template=True)` boosts the MATH-500 performance substantially (80%); note that it is not clear whether the MATH-500 test set was part of the training data; in the age of LLMs, we can assume that any data available on the internet has been part of the training data (also see the discussion [here](https://github.com/rasbt/LLMs-from-scratch/pull/828#issuecomment-3324829736))
- The run for the 500 MATH-500 examples corresponds to changing the code here in the `evaluate_math500_stream` function call from `math_data=math_data[:10],` to `math_data=math_data,`
- The bonus materials contain a script to run the evaluation batched mode for higher throughput (see [../02_math500-verifier-scripts/README.md](../02_math500-verifier-scripts/README.md); on an H100, with a batch size of 128, the base model can be evaluated in 3.3 min, and the reasoning model can be evaluated in 14.6  min

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch03/CH03_F11_raschka.webp?1" width="500px">

- For convenience, you can use the [../02_math500-verifier-scripts/evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py) script, which runs the MATH-500 evaluation code as a standalone script from the command line (see the [../02_math500-verifier-scripts/README.md](../02_math500-verifier-scripts/README.md) for more usage information)
- The [../02_math500-verifier-scripts/evaluate_math500_batched.py](../02_math500-verifier-scripts/evaluate_math500_batched.py) script runs the code in this chapter in batched mode
  - This means it processes multiple examples per forward pass to accelerate the evaluation while requiring more RAM
  - With a batch size of 128, this reduces the runtime of the base model, when evaluating all 500 samples, from 13.3 min to 3.3 min on an H100 GPU
  - Similarly, it reduces the runtime of the reasoning model from 185.4 min to 14.6 min for the 500 examples in the dataset
  - Note that the H100 is used as an example, and the script is compatible with other GPUs (or CPUs) as well

&nbsp;
## Summary

- No code in this section
