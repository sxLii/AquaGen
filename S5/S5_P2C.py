"""
S5_P2C.py — Paper-to-Code Pipeline (4-stage, deepseek-reasoner)

Adapted from S5_Paper2Code, consolidated into a single script.

Workflow:
  Stage 1 : Planning  (4 sub-steps, multi-turn trajectory)
    1a. Overall implementation plan
    1b. Architecture design  (file list, class/interface diagram)
    1c. Logic design  (task list + dependency analysis)
    1d. Config generation  (config.yaml with hyperparams from paper)
  Stage 2 : Per-file Logic Analysis
  Stage 3 : Per-file Code Generation  (accumulated file context)
  Stage 4 : Debugging / Review  (SEARCH/REPLACE patches)
"""

import os
import re
import json
import copy
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
PAPER_MD   = "S5/data/Paper.md"
NET_FILE   = "S5/data/Ji.inp"
OUTPUT_DIR = "S5/results_P2C"
REPO_DIR   = "S5/results_P2C/repo"
MODEL      = "deepseek-reasoner"

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ## Replace with your actual API key
    base_url="https://api.deepseek.com/v1",
)

# ── Read inputs ────────────────────────────────────────────────────────────────
with open(PAPER_MD, "r", encoding="utf-8") as f:
    paper_text = f.read()

with open(NET_FILE, "r", encoding="utf-8") as f:
    net_content = f.read()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPO_DIR, exist_ok=True)
artifacts_dir = os.path.join(OUTPUT_DIR, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions  (adapted from utils.py)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CTX = 131072   # deepseek-reasoner context window

def api_call(messages, max_tokens=16000):
    """
    Call deepseek-reasoner; strip <think>…</think> from the response.
    On 400 context-overflow, parse the reported prompt token count from the
    error message and retry with the remaining budget.
    """
    from openai import BadRequestError
    import re as _re

    def _clean(resp_content):
        if "</think>" in resp_content:
            resp_content = resp_content.split("</think>")[-1].strip()
        return resp_content

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
        return _clean(resp.choices[0].message.content or "")

    except BadRequestError as e:
        msg = str(e)
        # Extract prompt token count from error message, e.g.
        # "... you requested 134920 tokens (70920 in the messages, 64000 in the completion)"
        m = _re.search(r'\((\d+) in the messages', msg)
        if m:
            prompt_tokens = int(m.group(1))
            safe_max = max(MODEL_CTX - prompt_tokens - 512, 1024)
            print(f"  [warn] Context overflow — prompt={prompt_tokens} tokens. "
                  f"Retrying with max_tokens={safe_max}.")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=safe_max,
                stream=False,
            )
            return _clean(resp.choices[0].message.content or "")
        raise


def content_to_json(data: str) -> dict:
    """
    Parse the [CONTENT]…[/CONTENT] JSON blocks produced by the planning stage.
    Falls back through several cleaning passes before giving up.
    Adapted from utils.py content_to_json / content_to_json2 / content_to_json3.
    """
    def _try_parse(text):
        clean = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', text).strip()
        clean = re.sub(r'(".*?"),\s*#.*', r'\1,', clean)     # trailing comments
        clean = re.sub(r'(".*?")\s*#.*',  r'\1',  clean)
        clean = re.sub(r',\s*\]', ']', clean)                 # trailing commas
        clean = re.sub(r'\n\s*', '', clean)
        return clean

    for attempt in range(3):
        try:
            text = _try_parse(data)
            if attempt == 2:                                   # last resort
                text = re.sub(r'"""', '"', text)
                text = re.sub(r"'''", "'", text)
                text = re.sub(r"\\", "'",  text)
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Fallback: extract Logic Analysis + Task list via regex
    m = re.search(
        r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])',
        data,
    )
    if m:
        return {
            "Logic Analysis": json.loads(m.group(1)),
            "Task list":      json.loads(m.group(2)),
        }
    return {}


def extract_code_from_content(content: str) -> str:
    """Extract the first fenced code block (any language). Returns '' if none."""
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if matches:
        return matches[0]
    # Second attempt: python-specific
    m = re.search(r'```python\s*(.*?)```', content, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_yaml_from_content(content: str) -> str:
    """Extract yaml fenced block."""
    m = re.search(r'```yaml\s*\n(.*?)\n```', content, re.DOTALL)
    return m.group(1) if m else ""


def read_python_files(directory: str) -> dict[str, str]:
    """Recursively read all .py files; keys are paths relative to directory."""
    result = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, directory)
                with open(abs_path, "r", encoding="utf-8") as f:
                    result[rel_path] = f.read()
    return result


def parse_and_apply_changes(responses: list[str], repo_dir: str, save_num: int = 1):
    """
    Apply SEARCH/REPLACE edits from LLM responses to files in repo_dir.
    Format expected:
        Filename: some_file.py
        <<<<<<< SEARCH
        original code
        =======
        corrected code
        >>>>>>> REPLACE
    Adapted from 4_debugging.py.
    """
    sr_pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"

    for response in responses:
        file_blocks = re.split(r"Filename:\s*([^\n]+)", response)
        if len(file_blocks) < 3:
            print(f"  [debug] No Filename: patterns found – skipping this response block.")
            continue

        for i in range(1, len(file_blocks), 2):
            filename   = file_blocks[i].strip()
            diff_block = file_blocks[i + 1]
            filepath   = os.path.join(repo_dir, filename)

            matches = re.findall(sr_pattern, diff_block, re.DOTALL)
            if not matches:
                print(f"  [debug] No SEARCH/REPLACE found for {filename}")
                continue
            if not os.path.exists(filepath):
                print(f"  [warn] File not found: {filepath}")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()

            modified = False
            for idx, (search_text, replace_text) in enumerate(matches, 1):
                search_text  = search_text.strip()
                replace_text = replace_text.strip()
                if search_text in file_content:
                    file_content = file_content.replace(search_text, replace_text)
                    modified = True
                    print(f"  ✓ {filename}: patch {idx} applied")
                else:
                    print(f"  ✗ {filename}: patch {idx} – search text not found")

            if modified:
                backup = f"{filepath}.{save_num:03d}.bak"
                os.rename(filepath, backup)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(file_content)
                print(f"  saved {filename}  (backup → {backup})")


def format_json_data(data: dict) -> str:
    """Pretty-print a dict as sectioned text (from utils.py format_json_data)."""
    out = ""
    for key, value in data.items():
        out += "-" * 40 + "\n"
        out += f"[{key}]\n"
        if isinstance(value, list):
            for item in value:
                out += f"- {item}\n"
        else:
            out += str(value) + "\n"
        out += "\n"
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Planning  (multi-turn trajectory, from 1_planning.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Stage 1: Planning")
print("=" * 60)

PLAN_SYSTEM = """\
You are an expert researcher and strategic planner with a deep understanding of
experimental design and reproducibility in scientific research.
You will receive a research paper in Markdown format.
Your task is to create a detailed and efficient plan to reproduce the experiments
and methodologies described in the paper.

Instructions:
1. Align with the Paper: Your plan must strictly follow the methods, datasets,
   model configurations, hyperparameters, and experimental setups described.
2. Be Clear and Structured: Present the plan in a well-organized, easy-to-follow
   format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical
   implementation while ensuring fidelity to the original experiments.\
"""

plan_user_msg = {
    "role": "user",
    "content": f"""## Paper
{paper_text}

## Network / Data File
{net_content}

## Task
1. We want to reproduce the method described in the attached paper.
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan covering:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements,
     experimental settings, hyperparameters, and evaluation metrics.
4. The plan should be as **detailed and informative** as possible.

## Data
The SWMM network file (Ji.inp / Ji.txt) above is the **sole data source**.
Parse it directly to extract network topology, node/link attributes, and any
parameters needed for simulation or model training.
**Do NOT plan to generate synthetic data** — all training, validation, and test
data must be derived from this file (e.g. by running SWMM simulations or
extracting structured features from the .inp sections).

## Requirements
- Do not provide actual code yet; focus on a thorough, clear strategy.
- If something is unclear from the paper, mention it explicitly.

## Instruction
Provide a strong roadmap that will make it easier to write the code later.""",
}

file_list_user_msg = {
    "role": "user",
    "content": """Your goal is to design a concise, usable, and complete software system for
reproducing the paper's method. Use PyTorch as the primary deep learning
framework. Use other open-source libraries (numpy, scipy, matplotlib, pyyaml,
etc.) only for non-neural-network tasks. Keep the overall architecture simple.

Data constraint: the SWMM network file (Ji.inp) is the only data source.
The dataset module must parse this file and derive all required inputs from it.
Do NOT include any synthetic data generation in the design.

Based on the plan above, design the software system now.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will ...",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "\\nclassDiagram\\n    class Main {\\n        +run()\\n    }\\n ...",
    "Program call flow": "\\nsequenceDiagram\\n    participant M as Main\\n ...",
    "Anything UNCLEAR": "Need clarification on ..."
}
[/CONTENT]

## Nodes
- Implementation approach: str  — Summarize the chosen solution strategy.
- File list: List[str]  — Relative paths only. ALWAYS include main.py.
- Data structures and interfaces: Optional[str]  — Mermaid classDiagram with
  all classes, methods (including __init__), type annotations, and relationships.
  Be very detailed; the API should be comprehensive.
- Program call flow: Optional[str]  — Mermaid sequenceDiagram, complete and
  detailed, covering CRUD and init of each object.
- Anything UNCLEAR: str  — Mention ambiguities and ask for clarifications.

## Constraint
Output wrapped inside [CONTENT][/CONTENT] exactly like the format example.

## Action
Generate the output now.""",
}

task_list_user_msg = {
    "role": "user",
    "content": """Break down the tasks according to the design above, generate a task list,
and analyze task dependencies.

-----

## Format Example
[CONTENT]
{
    "Required packages": ["numpy==1.21.0", "torch==1.9.0"],
    "Required Other language third-party packages": ["No third-party dependencies required"],
    "Logic Analysis": [
        ["dataset.py",    "DatasetLoader class — loads and preprocesses data ..."],
        ["model.py",      "Defines the model architecture ..."],
        ["trainer.py",    "Trainer class — training loop ..."],
        ["evaluation.py", "Evaluation class — metrics ..."],
        ["main.py",       "Entry point — orchestrates the pipeline ..."]
    ],
    "Task list": ["dataset.py", "model.py", "trainer.py", "evaluation.py", "main.py"],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both dataset.py and trainer.py share ...",
    "Anything UNCLEAR": "Clarification needed on ..."
}
[/CONTENT]

## Nodes
- Required packages: Optional[List[str]]  — Third-party packages in requirements.txt format.
- Required Other language third-party packages: List[str]
- Logic Analysis: List[List[str]]  — [filename, detailed description] pairs,
  including dependency analysis and imports. Be as detailed as possible.
- Task list: List[str]  — Filenames ordered by dependency (implement dependencies first).
- Full API spec: str  — OpenAPI 3.0 spec (leave blank if no front/back communication).
- Shared Knowledge: str  — Common utilities or config variables.
- Anything UNCLEAR: str

## Constraint
Output wrapped inside [CONTENT][/CONTENT] exactly like the format example.

## Action
Generate the output now.""",
}

config_user_msg = {
    "role": "user",
    "content": """Based on the paper, plan, and design specified above, generate a config.yaml.
Extract all training details from the paper (learning rate, batch size, epochs, etc.).
DO NOT FABRICATE DETAILS — only use values the paper provides.

You must write `config.yaml`.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
""",
}

# Build multi-turn trajectory (from 1_planning.py loop logic)
trajectories = [{"role": "system", "content": PLAN_SYSTEM}]
planning_responses = []

stage_msgs = [
    ("1a. Overall plan",       plan_user_msg),
    ("1b. Architecture design", file_list_user_msg),
    ("1c. Logic design",        task_list_user_msg),
    ("1d. Config generation",   config_user_msg),
]

for stage_name, user_msg in stage_msgs:
    print(f"\n  [{stage_name}]")
    trajectories.append(user_msg)
    content = api_call(trajectories, max_tokens=16000)
    planning_responses.append(content)
    trajectories.append({"role": "assistant", "content": content})
    print(f"    → {len(content.splitlines())} lines")

# Save planning trajectories
traj_path = os.path.join(OUTPUT_DIR, "planning_trajectories.json")
with open(traj_path, "w", encoding="utf-8") as f:
    json.dump(trajectories, f, ensure_ascii=False, indent=2)
print(f"\n  Planning trajectories saved → {traj_path}")

# ── Extract planning artifacts (from 1.1_extract_config.py) ──────────────────
# planning_responses: [0]=plan, [1]=arch_design, [2]=logic_design, [3]=config
overall_plan  = planning_responses[0]
arch_response = planning_responses[1]
logic_response = planning_responses[2]
config_response = planning_responses[3]

# Parse YAML config
config_yaml = extract_yaml_from_content(config_response)
if not config_yaml:
    # Try without language tag
    m = re.search(r'```\s*\n(.*?)\n```', config_response, re.DOTALL)
    config_yaml = m.group(1) if m else config_response

config_yaml_path = os.path.join(OUTPUT_DIR, "planning_config.yaml")
with open(config_yaml_path, "w", encoding="utf-8") as f:
    f.write(config_yaml)

# Parse arch_design and logic_design JSON
arch_design  = content_to_json(arch_response)
logic_design = content_to_json(logic_response)

formatted_arch  = format_json_data(arch_design)  if arch_design  else arch_response
formatted_logic = format_json_data(logic_design) if logic_design else logic_response

# Save artifacts
with open(os.path.join(artifacts_dir, "1.1_overall_plan.txt"),  "w", encoding="utf-8") as f:
    f.write(overall_plan)
with open(os.path.join(artifacts_dir, "1.2_arch_design.txt"),   "w", encoding="utf-8") as f:
    f.write(formatted_arch)
with open(os.path.join(artifacts_dir, "1.3_logic_design.txt"),  "w", encoding="utf-8") as f:
    f.write(formatted_logic)
with open(os.path.join(artifacts_dir, "1.4_config.yaml"),       "w", encoding="utf-8") as f:
    f.write(config_yaml)

# Extract task list
def get_task_list(design: dict) -> list[str]:
    for key in ("Task list", "task_list", "task list"):
        if key in design:
            return design[key]
    return []

def get_logic_analysis(design: dict) -> list:
    for key in ("Logic Analysis", "logic_analysis", "logic analysis"):
        if key in design:
            return design[key]
    return []

todo_file_lst   = get_task_list(logic_design)
logic_analysis  = get_logic_analysis(logic_design)
logic_analysis_dict = {item[0]: item[1] for item in logic_analysis if isinstance(item, (list, tuple)) and len(item) >= 2}

if not todo_file_lst:
    print("\n[WARN] Could not parse Task list from logic design. Check artifacts/1.3_logic_design.txt.")
    todo_file_lst = ["main.py"]

print(f"\n  Task list: {todo_file_lst}")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Per-file Logic Analysis  (from 2_analyzing.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Stage 2: Per-file Logic Analysis")
print("=" * 60)

ANALYSIS_SYSTEM = f"""\
You are an expert researcher, strategic analyzer, and PyTorch software engineer
with a deep understanding of experimental design and reproducibility in scientific
research.

You will receive:
- A research paper in Markdown format
- An overview of the implementation plan
- An architecture design (file list, class interfaces, call flow)
- A task breakdown with logic analysis and task list
- A configuration file (config.yaml)

Your task is to conduct a comprehensive logic analysis to accurately reproduce
the experiments and methodologies described in the paper.

Rules:
1. Align with the Paper: strictly follow methods, datasets, model configurations,
   hyperparameters, and experimental setups.
2. PyTorch first: all neural network components (models, losses, optimizers,
   data loaders) must use PyTorch (torch, torch.nn, torch.optim, torch.utils.data).
   Do not propose TensorFlow, Keras, or sklearn for any neural network work.
3. Data from NET_FILE only: all data must be parsed directly from the SWMM
   network file (Ji.inp). Do NOT propose or assume any synthetic data generation.
4. Be Clear and Structured: present analysis in a logical, well-organized,
   actionable format.
5. Follow the design: YOU MUST FOLLOW "Data structures and interfaces".
   Do not use public member functions that do not exist in the design.
6. Reference configuration: always use settings from config.yaml.
   Do not invent or assume any values.\
"""

analyzing_artifacts_dir = os.path.join(OUTPUT_DIR, "analyzing_artifacts")
os.makedirs(analyzing_artifacts_dir, exist_ok=True)

detailed_logic_analysis_dict: dict[str, str] = {}

for todo_file_name in todo_file_lst:
    if todo_file_name == "config.yaml":
        continue

    print(f"\n  [ANALYSIS] {todo_file_name}")
    file_desc = logic_analysis_dict.get(todo_file_name, "")
    draft_desc = (
        f"Write the logic analysis in '{todo_file_name}', intended for '{file_desc}'."
        if file_desc.strip()
        else f"Write the logic analysis in '{todo_file_name}'."
    )

    analysis_messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user",   "content": f"""## Paper
{paper_text}

## Network / Data File
{net_content}

-----

## Overview of the plan
{overall_plan}

-----

## Design
{arch_response}

-----

## Task
{logic_response}

-----

## Configuration file
```yaml
{config_yaml}
```

-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the
plan, the design, the task, and the configuration file.
You do NOT need to provide actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""},
    ]

    analysis_content = api_call(analysis_messages, max_tokens=8000)
    detailed_logic_analysis_dict[todo_file_name] = analysis_content

    save_name = todo_file_name.replace("/", "_")
    with open(os.path.join(analyzing_artifacts_dir, f"{save_name}_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(analysis_content)
    print(f"    → {len(analysis_content.splitlines())} lines")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Per-file Code Generation  (from 3_coding.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Stage 3: Per-file Code Generation")
print("=" * 60)

CODE_SYSTEM = """\
You are an expert researcher and software engineer with a deep understanding of
experimental design and reproducibility in scientific research.

Your task is to write code to reproduce the experiments and methodologies
described in the paper.

Rules:
- PyTorch is the primary framework: use torch, torch.nn, torch.optim,
  torch.utils.data for all neural network components.
- Data from NET_FILE only: load and parse the SWMM .inp file (path from config)
  to build all datasets. Do NOT generate synthetic data anywhere in the code.
- Write elegant, modular, maintainable code following Google-style guidelines.
- Strictly align with the paper's methodology, experimental setup, and metrics.
- Follow "Data structures and interfaces" exactly; do not change the design.
- Use configuration values from config.yaml; do not fabricate values.
- Set default values for all settings; use strong types and explicit variables.
- Use CUDA if available (torch.device("cuda" if torch.cuda.is_available() else "cpu")).
- Avoid circular imports.
- Write out EVERY code detail; leave no TODOs.
- Write code inside triple-backtick fenced code blocks.\
"""

coding_artifacts_dir = os.path.join(OUTPUT_DIR, "coding_artifacts")
os.makedirs(coding_artifacts_dir, exist_ok=True)

done_file_lst:  list[str]        = ["config.yaml"]
done_file_dict: dict[str, str]   = {"config.yaml": config_yaml}

for todo_file_name in todo_file_lst:
    if todo_file_name == "config.yaml":
        continue

    print(f"\n  [CODING] {todo_file_name}")

    # Accumulate previously generated code (from 3_coding.py get_write_msg)
    code_files_section = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"):
            continue
        code_files_section += f"\n```python\n{done_file_dict[done_file]}\n```\n"

    file_analysis = detailed_logic_analysis_dict.get(todo_file_name, "")

    coding_messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user",   "content": f"""# Context
## Overview of the plan
{overall_plan}

-----

## Design
{arch_response}

-----

## Task
{logic_response}

-----

## Configuration file
```yaml
{config_yaml}
```

-----

## Code Files (already implemented)
{code_files_section if code_files_section.strip() else "(none yet)"}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on everything above, write ONLY the file "{todo_file_name}".

1. Only one file: implement THIS ONLY ONE FILE completely.
2. Complete code: no stubs, no TODOs — write every detail.
3. Follow the design exactly: use only the interfaces defined in "Data structures
   and interfaces". Do not add or rename public methods.
4. Reference config.yaml for all hyperparameters.
5. Import every external symbol you use before using it.
6. Output paths must point to S5/results_P2C/ and be created with os.makedirs if absent.
7. Use CUDA if available.

## Logic Analysis for {todo_file_name}:
{file_analysis}

## Code: {todo_file_name}"""},
    ]

    code_content = api_call(coding_messages, max_tokens=64000)

    # Extract code from response
    code = extract_code_from_content(code_content)
    if not code:
        code = code_content  # fallback: use raw response

    done_file_lst.append(todo_file_name)
    done_file_dict[todo_file_name] = code

    # Save to repo
    save_name = todo_file_name.replace("/", "_")
    if "/" in todo_file_name:
        sub_dir = os.path.join(REPO_DIR, os.path.dirname(todo_file_name))
        os.makedirs(sub_dir, exist_ok=True)
    out_path = os.path.join(REPO_DIR, todo_file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Save artifact (raw LLM response)
    with open(os.path.join(coding_artifacts_dir, f"{save_name}_coding.txt"), "w", encoding="utf-8") as f:
        f.write(code_content)

    print(f"    → {len(code.splitlines())} lines  saved to {out_path}")

# Also copy config.yaml to repo
import shutil
shutil.copy(config_yaml_path, os.path.join(REPO_DIR, "config.yaml"))

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Debugging / Review  (from 4_debugging.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Stage 4: Debugging Review")
print("=" * 60)

# Compile all generated code
python_dict = read_python_files(REPO_DIR)
codes_section = ""
for todo_file in todo_file_lst:
    if todo_file.endswith(".yaml"):
        continue
    if todo_file in python_dict:
        codes_section += f"```python\n## File name: {todo_file}\n{python_dict[todo_file]}\n```\n\n"

codes_section += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"

DEBUG_SYSTEM = """\
You are a highly capable code assistant specializing in debugging real-world code
repositories for scientific ML. You will be provided with a complete code repository.

Your objective is to proactively identify and fix all bugs before execution.

Review focus:
- Tensor shape / dtype mismatches throughout.
- Train/val/test split correctness; no normalization leakage (fit scaler on train only).
- Autoregressive rollout: detach predictions before feeding as next-step input.
- Gradient flow: loss.backward() must propagate through the full computation graph.
- optimizer.zero_grad() placement; scheduler.step() timing.
- model.eval() + torch.no_grad() during validation and test inference.
- Output paths pointing to S5/results/; created with os.makedirs if absent.
- Logical inconsistencies between the paper method and the implementation.

Guidelines:
- Provide exact lines needed to resolve each issue using SEARCH/REPLACE format.
- If multiple fixes are needed, provide them sequentially.
- Do not make speculative edits; justify each change.
- Prioritize minimal, effective fixes that preserve the original intent.

Output format — for each modified file:
Filename: <filename>
<<<<<<< SEARCH
<original lines>
=======
<corrected lines>
>>>>>>> REPLACE

## Answer\
"""

debug_messages = [
    {"role": "system", "content": DEBUG_SYSTEM},
    {"role": "user",   "content": f"### Code Repository\n{codes_section}"},
]

debug_response = api_call(debug_messages, max_tokens=32000)

# Save debug response
debug_artifact_path = os.path.join(artifacts_dir, "4_debug_patches.txt")
with open(debug_artifact_path, "w", encoding="utf-8") as f:
    f.write(debug_response)
print(f"  Debug response saved → {debug_artifact_path}")

# Apply SEARCH/REPLACE patches
parse_and_apply_changes([debug_response], REPO_DIR, save_num=1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
final_files = list(read_python_files(REPO_DIR).keys()) + ["config.yaml"]
print(f"Done. {len(final_files)} files in {REPO_DIR}/")
for fn in final_files:
    fp = os.path.join(REPO_DIR, fn)
    lines = len(open(fp, encoding="utf-8").read().splitlines()) if os.path.exists(fp) else 0
    print(f"  ✓ {fn}  ({lines} lines)")
print("=" * 60)
