"""
refer: https://github.com/matlab-deep-learning/Inverse-Problems-using-Physics-Informed-Neural-Networks-PINNs

S4.py — MATLAB → Modular Python Conversion via LLM
Workflow:
    1. Read S4/data/InversePinnConstantCoef.m (MATLAB code for a PINN inverse problem)
    2. First LLM call: analyze MATLAB code and plan modular Python architecture (JSON)
    3. Second LLM call: generate Python files according to the modular architecture (JSON)
    4. Save generated Python files to S4/results/
"""

import json
import os
from openai import OpenAI

MATLAB_FILE  = "S4/data/InversePinnConstantCoef.m"
TASK_FILE    = "S4/data/task_en.md"
OUTPUT_DIR   = "S4/results"

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ## Replace with your actual API key
    base_url="https://api.deepseek.com/v1",
)

# ── Read MATLAB source code and task requirements ─────────────────────────────
with open(MATLAB_FILE, "r", encoding="utf-8") as f:
    matlab_code = f.read()

with open(TASK_FILE, "r", encoding="utf-8") as f:
    task_requirements = f.read()

# ── First call: analyze MATLAB code and plan modular architecture ─────────────
PLAN_SYSTEM = """You are a software architect specializing in scientific computing and deep learning.
Analyze the provided MATLAB PINN (Physics-Informed Neural Network) code and produce a modular
Python project architecture plan.

The plan must follow the modular design principles described in the task requirements, splitting
the code into at least these modules:
- data_preprocessing.py  : collocation point generation, boundary node extraction
- model_definition.py    : neural network architecture, forward pass, loss function
- model_training.py      : training loop, ADAM optimizer updates, learning rate schedule
- model_inference.py     : load trained model, predict on mesh nodes
- performance_evaluation.py : compute errors, plot solution, report recovered coefficient

Return ONLY a valid JSON object with key "modules", a list of objects each having:
  - "filename": the Python filename
  - "description": one-sentence purpose
  - "matlab_sections": list of MATLAB section names/lines this module corresponds to
No markdown, no extra text."""

PLAN_USER = f"""MATLAB code to analyze:

{matlab_code}

Task requirements (modular design principles):

{task_requirements}"""

print("Step 1: LLM planning modular architecture...")
plan_response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user",   "content": PLAN_USER},
    ],
    max_tokens=64000,
    stream=False,
)

plan_content = plan_response.choices[0].message.content.strip()
if plan_content.startswith("```"):
    plan_content = "\n".join(plan_content.splitlines()[1:])
    if plan_content.rstrip().endswith("```"):
        plan_content = plan_content.rstrip()[:-3]

architecture_plan = json.loads(plan_content)
print(f"  Planned {len(architecture_plan['modules'])} modules:")
for m in architecture_plan["modules"]:
    print(f"  • {m['filename']}: {m['description']}")

# ── Second call: generate Python code for each module ─────────────────────────
GEN_SYSTEM = """You are an expert Python developer specializing in PyTorch-based Physics-Informed
Neural Networks (PINNs).

Convert the provided MATLAB PINN code for an inverse problem (recovering the diffusion coefficient c
in a Poisson equation on a unit disk) into fully functional, modular Python code.

Requirements:
- Use PyTorch (torch, torch.autograd) instead of MATLAB Deep Learning Toolbox
- Replace MATLAB PDE toolbox mesh generation with a simple random/grid collocation point sampler
- Each module must be self-contained with clear imports
- Include a main.py that imports and orchestrates all modules (training pipeline)
- The recovered coefficient c and the PINN solution plot must be produced by inference + evaluation modules
- All files must have docstrings explaining their purpose
- Do NOT use MATLAB-specific toolbox calls; approximate mesh/boundary nodes with analytic unit-disk sampling

Return ONLY a valid JSON object where each key is a filename (e.g. "model_definition.py") and
the value is the complete file content as a string.
Include all modules from the architecture plan plus main.py.
No markdown, no extra text."""

GEN_USER = f"""Architecture plan:
{json.dumps(architecture_plan, indent=2)}

Original MATLAB code:
{matlab_code}"""

print("\nStep 2: LLM generating Python code for each module...")
gen_response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user",   "content": GEN_USER},
    ],
    max_tokens=64000,
    stream=False,
)

gen_content = gen_response.choices[0].message.content.strip()
if gen_content.startswith("```"):
    gen_content = "\n".join(gen_content.splitlines()[1:])
    if gen_content.rstrip().endswith("```"):
        gen_content = gen_content.rstrip()[:-3]

generated_files = json.loads(gen_content)

# ── Save module files ─────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving generated Python files to {OUTPUT_DIR}/")
for filename, content in generated_files.items():
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ {filename}  ({len(content.splitlines())} lines)")

print(f"\nDone: generated {len(generated_files)} Python files")
