"""
S6.py — Source Code Documentation & Annotation via LLM (3-step pipeline)
Workflow:
  1. Read all Python source files under S6/case1/src/
  2. First LLM call (deepseek-chat):
    Analyze code only -> extract compact spec JSON
    (import list, module responsibilities, call graph, run order, project summary)
  3. Second LLM call (deepseek-chat):
    Generate documentation files from the spec
    (readme.md + requirements.txt + open_source_guide.md)
  4. Third LLM call (deepseek-reasoner):
    Original code + spec -> add inline comments -> results/src/
  5. Save all files to S6/case1/results/
"""

import json
import os
import glob
from openai import OpenAI

SRC_DIR    = "S6/case1/src"
OUTPUT_DIR = "S6/case1/results"

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ## Replace with your actual API key
    base_url="https://api.deepseek.com",
)

# ── Read all source files ─────────────────────────────────────────────────────
src_files = {}
for path in sorted(glob.glob(os.path.join(SRC_DIR, "*.py"))):
    filename = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        src_files[filename] = f.read()

print(f"Read {len(src_files)} source files: {list(src_files.keys())}")

# Build concatenated "filename + content" text for steps 1 and 3
all_code = "\n\n".join(
    f"=== {fname} ===\n{content}"
    for fname, content in src_files.items()
)

def _strip_fence(text: str) -> str:
    """Strip markdown code fences robustly."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════════
# Step 1 (deepseek-chat): Analyze code -> compact spec JSON
# Goal: compress source code into structured information for later steps,
#       avoiding repeated transfer of large raw code in prompts.
# ════════════════════════════════════════════════════════════════════════════
ANALYZE_SYSTEM = """You are a senior Python code analyst.

Read the provided Python source files carefully and extract a concise, structured
specification of the project. Be precise — extract facts, not summaries.

Return ONLY a valid JSON object with these keys:
{
  "project_name": "string",
  "project_description": "2-3 sentence description of what the project does",
  "problem_domain": "string (e.g. 'Physics-Informed Neural Networks for inverse PDE')",
  "third_party_imports": [
    {"package": "import name", "pip_name": "pip install name", "version_hint": "≥X.Y or ''"}
  ],
  "stdlib_imports": ["list of standard library modules used"],
  "modules": [
    {
      "filename": "string",
      "purpose": "one-sentence description",
      "classes": [{"name": "string", "role": "string"}],
      "key_functions": [{"name": "string", "role": "string", "inputs": "string", "outputs": "string"}],
      "imports_from": ["other src files this module imports from"]
    }
  ],
  "workflow": [
    {"step": 1, "action": "string", "module": "string", "details": "string"}
  ],
  "entry_point": "filename of the main entry script",
  "cli_args": [{"arg": "string", "default": "string", "description": "string"}],
  "output_artifacts": ["list of files/plots the pipeline produces"]
}
No markdown, no extra text."""

ANALYZE_USER = f"Source files to analyze:\n\n{all_code}"

print("\nStep 1 (chat): analyzing code and extracting project spec...")
analyze_response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": ANALYZE_SYSTEM},
        {"role": "user",   "content": ANALYZE_USER},
    ],
    stream=False,
    # reasoning_effort="high",
    extra_body={"thinking": {"type": "disabled"}}
    
)

spec = json.loads(_strip_fence(analyze_response.choices[0].message.content))
print(f"  Project: {spec['project_name']}")
print(f"  Module count: {len(spec['modules'])}, Workflow steps: {len(spec['workflow'])}")
print(f"  Third-party dependencies: {[p['pip_name'] for p in spec['third_party_imports']]}")

# ── Save Step 1 output: spec JSON ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
spec_path = os.path.join(OUTPUT_DIR, "step1_spec.json")
with open(spec_path, "w", encoding="utf-8") as f:
    json.dump(spec, f, ensure_ascii=False, indent=2)
print(f"  Step 1 result saved -> {spec_path}")

# ════════════════════════════════════════════════════════════════════════════
# Step 2 (deepseek-chat): generate documentation files from the spec
# Input is compact spec JSON only; raw code is not needed for document generation.
# ════════════════════════════════════════════════════════════════════════════
DOCS_SYSTEM = """You are a technical writer and open-source advisor.

Given a structured project specification, generate three documentation files.

Return ONLY a valid JSON object with exactly these keys:
{
  "readme.md": "complete readme content",
  "requirements.txt": "pip requirements content",
  "open_source_guide.md": "open source platform and license guide content"
}

Requirements for each file:

readme.md:
- Project title and badges placeholder (build, license)
- 2-3 sentence project description
- Table of contents
- "Project Structure" section: table listing each module, its purpose, and key classes/functions
- "Workflow" section: numbered step-by-step run order (what does what, what calls what next)
- "Quick Start" section: installation and run commands
- "Command-line Arguments" section: table of all CLI args with defaults and descriptions
- "Output Artifacts" section: what files the pipeline produces
- "Citation / Reference" section placeholder

requirements.txt:
- One package per line with version constraint (≥ minimum version)
- Include a short inline comment for each package explaining why it is needed
- Group: core ML, utilities, visualization

open_source_guide.md:
- Title: "Open Source Platform & License Guide"
- Section 1 "Recommended Platforms": compare GitHub, GitLab, Bitbucket, Hugging Face for this
  type of project (scientific ML). For each: pros, cons, best for.
  Give a final recommendation with justification.
- Section 2 "License Recommendations": compare MIT, Apache 2.0, GPL-3.0, BSD-3-Clause for
  academic scientific software. For each: key terms, commercial use allowed?, patent clause?,
  copyleft?. Give a final recommendation for this project with justification.
- Section 3 "Checklist before publishing": 10-item checklist for open-sourcing a research repo.

No markdown fences, no extra text outside the JSON."""

DOCS_USER = f"Project specification:\n\n{json.dumps(spec, indent=2)}"

print("\nStep 2 (chat): generating documentation files (readme.md / requirements.txt / open_source_guide.md)...")
docs_response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": DOCS_SYSTEM},
        {"role": "user",   "content": DOCS_USER},
    ],
    stream=False,
    # reasoning_effort="high",
    extra_body={"thinking": {"type": "disabled"}}
)

docs_files = json.loads(_strip_fence(docs_response.choices[0].message.content))
print(f"  Generated documents: {list(docs_files.keys())}")

# ── Save Step 2 outputs: documentation files ──────────────────────────────────
for filename, content in docs_files.items():
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
print(f"  Step 2 result saved -> {out_path}  ({len(content.splitlines())} lines)")

# ════════════════════════════════════════════════════════════════════════════
# Step 3 (deepseek-reasoner): original code + spec -> add inline comments
# Focuses on code understanding, not documentation writing. The spec provides context.
# ════════════════════════════════════════════════════════════════════════════
ANNOTATE_SYSTEM = """You are an expert Python developer adding educational inline comments
to scientific machine learning code.

Rules for annotation:
- Keep ALL existing code and docstrings unchanged — only ADD comments
- Add a # comment above or beside non-obvious lines/blocks explaining WHY, not just WHAT
- For each class: add a brief # ── section header before it
- For each major step inside a function (e.g. gradient computation, loss weighting):
  add a short explanatory comment
- Do NOT add trivial comments like "# import torch" or "# call forward()"
- Do NOT rewrite or reformat any code
- Focus on: autodiff mechanics, loss term purposes, numerical choices, data flow

Return ONLY a valid JSON where each key is the original filename and the value is the
complete annotated file content as a string.
No markdown, no extra text."""

ANNOTATE_USER = f"""Project specification (for context):
{json.dumps({"modules": spec["modules"], "workflow": spec["workflow"]}, indent=2)}

Source files to annotate:
{all_code}"""

print("\nStep 3 (reasoner): adding inline comments to source code...")
annotate_response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": ANNOTATE_SYSTEM},
        {"role": "user",   "content": ANNOTATE_USER},
    ],
    max_tokens=100000,
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

try:
    annotated_files = json.loads(_strip_fence(annotate_response.choices[0].message.content))
    print(f"  Annotated: {list(annotated_files.keys())}")
except json.JSONDecodeError:
    print("  Warning: annotation response format was invalid; using original source files")
    annotated_files = src_files

  # ── Save Step 3 outputs: annotated source files -> S6/results/src/ ───────────
src_out_dir = os.path.join(OUTPUT_DIR, "src")
os.makedirs(src_out_dir, exist_ok=True)
for filename, content in annotated_files.items():
    out_path = os.path.join(src_out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Step 3 result saved -> src/{filename}  ({len(content.splitlines())} lines)")

total = len(docs_files) + len(annotated_files)
print(f"\nDone: generated {total} files -> {OUTPUT_DIR}/")
