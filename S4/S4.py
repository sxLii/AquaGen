"""Generate and automatically repair a modular PyTorch PINN project.

Run this generator in the ``WDS312`` Conda environment. LLM calls stay in that
environment, while generated PyTorch code is validated by the Python interpreter
from ``torchPY312``. Only code that passes validation is copied to ``results``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
MATLAB_FILE = SCRIPT_DIR / "data" / "InversePinnConstantCoef.m"
TASK_FILE = SCRIPT_DIR / "data" / "task_en.md"
OUTPUT_DIR = SCRIPT_DIR / "results"

TORCH_PYTHON = Path(
    os.getenv(
        "TORCH_PYTHON",
        str(Path.home() / "anaconda3" / "envs" / "torchPY312" / "bin" / "python"),
    )
)
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "3"))
VALIDATION_TIMEOUT = int(os.getenv("VALIDATION_TIMEOUT", "300"))

REQUIRED_FILES = {
    "data_preprocessing.py",
    "model_definition.py",
    "model_training.py",
    "model_inference.py",
    "performance_evaluation.py",
    "main.py",
}

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Replace with your actual API key
    base_url="https://api.deepseek.com",
)


PLAN_SYSTEM = """You are a senior software architect specializing in PyTorch and Physics-Informed
Neural Networks. Plan a runnable modular Python implementation of the supplied MATLAB inverse PINN.

The physical problem is:
    eta_tt - gH*eta_xx = 0,  (x,t) in [0,1] x [0,2]
    eta_exact = sin(pi*x)*cos(pi*sqrt(0.981)*t)
The trainable scalar gH starts at 0.5; its exact value is 0.981 m^2/s^2.

Plan exactly these files:
- data_preprocessing.py
- model_definition.py
- model_training.py
- model_inference.py
- performance_evaluation.py
- main.py

Use this exact public API contract. Do not rename these symbols or change their return counts:

data_preprocessing.py
- exact_solution(points: torch.Tensor) -> torch.Tensor of shape (N,1)
- generate_training_data(n_domain: int, n_boundary: int, seed: int,
  device: torch.device) -> dict[str, torch.Tensor] with keys domain_points,
  domain_targets, boundary_points, boundary_targets

model_definition.py
- PINN(hidden_layers: int = 3, hidden_width: int = 50, initial_gH: float = 0.5)
- PINN.forward(points) accepts (N,2) and returns (N,1)
- compute_loss(pinn, domain_points, domain_targets, boundary_points,
  boundary_targets) -> tuple[torch.Tensor, dict[str, torch.Tensor]]

model_training.py
- train(pinn, training_data, *, epochs, batch_size, initial_lr, lr_decay,
  device) -> tuple[PINN, list[dict[str, float]]]
- model_training.py explicitly imports compute_loss from model_definition.py

model_inference.py
- predict_on_grid(pinn, L: float, T: float, n_x: int, n_t: int)
  -> tuple[np.ndarray, np.ndarray, np.ndarray], each shaped (n_t,n_x)

performance_evaluation.py
- compute_metrics(predictions: np.ndarray, exact: np.ndarray) -> dict[str, float]
- save_solution_plot(X, T_grid, predictions, exact, gH, output_path) -> None

main.py
- main(argv=None) -> None
- supports ordinary CLI options: --epochs, --n-domain, --n-boundary, --batch-size,
  --n-x, --n-t, --seed, --device {auto,cpu,cuda}, and --output-dir
- default values run the full experiment; smaller values remain valid normal execution
- do not add special validation-only CLI options or environment-specific imports

The plan must define every cross-module import, function signature, tensor shape, dtype, device
transition, and autograd requirement. The dependency graph must be acyclic. Every name used in a
module must be defined there or explicitly imported there. Collocation coordinates must have
requires_grad=True before the PINN forward pass. Every model input must be on the model's device.

Return only strict JSON with this schema:
{
  "modules": [
    {
      "filename": "required filename",
      "description": "responsibility",
      "public_api": ["exact signatures"],
      "imports_from_generated_modules": {"module": ["symbols"]},
      "tensor_contract": ["shape/device/autograd rules"]
    }
  ],
  "execution_flow": ["ordered steps"],
  "integration_checks": ["checks"]
}
No Markdown fences or extra text."""


GEN_SYSTEM = """You are a senior PyTorch scientific-software engineer. Generate a complete,
runnable project from the architecture plan and MATLAB source. The generated project will be tested
by actually running main.py in a separate Conda environment. Correct all integration problems before
returning your answer.

PHYSICS
- Solve eta_tt - gH*eta_xx = 0 on [0,1] x [0,2].
- exact_solution(points) = sin(pi*x)*cos(pi*sqrt(0.981)*t), shape (N,1).
- PINN input/output shapes are (N,2)/(N,1), using a tanh MLP.
- gH is an nn.Parameter initialized to 0.5 and learned jointly with network weights.
- Loss is 0.4*PDE + 0.6*boundary + 0.5*data.
- Delay gH optimization until 10% of epochs; use Adam and
  lr=initial_lr/(1+lr_decay*global_step).

EXACT PUBLIC API — DO NOT DEVIATE
1. data_preprocessing.py:
   exact_solution(points) -> (N,1) tensor
   generate_training_data(n_domain, n_boundary, seed, device) -> dictionary containing exactly
   domain_points, domain_targets, boundary_points, boundary_targets.
2. model_definition.py:
   PINN(hidden_layers=3, hidden_width=50, initial_gH=0.5)
   compute_loss(pinn, domain_points, domain_targets, boundary_points, boundary_targets)
   -> (total_loss, loss_components), where loss_components is a dictionary.
3. model_training.py:
   train(pinn, training_data, *, epochs, batch_size, initial_lr, lr_decay, device)
   -> (trained_pinn, history). It MUST contain:
       from model_definition import compute_loss
4. model_inference.py:
   predict_on_grid(pinn, L, T, n_x, n_t) -> (X, T_grid, predictions) as CPU NumPy
   arrays shaped (n_t,n_x).
5. performance_evaluation.py:
   compute_metrics(predictions, exact) -> dictionary of float metrics
   save_solution_plot(X, T_grid, predictions, exact, gH, output_path) -> None
6. main.py defines main(argv=None), imports every called symbol explicitly, and supports:
   --epochs, --n-domain, --n-boundary, --batch-size, --n-x, --n-t, --seed,
   --device {auto,cpu,cuda}, --output-dir.

AUTOGRAD AND DEVICE RULES
- In compute_loss, create the differentiable domain tensor before the forward pass:
      domain_points = domain_points.detach().clone().to(model_device).requires_grad_(True)
      eta = pinn(domain_points)
- Compute eta_x, eta_t, eta_xx, and eta_tt against that exact domain_points object using
  torch.autograd.grad with create_graph=True. Use slices [:,0:1] and [:,1:2].
- Never place compute_loss or training inside no_grad/inference_mode. Keep total_loss connected until
  loss.backward(). The training order is zero_grad, compute_loss, backward, optimizer step.
- A DataLoader batch may start with requires_grad=False; compute_loss is responsible for its local
  differentiable clone.
- main moves the model to the selected device. Training moves all batches/targets to that device.
- predict_on_grid infers model_device = next(pinn.parameters()).device, creates/moves coordinates to
  it before pinn(coords), uses eval() and no_grad(), then returns detach().cpu().numpy() arrays.
- Never mix CPU/CUDA tensors. Never call .numpy() on a CUDA tensor.

INTEGRATION AND RUNTIME RULES
- Files are top-level siblings. Use "from model_definition import ...", never relative imports.
- Each non-builtin name is locally defined, accessed through an imported module, or explicitly
  imported. Imported spellings must exactly match definitions. Avoid circular/wildcard imports.
- Use only the standard library, NumPy, PyTorch, and Matplotlib. Set a headless Matplotlib backend
  before importing pyplot. Save plots; never call plt.show().
- main.py must create output directories, print the device, train, infer, evaluate, save a plot, and
  print recovered gH, recovered depth, and metrics.
- General CLI values as small as epochs=2, n_domain=64, n_boundary=8, batch_size=64, n_x=12,
  n_t=12 must execute the same normal pipeline successfully. Do not create a separate validation path.
- --device auto selects CUDA when available. --device cuda fails clearly if unavailable.
- Validate all positive integer CLI values. Handle a final training batch of size one without squeeze.
- Avoid checkpoint dependencies unless main actually saves and loads one consistently.

Before answering, mentally run:
  python main.py --epochs 2 --n-domain 64 --n-boundary 8 --batch-size 64 \
      --n-x 12 --n-t 12 --device auto --output-dir validation_artifacts
Audit imports, exact signatures, return unpacking, tensor shapes, requires_grad timing, devices,
NumPy conversion, plotting, and directory creation. Fix every issue found.

Return only one strict JSON object. Its keys must be exactly data_preprocessing.py,
model_definition.py, model_training.py, model_inference.py, performance_evaluation.py, and main.py.
Each value is the complete Python source string. No Markdown, commentary, TODOs, or omitted code."""


REPAIR_SYSTEM = """You repair generated multi-file PyTorch projects. The current project failed
real validation in the torchPY312 Conda environment. Use the traceback as ground truth, find the
root cause and any closely related integration defects, and return a corrected complete project.

Preserve the required six filenames and exact public API. Do not merely suppress the exception,
skip training/inference, catch-and-ignore errors, disable CUDA, or remove functionality. The normal
small-parameter main.py command must train, infer, evaluate, and save its figure successfully.

Recheck imports, signatures, return unpacking, requires_grad before forward, second derivatives,
CPU/CUDA placement, detach().cpu().numpy(), array shapes, and headless plotting.

Return only a strict JSON object containing all six complete corrected files. No Markdown or extra
text."""


def extract_json(raw: str, response_name: str) -> dict[str, Any]:
    """Parse strict JSON while tolerating a single surrounding code fence."""
    content = raw.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        preview = content[:500].replace("\n", "\\n")
        raise ValueError(
            f"{response_name} is not valid JSON: {exc}; response starts with {preview!r}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"{response_name} must be a JSON object")
    return value


def validate_file_mapping(files: dict[str, Any]) -> dict[str, str]:
    """Reject missing, extra, unsafe, or empty generated files."""
    received = set(files)
    if received != REQUIRED_FILES:
        raise ValueError(
            f"Expected files {sorted(REQUIRED_FILES)}, received {sorted(received)}"
        )
    validated: dict[str, str] = {}
    for filename, source in files.items():
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe filename returned by model: {filename!r}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Source for {filename} must be a non-empty string")
        validated[filename] = source
    return validated


def write_stage(files: dict[str, str], stage_dir: Path) -> None:
    """Write a candidate project to its isolated staging directory."""
    for filename, source in files.items():
        (stage_dir / filename).write_text(source, encoding="utf-8")


def run_command(command: list[str], cwd: Path, label: str) -> tuple[bool, str]:
    """Run one validation command and return a compact diagnostic report."""
    printable = " ".join(command)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=VALIDATION_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        report = (
            f"[{label}] TIMEOUT after {VALIDATION_TIMEOUT}s\n"
            f"Command: {printable}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
        return False, report[-12000:]

    output = (completed.stdout + "\n" + completed.stderr).strip()
    report = (
        f"[{label}] exit_code={completed.returncode}\n"
        f"Command: {printable}\nOUTPUT:\n{output}"
    )
    return completed.returncode == 0, report[-12000:]


def validate_candidate(stage_dir: Path) -> tuple[bool, str]:
    """Compile, import, and run the normal project pipeline under torchPY312."""
    if not TORCH_PYTHON.is_file():
        return False, f"torchPY312 Python interpreter not found: {TORCH_PYTHON}"

    commands = [
        (
            "syntax compilation",
            [str(TORCH_PYTHON), "-m", "compileall", "-q", "."],
        ),
        (
            "module imports",
            [
                str(TORCH_PYTHON),
                "-c",
                (
                    "import data_preprocessing, model_definition, model_training, "
                    "model_inference, performance_evaluation, main; "
                    "print('All generated modules imported successfully')"
                ),
            ],
        ),
        (
            "normal end-to-end run",
            [
                str(TORCH_PYTHON),
                "main.py",
                "--epochs",
                "2",
                "--n-domain",
                "64",
                "--n-boundary",
                "8",
                "--batch-size",
                "64",
                "--n-x",
                "12",
                "--n-t",
                "12",
                "--device",
                "auto",
                "--output-dir",
                str(stage_dir / "validation_artifacts"),
            ],
        ),
    ]

    reports: list[str] = []
    for label, command in commands:
        passed, report = run_command(command, stage_dir, label)
        reports.append(report)
        if not passed:
            return False, "\n\n".join(reports)
    return True, "\n\n".join(reports)


def request_completion(system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
    """Call the configured model and parse its JSON response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty response")
    return extract_json(content, "LLM response")


def repair_candidate(
    files: dict[str, str],
    validation_report: str,
    architecture_plan: dict[str, Any],
    model: str,
) -> dict[str, str]:
    """Ask the LLM to repair a failed candidate using its real traceback."""
    repair_user = f"""Repair the project using the validation report.

<architecture_plan>
{json.dumps(architecture_plan, ensure_ascii=False, indent=2)}
</architecture_plan>

<validation_report>
{validation_report}
</validation_report>

<current_project_json>
{json.dumps(files, ensure_ascii=False)}
</current_project_json>"""
    repaired = request_completion(REPAIR_SYSTEM, repair_user, model)
    return validate_file_mapping(repaired)


def publish_candidate(files: dict[str, str]) -> None:
    """Publish a validated project without retaining validation-only artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        (OUTPUT_DIR / filename).write_text(files[filename], encoding="utf-8")


def main() -> None:
    """Plan, generate, validate, repair if necessary, and publish the project."""
    if not MATLAB_FILE.is_file():
        raise FileNotFoundError(f"MATLAB source not found: {MATLAB_FILE}")
    if not TASK_FILE.is_file():
        raise FileNotFoundError(f"Task requirements not found: {TASK_FILE}")
    if MAX_REPAIR_ATTEMPTS < 0:
        raise ValueError("MAX_REPAIR_ATTEMPTS must be non-negative")

    matlab_code = MATLAB_FILE.read_text(encoding="utf-8")
    task_requirements = TASK_FILE.read_text(encoding="utf-8")

    plan_user = f"""MATLAB code to analyze:

{matlab_code}

Task requirements:

{task_requirements}"""

    print("Step 1: planning the modular architecture...")
    architecture_plan = request_completion(
        PLAN_SYSTEM, plan_user, model="deepseek-v4-flash"
    )
    modules = architecture_plan.get("modules")
    if not isinstance(modules, list):
        raise ValueError("Architecture plan does not contain a modules list")
    planned_files = {
        module.get("filename") for module in modules if isinstance(module, dict)
    }
    if planned_files != REQUIRED_FILES:
        raise ValueError(
            f"Architecture plan has wrong files: {sorted(str(x) for x in planned_files)}"
        )
    for module in modules:
        print(f"  - {module['filename']}: {module.get('description', '')}")

    generation_user = f"""Architecture plan:
{json.dumps(architecture_plan, ensure_ascii=False, indent=2)}

Original MATLAB code:
{matlab_code}

Task requirements:
{task_requirements}"""

    print("\nStep 2: generating the initial Python project...")
    generated_files = validate_file_mapping(
        request_completion(GEN_SYSTEM, generation_user, model="deepseek-v4-pro")
    )

    with tempfile.TemporaryDirectory(prefix="s4_v2_candidate_") as temporary_dir:
        stage_dir = Path(temporary_dir)
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            shutil.rmtree(stage_dir / "validation_artifacts", ignore_errors=True)
            write_stage(generated_files, stage_dir)

            print(
                f"\nStep 3: validating with torchPY312 "
                f"(attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS + 1})..."
            )
            passed, report = validate_candidate(stage_dir)
            print(report)

            if passed:
                publish_candidate(generated_files)
                print(
                    f"\nDone: validated and saved {len(generated_files)} files "
                    f"to {OUTPUT_DIR}"
                )
                return

            if attempt == MAX_REPAIR_ATTEMPTS:
                raise RuntimeError(
                    "Generated project still failed after all repair attempts.\n" + report
                )

            print("\nValidation failed; sending the traceback to the LLM for repair...")
            generated_files = repair_candidate(
                generated_files,
                report,
                architecture_plan,
                model="deepseek-v4-pro",
            )


if __name__ == "__main__":
    main()
