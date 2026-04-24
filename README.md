# AquaGen
AquaGen is the code repository for our paper, introducing a novel framework that bridges the gap between data privacy in water system and the need for open scientific research.

## AquaGen Demos Overview

This directory contains 6 core example workflows (S1, S2, S4, S5, S6) related to Urban Drainage Systems (UDS). It comprehensively demonstrates the two core capabilities of the AquaGen framework: **Secure Open-Sourcing of Water Data** and **LLM-Driven Research Code Pipelines**. 

Covered topics include:
- Desensitization of low-sensitivity data
- Synthetic data generation based on mechanistic model (SWMM) outputs and Generative AI
- Task-oriented modular refactoring (MATLAB PINN code to modular Python)
- Multi-stage automated pipeline from academic papers to executable code (Paper-to-Code)
- Automated code documentation and annotation enhancement

*Note: The `S3/` directory is not included in the current workspace.*

---

## Directory Structure & Functions

### S1: Water System Data Desensitization
- **Entry Script:** `S1/S1.py`
- **Input:** `S1/data/undessensitized_water_system_data_1152_rows.csv`
- **Task Document:** `S1/data/task.md`
- **Outputs:**
  - `S1/results/desensitized_output.csv`
  - `S1/results/desensitized_output.json`
- **Key Features:** Uses LLMs to perform rule-based anonymization, remove specific location fields, reduce temporal granularity (e.g., minute to hour), and aggregate rainfall data to prevent critical infrastructure exposure.

### S2: Synthetic Data Generation (SWMM-based)
- **Entry Script:** `S2/S2.py`
- **SWMM Extraction Tool:** `S2/data/generate_swmm_csv.py`
- **Input Sample:** `S2/data/swmm_subset_Tank1_CSO8_C14.csv`
- **Task Document:** `S2/data/task.md`
- **Outputs:**
  - `S2/results/synthetic_output.csv`
  - `S2/results/synthetic_output.json`
- **Key Features:** Addresses highly sensitive data scenarios by fusing Generative AI with mechanistic models:
  1. The 1st LLM call learns the statistical patterns and variable relationships between real rainfall and pipe network responses.
  2. The 2nd LLM call generates 288 rows (1 day, 5-minute granularity) of synthetic data that adheres to physical hydraulic laws.

### S3: AI supported federated learning
- **Generated Code and Entry Script:** `S3/federated_learning_example_v2.py`
- **Origianl Data:** `S3/original_data.xlsx`
- **Output model parameters:** `federated_model.pth`
- **Output model config:** `federated_model_config.pth`

### S4: Task-Oriented Modularization (PINN MATLAB -> Python)
- **Entry Script:** `S4/S4.py`
- **Source MATLAB:** `S4/data/InversePinnConstantCoef.m`
- **Task Documents:** `S4/data/task.md`, `S4/data/task_en.md`
- **Generated Code Directory:** `S4/results/`
  - `main.py`
  - `data_preprocessing.py`
  - `model_definition.py`
  - `model_training.py`
  - `model_inference.py`
  - `performance_evaluation.py`
- **Training Artifacts:**
  - `S4/results/Output/pinn_model.pth`
  - `S4/results/Output/training_results.txt`
- **Key Features:** Uses AI Agents to refactor scattered experimental code into a standardized, modular 5-step Python workflow.

### S5: Paper-to-Code 4-Stage Pipeline
- **Entry Script:** `S5/S5_P2C.py`
- **Inputs:**
  - Paper: `S5/data/Paper.md`, `S5/data/Paper.json`
  - Network config: `S5/data/Ji.inp`
  - Task spec: `S5/data/task.md`
- **Output Directory:** `S5/results_P2C/`
  - Planning artifacts: `artifacts/` (1.1~1.4)
  - Analysis records: `analyzing_artifacts/`
  - Generated code: `repo/` (`config.py`, `model.py`, `trainer.py`, `main.py`, etc.)
  - Logs & Configs: `log.txt`, `planning_config.yaml`
- **Key Features:** A complete LLM pipeline to directly reproduce code from math-heavy papers. Includes planning (architecture/logic/config), file-by-file logic analysis, code generation, and SEARCH/REPLACE-based debugging.

### S6: Automated Documentation & Annotation Enhancement
- **Entry Script:** `S6/S6.py`
- **Source Code:** `S6/data/src/` (`MOC.py`, `FVM.py`, `LBM.py`, `SPH.py`)
- **Output Directory:** `S6/results/`
  - `readme.md`
  - `requirements.txt`
  - `open_source_guide.md`
  - `step1_spec.json`
  - `src/` (Code with enhanced comments)
- **Key Features:** AI Agent reads the entire codebase to extract specifications, automatically generates engineering docs and dependency lists, and enhances inline code comments.

---

## Quick Start

### 1. Environment Preparation

Python 3.10+ is recommended.

```bash
pip install openai pandas numpy matplotlib torch pyswmm
```

Install only the required dependencies if you are running a specific workflow.

### 2. Run in the Demos Root Directory
Bash
#### S1: Data Preprocessing & Desensitization
python S1/S1.py

#### S2: Generative Mechanistic Synthetic Data
python S2/S2.py

#### S4: Modularization Refactoring & Training
python S4/S4.py
python S4/results/main.py

#### S5: Paper-to-Code Automated Pipeline
python S5/S5_P2C.py

#### S6: Automated Documentation & Annotation
python S6/S6.py

### 3. Verify Typical Outputs
```Bash
ls -l S1/results
ls -l S2/results
ls -l S4/results/Output
ls -l S5/results_P2C/repo
ls -l S6/results
```

## Key Dependencies & External Requirements
- **LLM API** (DeepSeek, compatible with OpenAI SDK)
    - Primary models: deepseek-reasoner, deepseek-chat
- **SWMM / PySWMM** (Required for S2 to extract real simulation results and ensure physical hydraulic constraints)
- **PyTorch** (Required for S4, S5, S5_Claude training/inference)
- **GPU (CUDA)** is optional but recommended for accelerating model training.
