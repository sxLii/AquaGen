## The Importance of Modularization and Engineering-Oriented Packaging for Research Code

The author recommends decomposing complex AI code for water systems into multiple clear and independent modules, such as:

- Data Preprocessing
- Model Definition
- Model Training
- Model Inference
- Performance Evaluation

On top of this foundation, these basic modules can be further encapsulated into higher-level functional units, such as a training pipeline, an experiment configuration system, result analysis tools, and visualization/report-generation modules, thereby forming a more complete research software framework.

---

### Why Modular Decomposition Is Necessary

In many engineering and research projects, the original goal of code development is usually not to build a reusable software system, but rather to quickly validate an idea, run an experiment successfully, or generate the figures and metrics needed for a paper. As a result, early-stage research code often exhibits the following characteristics:

- multiple functions are mixed within a single script;
- parameters, file paths, and data formats are hard-coded;
- a large amount of copy-and-paste exists across experiments;
- data processing, model training, and evaluation logic are tightly coupled;
- unified interfaces and documentation are lacking.

Although such code may be sufficient to support a publication, significant problems emerge when it needs to be shared with others, reproduced, extended to new models, or migrated to new datasets. Readers often find it difficult to quickly determine which part of the code is responsible for which function. Even the original author may need a considerable amount of time to re-understand the overall logic after revisiting the code several months later.

Therefore, modularizing research code is not merely about making the code look cleaner; it is a necessary step toward improving reproducibility, maintainability, and shareability.

---

### Recommended Basic Module Division

#### 1. Data Preprocessing Module

The data preprocessing module is responsible for transforming raw data into inputs that can be directly used by the model. It typically includes:

- reading raw monitoring data;
- handling missing values and outliers;
- time alignment and resampling;
- feature construction;
- splitting training, validation, and test sets;
- data standardization or normalization;
- construction of input and output tensors.

The purpose of this module is to isolate the complexity of the raw data, so that downstream model components interact only with a unified data interface.

---

#### 2. Model Definition Module

The model definition module is responsible for describing the architecture and computational logic of the AI model, for example:

- the definition of neural network layers;
- input-output dimensional constraints;
- intermediate variables required by the loss function;
- physical constraint terms or mechanism-embedded components;
- model initialization strategies.

This module is particularly important for water-system AI problems, because many models do not only contain standard deep learning structures, but may also incorporate hydrodynamic mechanisms, mass-conservation constraints, or graph-structured information. If these elements are mixed directly into the training script, it becomes extremely difficult to independently replace the model or compare different model architectures later.

---

#### 3. Model Training Module

The model training module is mainly responsible for controlling the experimental process, including:

- the training loop;
- optimizer and learning-rate scheduling;
- batch iteration;
- loss computation and backpropagation;
- model saving and checkpoint management;
- early stopping strategies;
- logging.

Once the training process is isolated as an independent module, it becomes easier to:

- switch between different models within the same training framework;
- compare different loss functions and hyperparameters;
- save experimental results in a unified manner;
- improve consistency across experimental workflows.

---

#### 4. Model Inference Module

The model inference module is responsible for making predictions on new or test data and processing the outputs after training is completed. Typical functions include:

- loading trained model parameters;
- reading data for inference;
- performing forward propagation;
- denormalizing or inverse-transforming outputs;
- generating predicted time series, node-level results, or scenario simulation results;
- aligning predictions with ground truth for subsequent analysis.

Separating the inference process helps avoid the common situation in which a single script is responsible for training, testing, and plotting simultaneously. It also makes it more convenient to deploy the model in real-world applications.

---

#### 5. Performance Evaluation Module

The performance evaluation module is used to manage model-effectiveness analysis in a unified way, typically including:

- computation of error metrics;
- comparison across multiple scenarios;
- local performance analysis for different nodes or time periods;
- plotting;
- table export;
- organization of ablation-study results;
- evaluation of stability and generalization capability.

Making this module independent helps avoid the problem of evaluation logic being scattered across multiple notebooks or scripts, and makes result analysis more standardized, traceable, and reusable.

---

### Further Encapsulation into Higher-Level Functional Modules

After the basic functional decomposition is completed, the codebase can be further encapsulated into higher-level system modules. For example:

#### 1. Experiment Configuration Module

Used for the unified management of:

- data paths;
- model hyperparameters;
- number of training epochs;
- random seeds;
- device configuration;
- output directories.

This avoids the inefficient practice of manually modifying script parameters repeatedly and also improves experimental reproducibility.

---

#### 2. Training Pipeline Module

This module connects the full workflow — data loading → model initialization → training → validation → result saving — into a standardized pipeline that can be executed with one command.

This can significantly reduce repetitive code and ensure consistent workflows across experiments.

---

#### 3. Visualization and Reporting Module

This module automatically generates result figures, error summary tables, key curves, and comparison plots, thereby reducing the amount of manual post-processing required after experiments are completed.

---

#### 4. Utility Function Module

This module encapsulates common helper functions, such as:

- time-format conversion;
- metric computation functions;
- data-checking functions;
- file saving and loading functions;
- random-seed fixing functions.

Although each utility function may seem simple in isolation, centralized encapsulation can substantially reduce code duplication.

---

### Direct Benefits of Modularization

Once code is decomposed and encapsulated, its benefits are mainly reflected in the following aspects.

#### 1. Improved Readability

A modular structure allows readers to quickly understand the overall workflow of the code. For example, when a new reader sees the project directory, they can immediately identify:

- which part of the code is responsible for reading data;
- which part defines the network;
- which part handles training;
- which part evaluates the results.

This kind of structural clarity is often more important than writing locally sophisticated code.

---

#### 2. Improved Reusability

When functions are split into independent modules, researchers can much more easily reuse existing code. For example:

- replacing the dataset while retaining the original training framework;
- replacing the model architecture while retaining the data-processing and evaluation logic;
- adding new metrics to an existing evaluation module;
- reusing the same utility-function set in other projects.

This is particularly critical for iterative research workflows.

---

#### 3. Reduced Difficulty for Future Users

Research code is often not intended solely for the original author. It may also be used by:

- laboratory colleagues for reproduction;
- collaborators for extension;
- reviewers for inspection;
- open-source community users;
- the future self of the author for maintenance.

If the code is poorly organized, later users must spend substantial time understanding the context and may even misuse the code. Modularization can significantly reduce this communication cost.

---

#### 4. Easier Maintenance and Secondary Development

When one part of the functionality needs to be modified, a modular structure can confine the impact to a local area. For example:

- changing the data-standardization method requires only adjustments to the preprocessing module;
- replacing the neural network architecture requires only modifications to the model definition module;
- adding new evaluation metrics requires only updates to the evaluation module.

This property of local modifiability is central to maintainability.

---

#### 5. Better Support for Research Sharing and Reproducibility

In academic research, code sharing is becoming increasingly important. A project with a clear structure and explicit module boundaries is much easier to:

- attach as supplementary material to a paper;
- publish on GitHub;
- reproduce by other researchers;
- adapt into teaching examples or engineering prototypes.

By contrast, a project consisting of only a single file, with hard-coded dependencies and a disorganized workflow, is difficult for others to use effectively even if it is made public.

---

### Particularly Important for Engineering-Oriented Code

This modular design philosophy is especially important for engineering-oriented AI code. The reason is that engineering problems usually involve:

- heterogeneous data from multiple sources;
- a clear but complex physical background;
- multi-stage processing workflows;
- strong scenario dependence;
- relatively high barriers to reproduction.

For example, in water-system AI research, the code may simultaneously involve:

- raw monitoring-data cleaning;
- spatiotemporal feature construction;
- graph-structured modeling;
- embedding of physical constraints;
- training under multiple operating conditions;
- multi-metric performance evaluation.

If all of these components are written in a single script, the code becomes difficult not only to debug, but also to explain clearly to others in terms of how the research method is actually implemented. Therefore, modularization is not an optional matter of coding style; rather, it is a fundamental requirement for standardization and software-oriented development in engineering research.
