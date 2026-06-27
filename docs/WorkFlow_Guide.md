# FasalDacsaab — Developer Workflow Guide

This document defines the development workflow, coding standards, version control policies, and testing guidelines for **FasalDacsaab**, the crop disease diagnosis and computer vision platform.

---

## 1. Project Workflow Overview

The development cycle of FasalDacsaab consists of sequential, iterative tasks spanning data preparation, training, explaining, and deploying.

```mermaid
flowchart LR
    A[(PlantVillage Raw Data)] --> B[Pre-processing & Augment]
    B --> C[PyTorch Model Training]
    C --> D[Grad-CAM Analysis]
    D --> E[ONNX Serialization]
    E --> F[Streamlit Web App]
```

### Git Branching & Version Control

1. **Branching Model:**
   - Use `main` as the stable production branch.
   - For individual features or experiments, use feature branches (e.g., `feat/data-loader`, `exp/efficientnet-backbone`, `feat/grad-cam`).
2. **Commit Strategy:**
   - Commit often with semantic commit messages (e.g., `feat: add custom dataset loaders for plantvillage`, `fix: resolve learning rate schedule scaling issue`).
   - Push code incrementally to remote branches to preserve a complete commit graph.

---

## 2. Machine Learning Workflow Specifics

Development tasks should be executed with the following engineering standards:

### Data Engineering & Ingestion

- Raw data must reside in `/data/raw/` (ignored via `.gitignore`).
- Preprocessed or parsed metadata (e.g., image-path mappings, split indices) is saved to `/data/processed/`.
- The dataset module in `src/dataset.py` must support training, validation, and testing phases with consistent class mapping indices.

### Modeling & Training Standards

- Model backbones (e.g., ResNet, EfficientNet) must be configured via a config file or arguments.
- Always use seed values for randomization (e.g., NumPy, PyTorch, Python random) to ensure reproducibility.
- Training scripts (`src/train.py`) must log the following metrics at each epoch:
  - Training Loss & Accuracy.
  - Validation Loss & Accuracy.
- Keep saved checkpoints in `/models/checkpoints/` and save the final best model as `/models/FasalDacsaab_best.pth`.

### Interpretability Pipeline

- The explainability component should be implemented in `src/explain.py`.
- Heatmap overlay images must be saved in png format during test phase evaluations to verify correct visual activations (e.g., focus on leaf lesions rather than background soil).

### Inference Optimization (ONNX)

- Standardize inference input sizes (e.g., 224x224 RGB tensor).
- Benchmark standard PyTorch inference latency against ONNX Runtime inference latency over a 100-sample test batch.

---

## 3. Coding & Documentation Standards

### Code Formatting

- Code must follow PEP-8 standards. Use a formatter like `black` or `autopep8`.
- Include docstrings in all core functions and classes using the Google Style docstring format.

### Repository Layout

```
FasalDacsaab/
├── .git/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── adr/
│   └── architecture_diagram.png
├── models/
│   ├── checkpoints/
│   └── FasalDacsaab_best.onnx
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_eval.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── explain.py
│   ├── models.py
│   └── train.py
├── app.py
├── requirements.txt
├── Execution_Plan.md
├── Objective.md
└── WorkFlow_Guide.md
```

### Technical Documentation Requirements

Each feature integration requires corresponding documentation:

- **Architecture Decision Records (ADRs):** Write decisions around datasets, neural network backbones, and server frameworks in `/docs/adr/`.
- **Model Card:** Create a detailed description of the final model's performance, target metrics, training environment, and intentional usage bounds.
