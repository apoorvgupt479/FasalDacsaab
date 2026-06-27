# FasalDacsaab — Smart Agritech Crop Disease Diagnostic Platform

FasalDacsaab is an end-to-end agritech computer vision platform designed to classify crop leaf diseases and provide visual diagnosis explanations (Grad-CAM). The system helps growers perform instant health checks on plant foliage, providing both a class prediction (with confidence scoring) and a visual overlay showing which parts of the leaf (such as lesions or spots) triggered the diagnosis.

---

## 📂 Project Documentation

All primary design specifications, technical documentation, and project schedules are organized inside the [docs/](file:///c:/Users/apoor/Projects/PhalKawach/docs) directory:

- 📋 **[Detailed Objective](file:///c:/Users/apoor/Projects/PhalKawach/docs/detailed_objective.md)**: Product scenario, core technical metrics, and ML requirements.
- 🏗️ **[System Overview & Architecture](file:///c:/Users/apoor/Projects/PhalKawach/docs/system_overview.md)**: Layout of functional modules (Data, Model, Explainability, Optimization, UI).
- 🚀 **[Developer Workflow Guide](file:///c:/Users/apoor/Projects/PhalKawach/docs/WorkFlow_Guide.md)**: Coding standards, repository structure, and Git branching strategies.
- 📅 **[Execution Plan & Milestones](file:///c:/Users/apoor/Projects/PhalKawach/docs/Execution_Plan.md)**: 5-week project roadmap and milestone specifications.
- 🎯 **[Step-by-Step Task Checklist](file:///c:/Users/apoor/Projects/PhalKawach/docs/checklist_of_tasks.md)**: Granular task list for model training, metrics analysis, ONNX optimization, and deployment.
- 📝 **[Project Brief](file:///c:/Users/apoor/Projects/PhalKawach/docs/Objective.md)**: High-level vision and background of the project.

---

## 🛠️ Tech Stack & Key Modules

- **Deep Learning Framework:** PyTorch (backbones include ResNet50 and EfficientNet).
- **Explainability (XAI):** Custom Grad-CAM (Gradient-weighted Class Activation Mapping) overlay mapping.
- **Inference Optimization:** ONNX Runtime for high-throughput, low-latency CPU/GPU execution.
- **Presentation Layer:** Streamlit interactive web application.

---

## ⚡ Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/apoorvgupt479/FasalDacsaab.git
cd FasalDacsaab
```

### 2. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
