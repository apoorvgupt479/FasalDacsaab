# FasalDacsaab — Step-by-Step Task Checklist

This document tracks the incremental software and machine learning engineering tasks required to build **FasalDacsaab**.

---

## Phase 1: Local Workspace & Data Ingestion Setup
- [ ] Initialize Python virtual environment and install core packages (`torch`, `torchvision`, `numpy`, `matplotlib`, `streamlit`, `onnxruntime`, `opencv-python`).
- [ ] Establish repository directory structure:
  - `/data/raw` (untracked raw images)
  - `/data/processed` (metadata file mappings)
  - `/src` (modules for dataset, model, train, explain)
  - `/notebooks` (EDA and prototyping notebooks)
  - `/models` (checkpoints and final artifacts)
  - `/docs` (documentation and reports)
- [ ] Create `.gitignore` to prevent tracking of dataset binary files (`/data/raw/`), temporary files, checkpoits, and local cache.
- [ ] Implement data ingestion scripts in `src/dataset.py`:
  - Fetch and extract the PlantVillage crop leaf disease dataset.
  - Parse metadata to establish target class structures and save mappings.
- [ ] Write a script or notebook (`notebooks/01_eda.ipynb`) to verify image resolutions, check class imbalances, and plot class distributions.

---

## Phase 2: Core Model Development & Transfer Learning
- [ ] Write PyTorch `Dataset` loaders with custom index maps.
- [ ] Design image pre-processing and data augmentation transformations (Resize 224x224, RandomHorizontalFlip, RandomRotation, Normalization matching ImageNet stats).
- [ ] Set up transfer learning module (`src/models.py`) importing pretrained convolutional backbones (e.g., ResNet50/EfficientNet).
- [ ] Implement target classification head mapping backbone output features to the 38 plant/disease classes.
- [ ] Implement training loop script (`src/train.py`):
  - Configure Adam optimizer and CrossEntropyLoss criterion.
  - Establish train/validation splits (e.g., 80/20 or 70/15/15).
  - Add checkpoint saving logic.
  - Implement Early Stopping based on validation loss patience.
  - Integrate a learning rate scheduler (e.g., `ReduceLROnPlateau`).

---

## Phase 3: Explainability Engine & Advanced Diagnostics
- [ ] Construct the Grad-CAM (Gradient-weighted Class Activation Mapping) module in `src/explain.py`:
  - Register forward and backward hooks to capture activation maps and gradients of the final convolutional layer.
  - Implement weighted pooling of gradients to form visual heatmaps.
  - Write overlays generating jet-colored visual highlights on original input crop images.
- [ ] Implement model validation profiling:
  - Generate per-class Precision, Recall, and F1-Score reports.
  - Save confusion matrices to analyze model blindspots (e.g., confusing Tomato Early Blight with Tomato Late Blight).
  - Write a diagnostic script to extract the top-5 worst-predicted images based on classification loss for qualitative review.

---

## Phase 4: Web Application & Serialization
- [ ] Export the fine-tuned PyTorch model (`.pth`) to optimized ONNX (`.onnx`) format.
- [ ] Write a latency benchmark script comparing inference time between raw PyTorch and ONNX Runtime backends.
- [ ] Create the **Streamlit** dashboard in `app.py`:
  - Implement a drag-and-drop file uploader accepting JPG/PNG leaf images.
  - Integrate ONNX Runtime inference to compute leaf diagnostics.
  - Display horizontal progress bars representing prediction confidence percentages.
  - Render side-by-side visualization showing the uploaded image next to the Grad-CAM highlight overlay.

---

## Phase 5: Production Deployment & Technical Documentation
- [ ] Deploy Streamlit application onto a public cloud platform (e.g., Streamlit Community Cloud or Hugging Face Spaces).
- [ ] Compile final documentation:
  - Draft project README with clear setup, usage, and quickstart commands.
  - Create model card (`docs/model_card.md`) stating performance metrics, parameters, and limitations.
  - Review and document Architecture Decision Records (ADRs).
