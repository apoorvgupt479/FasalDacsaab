# FasalDacsaab — Advanced Image Classification & Detection for Agritech

FasalDacsaab is an intelligent agricultural crop monitoring application designed to assist farmers in real-time crop disease diagnosis and health assessment. By analyzing leaf images uploaded by users, the application identifies crop health issues (e.g., specific diseases, early blights) or determines harvest readiness, providing actionable insights with visual explanations.

## 1. Product Scenario

Modern agriculture faces challenges in timely disease identification. FasalDacsaab addresses this by placing automated crop diagnostics directly in the hands of growers. Users submit photographs of plant leaves or fruits, and the underlying computer vision models evaluate the sample to diagnose conditions or determine maturity.

## 2. Problem Statement & Scope

Build the end-to-end classification/detection pipeline for **FasalDacsaab**:

* **Dataset:** PlantVillage (leaf disease classification, recommended) or a custom crop condition image dataset.
* **Task:** Image Classification (image → disease class/healthy state) or Object Detection (image → localized disease bounding boxes + labels).
* **Model:** Transfer learning using a pretrained backbone (e.g., ResNet, EfficientNet, or YOLO) with appropriate justification.
* **Training Pipeline:** Multi-stage pipeline with robust train/val/test splitting, image augmentation (rotation, flip, color jitter), early stopping, and learning rate scheduling.
* **Evaluation:** Comprehensive performance profiling including per-class Precision, Recall, F1-Score, Confusion Matrix, and visualization of top-5 misclassified samples.
* **Deployment:** Interactive web interface (Streamlit or Gradio) for image uploads, inference predictions, confidence display, and visual model explanations.
* **Export:** Model serialization via ONNX format with runtime optimization analysis for edge device deployment.

## 3. Core Tech Stack & Learning Goals

* **PyTorch Foundations:** Tensors, Custom Dataset Classes, DataLoaders, and modular training/evaluation loops.
* **Transfer Learning:** Fine-tuning strategies (layer freezing, discriminative learning rates).
* **Data Augmentation:** Robust transformations using `torchvision.transforms` or `albumentations` to prevent overfitting.
* **Classification/Detection Metrics:** Confusion matrix analysis, calibrated confidence scores, and mean Average Precision (mAP) if implementing detection.
* **Deployment & Inference:** Gradio/Streamlit frontend and ONNX Runtime backends.

## 4. Visual Explainability (Grad-CAM)

To establish model reliability, FasalDacsaab integrates a **"Show Me Why"** explainability layer:

* **Grad-CAM (Gradient-weighted Class Activation Mapping):** Visualizes the activation maps of the final convolutional layer.
* **Saliency Overlay:** Generates heatmaps overlaid on the original user-submitted images, highlighting the exact visual regions (e.g., spots, lesions) that informed the prediction.

## 5. System Deliverables

* **Repository Architecture:** Clean, modular structure containing ingestion scripts, model files, evaluation pipelines, and web app.
* **Model Artifacts:** Pretrained weights, serialized ONNX models, and detailed model cards.
* **Live Demo:** Working deployment of the Streamlit interface.
* **Evaluation Report:** Technical evaluation including metric logs, validation plots, and error analysis.
* **Architecture Decision Records (ADRs):** Clean records of key design and technology decisions.
