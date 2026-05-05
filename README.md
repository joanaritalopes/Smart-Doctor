# The Smart Doctor – Skin Lesion Classification

## Overview

This project tackles a 7-class skin lesion classification problem using the [DermaMNIST](https://medmnist.com/) dataset. The focus is on comparing CNN and Transformer-based architectures, applying interpretability techniques to understand model decisions, and (bonus) building a Retrieval-Augmented Generation pipeline for medical Q&A.

Medical imaging is a high-stakes domain: the distinction between melanoma and benign nevi, for instance, is clinically critical and visually subtle even for human experts. The goal is not just accuracy but understanding *why* a model makes a prediction.

---

## Dataset

**DermaMNIST** – 7 classes of dermoscopic skin lesion images:

| Class | Description |
|-------|-------------|
| Actinic keratoses | Pre-cancerous sun damage |
| Basal cell carcinoma | Most common skin cancer, highly curable |
| Benign keratosis | Harmless growths |
| Dermatofibroma | Benign skin nodules |
| Melanoma | Aggressive skin cancer – critical to detect |
| Melanocytic nevi | Common moles – visually similar to melanoma |
| Vascular lesions | Blood vessel growths |

~7,000 training images at 224×224 resolution (RGB).

---

## Parts Implemented

### Part 1 – CNNs vs. Vision Transformers

Fine-tuned two ImageNet-pretrained models on DermaMNIST:

- **ResNet-18** – Lightweight CNN, fast training, strong inductive bias for local features
- **ViT-B/16** – Vision Transformer, global attention, requires more data to generalise

Both models were evaluated on accuracy, confusion matrix, training dynamics, and parameter efficiency.

**Key observations:**
- ResNet-18 converges faster and is more stable with frozen backbone fine-tuning on a small dataset
- ViT benefits from unfreezing more layers progressively once training stabilises
- Both models struggle most with the melanoma/nevi distinction, mirroring human expert difficulty

### Part 2 – Model Interpretability

Two complementary techniques applied to the same test images:

- **GradCAM** (ResNet-18) – Highlights which spatial regions drove the CNN's prediction by backpropagating gradients into the final conv layer
- **Attention Rollout** (ViT-B/16) – Propagates attention weights across all transformer layers to show which image patches the ViT attended to

Overlaid heatmaps reveal whether models focus on clinically relevant lesion regions or background artifacts.

### Part 3 – RAG Pipeline

A local Retrieval-Augmented Generation pipeline using:
- `llama-index` for document ingestion and vector indexing
- `Ollama` (Gemma-3) as the local LLM
- `HuggingFace BGE` embeddings

Allows querying a corpus of dermatology documents with grounded, citation-backed answers.

---

## Project Structure

```
smart_doctor.py   # Main implementation
```

Key components:

| Component | Description |
|-----------|-------------|
| `build_resnet18` | ImageNet ResNet-18 with custom classification head |
| `build_vit` | ImageNet ViT-B/16 with custom classification head |
| `train_model` | Full training loop with cosine LR scheduler |
| `evaluate` | Computes loss and accuracy over a DataLoader |
| `get_predictions` | Returns (y_true, y_pred) for confusion matrix analysis |
| `GradCAM` | Hook-based GradCAM implementation for CNNs |
| `attention_rollout` | Attention Rollout for ViT models |
| `build_rag_pipeline` | Local RAG pipeline with llama-index + Ollama |

---

## Requirements

```bash
pip install torch torchvision medmnist numpy matplotlib scikit-learn

# For RAG pipeline (Part 4):
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
```

For Part 4 you also need [Ollama](https://ollama.com/) running locally:
```bash
ollama pull gemma3
ollama serve
```

---

## Usage

```python
from smart_doctor import (
    get_data_loaders, build_resnet18, build_vit,
    train_model, evaluate, get_predictions,
    plot_training_history, plot_confusion_matrix,
    GradCAM, visualize_gradcam,
    visualize_attention_rollout,
    CLASS_NAMES, CONFIG, DEVICE,
)
import torch.nn as nn

# Load data
train_loader, val_loader, test_loader, n_classes = get_data_loaders(64)

# Fine-tune ResNet-18
resnet = build_resnet18(n_classes, freeze_backbone=True)
history = train_model(resnet, train_loader, val_loader, num_epochs=20)
plot_training_history(history, title="ResNet-18")

# Evaluate
_, test_acc = evaluate(resnet, test_loader, nn.CrossEntropyLoss(), DEVICE)
print(f"Test accuracy: {test_acc:.4f}")

# Interpretability
sample_img, sample_label = next(iter(test_loader))
gradcam = GradCAM(resnet, target_layer=resnet.layer4[-1])
visualize_gradcam(sample_img[0], sample_label[0].item(), gradcam, CLASS_NAMES)
```
