"""
The Smart Doctor - Skin Lesion Classification
---------------------------------------------

Parts included:
  Part 1 - CNNs vs. Vision Transformers (ResNet-18 vs. ViT-B/16 on DermaMNIST)
  Part 2 - Model Interpretability (GradCAM + ViT Attention Rollout)
  Part 4 - RAG pipeline (bonus, requires local Ollama server)

Dataset: DermaMNIST - 7-class skin lesion classification
  actinic keratoses | basal cell carcinoma | benign keratosis |
  dermatofibroma | melanoma | melanocytic nevi | vascular lesions
"""

import os
import time
import math
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "data_flag": "dermamnist",
    "download": True,
    "size": 224,
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "num_workers": 0,
    "device": "mps" if torch.backends.mps.is_available() else (
              "cuda" if torch.cuda.is_available() else "cpu"),
}

DEVICE = torch.device(CONFIG["device"])

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


# ─── Part 1 – Data Loading ────────────────────────────────────────────────────

def get_augmentation_transforms():
    """Return training augmentation transforms suitable for dermoscopic images."""
    return transforms.Compose([
        transforms.Resize((CONFIG["size"], CONFIG["size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms():
    """Return deterministic transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize((CONFIG["size"], CONFIG["size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

import medmnist
from medmnist import INFO

def get_data_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load DermaMNIST via medmnist and return (train_loader, val_loader, test_loader, n_classes).

    Requires: pip install medmnist
    """

    info = INFO[CONFIG["data_flag"]]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    train_ds = DataClass(split="train", transform=get_augmentation_transforms(),
                         download=CONFIG["download"], size=CONFIG["size"])
    val_ds   = DataClass(split="val",   transform=get_eval_transforms(),
                         download=CONFIG["download"], size=CONFIG["size"])
    test_ds  = DataClass(split="test",  transform=get_eval_transforms(),
                         download=CONFIG["download"], size=CONFIG["size"])

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=CONFIG["num_workers"], pin_memory=True)

    return (make_loader(train_ds, True),
            make_loader(val_ds,   False),
            make_loader(test_ds,  False),
            n_classes)


# ─── Part 1 – Model Builders ──────────────────────────────────────────────────

def build_resnet18(n_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load ImageNet-pretrained ResNet-18 and replace the classification head.

    Args:
        n_classes:        Number of output classes.
        freeze_backbone:  If True, freeze all layers except the final FC layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model.to(DEVICE)


def build_vit(n_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load ImageNet-pretrained ViT-B/16 and replace the classification head.

    Args:
        n_classes:        Number of output classes.
        freeze_backbone:  If True, freeze all encoder parameters.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, n_classes)

    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, frozen_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ─── Part 1 – Training Pipeline ───────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, float]:
    """Return (avg_loss, accuracy) over the given DataLoader."""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = DEVICE,
) -> dict:
    """
    Fine-tune *model* and return a history dict with keys:
        train_loss, val_loss, train_acc, val_acc, epoch_time
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "epoch_time")}

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        running_loss = correct = total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        t_epoch = time.time() - t0

        train_loss = running_loss / total
        train_acc  = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        for k, v in zip(("train_loss", "val_loss", "train_acc", "val_acc", "epoch_time"),
                        (train_loss, val_loss, train_acc, val_acc, t_epoch)):
            history[k].append(v)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"time {t_epoch:.1f}s")

    return history


def plot_training_history(history: dict, title: str = "Training history"):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title(f"{title} – Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title(f"{title} – Accuracy")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout(); plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot a row-normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.show()


def get_predictions(model, loader, device):
    """Return (y_true, y_pred) numpy arrays over *loader*."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long()
            preds = model(images).argmax(dim=1).cpu()
            all_true.extend(labels.numpy())
            all_pred.extend(preds.numpy())
    return np.array(all_true), np.array(all_pred)


# ─── Part 2 – GradCAM ────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.

    Usage:
        gradcam = GradCAM(resnet, target_layer=resnet.layer4[-1])
        cam, pred_idx = gradcam(img_tensor.unsqueeze(0).to(device))
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.eval()
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))

        h = w = CONFIG["size"]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def visualize_gradcam(
    img_tensor: torch.Tensor,
    true_label: int,
    gradcam: GradCAM,
    class_names: List[str],
    alpha: float = 0.5,
):
    """Overlay a GradCAM heatmap on the original image."""
    cam, pred_idx = gradcam(img_tensor.unsqueeze(0).to(DEVICE))

    # Denormalise image
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_np = (img_tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

    heatmap = cm.jet(cam)[..., :3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
    overlay = (1 - alpha) * img_np + alpha * heatmap
    axes[2].imshow(overlay.clip(0, 1)); axes[2].axis("off")
    axes[2].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_idx]}")
    plt.tight_layout(); plt.show()


# ─── Part 2 – ViT Attention Rollout ───────────────────────────────────────────

def _patch_vit_for_rollout(model: nn.Module):
    """Monkey-patch ViT encoder blocks so attention weights are stored."""
    for block in model.encoder.layers:
        block.self_attention.forward_orig = block.self_attention.forward

        def _forward_with_weights(self, query, key, value, *args, **kwargs):
            B, N, C = query.shape
            head_dim = C // self.num_heads
            scale = head_dim ** -0.5

            qkv = torch.stack([
                F.linear(query, self.in_proj_weight[i*C:(i+1)*C], self.in_proj_bias[i*C:(i+1)*C])
                for i in range(3)
            ])
            q, k, v = qkv.unbind(0)
            q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._attn_weights = attn.detach()

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
            return out, attn

        import types
        block.self_attention.forward = types.MethodType(_forward_with_weights, block.self_attention)


def attention_rollout(model: nn.Module, img_tensor: torch.Tensor, discard_ratio: float = 0.9):
    """
    Compute Attention Rollout (Abnar & Zuidema, 2020) for a ViT.

    Returns a (h, w) numpy array of attention scores mapped back to the image grid.
    """
    _patch_vit_for_rollout(model)
    model.eval()

    with torch.no_grad():
        _ = model(img_tensor.unsqueeze(0).to(DEVICE))

    attn_maps = []
    for block in model.encoder.layers:
        if hasattr(block.self_attention, "_attn_weights"):
            # Average over heads, shape: (B, N, N)
            attn = block.self_attention._attn_weights.mean(dim=1)
            attn_maps.append(attn[0].cpu())  # drop batch dim

    # Rollout
    result = torch.eye(attn_maps[0].shape[-1])
    for attn in attn_maps:
        # Discard low-attention tokens
        flat = attn.flatten()
        threshold = flat.kthvalue(int(flat.numel() * discard_ratio)).values.item()
        attn = attn.clone()
        attn[attn < threshold] = 0.0

        # Add identity (residual connection)
        attn = attn + torch.eye(attn.shape[-1])
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = attn @ result

    # CLS token attends to all patches
    cls_attn = result[0, 1:]  # skip CLS itself
    grid_size = int(math.sqrt(cls_attn.numel()))
    mask = cls_attn.view(grid_size, grid_size).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def visualize_attention_rollout(
    img_tensor: torch.Tensor,
    model: nn.Module,
    true_label: int,
    class_names: List[str],
    alpha: float = 0.5,
):
    """Overlay attention rollout on the original image."""
    mask = attention_rollout(model, img_tensor)
    mask_resized = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    h = w = CONFIG["size"]
    mask_resized = F.interpolate(mask_resized, size=(h, w), mode="bilinear",
                                  align_corners=False).squeeze().numpy()

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_np = (img_tensor * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

    heatmap = cm.jet(mask_resized)[..., :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap

    with torch.no_grad():
        pred_idx = model(img_tensor.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask_resized, cmap="jet"); axes[1].set_title("Attention rollout"); axes[1].axis("off")
    axes[2].imshow(overlay.clip(0, 1)); axes[2].axis("off")
    axes[2].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_idx]}")
    plt.tight_layout(); plt.show()


# ─── Part 4 – RAG Pipeline ────────────────────────────────────────────

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def build_rag_pipeline(
    documents_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    top_k: int = 3,
    model_name: str = "gemma3",
    ollama_base_url: str = "http://localhost:11434",
):
    """
    Build a simple RAG pipeline using llama-index + local Ollama.

    Requires:
        pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
        ollama pull gemma3  (run in terminal)
        ollama serve        (run in terminal)

    Args:
        documents_dir: Path to directory containing text/PDF documents.
        chunk_size:    Token chunk size for splitting documents.
        chunk_overlap: Overlap between consecutive chunks.
        top_k:         Number of retrieved chunks per query.
        model_name:    Ollama model name.
        ollama_base_url: Base URL for the Ollama server.

    Returns:
        query_engine ready for .query(text) calls.
    """


    # Configure LLM and embedding model
    Settings.llm = Ollama(model=model_name, base_url=ollama_base_url, request_timeout=120)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load and index documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    return query_engine


# ─── Example Usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load data
    train_loader, val_loader, test_loader, n_classes = get_data_loaders(CONFIG["batch_size"])
    print(f"Classes: {n_classes}")

    # ── ResNet-18 ──────────────────────────────────────────────────────────────
    resnet = build_resnet18(n_classes, freeze_backbone=True)
    trainable, frozen = count_parameters(resnet)
    print(f"ResNet-18 | trainable: {trainable:,} | frozen: {frozen:,}")

    resnet_history = train_model(resnet, train_loader, val_loader,
                                  num_epochs=CONFIG["num_epochs"],
                                  lr=CONFIG["learning_rate"])
    plot_training_history(resnet_history, title="ResNet-18")

    criterion = nn.CrossEntropyLoss()
    _, resnet_test_acc = evaluate(resnet, test_loader, criterion, DEVICE)
    print(f"ResNet-18 test accuracy: {resnet_test_acc:.4f}")

    y_true, y_pred = get_predictions(resnet, test_loader, DEVICE)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, title="ResNet-18 Confusion Matrix")

    # ── ViT-B/16 ───────────────────────────────────────────────────────────────
    vit = build_vit(n_classes, freeze_backbone=True)
    trainable, frozen = count_parameters(vit)
    print(f"ViT-B/16  | trainable: {trainable:,} | frozen: {frozen:,}")

    vit_history = train_model(vit, train_loader, val_loader,
                               num_epochs=CONFIG["num_epochs"],
                               lr=CONFIG["learning_rate"])
    plot_training_history(vit_history, title="ViT-B/16")

    _, vit_test_acc = evaluate(vit, test_loader, criterion, DEVICE)
    print(f"ViT-B/16 test accuracy: {vit_test_acc:.4f}")

    y_true, y_pred = get_predictions(vit, test_loader, DEVICE)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, title="ViT-B/16 Confusion Matrix")

    # ── GradCAM on ResNet ───────────────────────────────────────────────────────
    sample_img, sample_label = next(iter(test_loader))
    sample_img   = sample_img[0]
    sample_label = sample_label[0].item()

    gradcam = GradCAM(resnet, target_layer=resnet.layer4[-1])
    visualize_gradcam(sample_img, sample_label, gradcam, CLASS_NAMES)

    # ── Attention Rollout on ViT ────────────────────────────────────────────────
    visualize_attention_rollout(sample_img, vit, sample_label, CLASS_NAMES)