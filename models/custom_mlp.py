"""
custom_mlp.py — Health Grade Classifier (PyTorch MLP from scratch)

A 3-layer Multi-Layer Perceptron for predicting DOHMH letter grades (A/B/C)
from engineered inspection features produced by ``data/preprocess.py``.

This module fulfils the course requirement that at least one algorithm be
implemented without wrapping a library black box.  The model, the training
loop (mini-batching, class weighting, early stopping), the hyperparameter
search, and the counterfactual feature optimiser are all written on top of
PyTorch primitives only.

Author: Rahul Adusumalli (ML Classifier Lead), extended by Ryan Han for
integration with the real preprocessed data pipeline.
Course: CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------

class CustomMLP(nn.Module):
    """
    A 3-layer feed-forward network with ReLU activations and dropout
    regularisation.  Written on top of PyTorch primitives (``nn.Linear``,
    ``nn.ReLU``, ``nn.Dropout``) rather than a high-level wrapper so the
    forward computation is fully explicit.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


# ---------------------------------------------------------------------------
# 2. Training utilities
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Per-epoch diagnostics produced by ``train_mlp``."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_f1: list = field(default_factory=list)
    val_f1: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    stopped_early: bool = False


def _compute_class_weights(y: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to ``num_classes``.

    The DOHMH target distribution is heavily skewed toward grade A (~70-80%),
    so cross-entropy without weighting rewards the model for ignoring B and C.
    """
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights


def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              criterion: nn.Module) -> tuple[float, float]:
    """Compute (loss, weighted-F1) on a tensor pair without grad."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        preds = logits.argmax(dim=1).cpu().numpy()
    f1 = f1_score(y.cpu().numpy(), preds, average="weighted", zero_division=0)
    return loss, f1


def train_mlp(model: nn.Module,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              epochs: int = 50,
              lr: float = 1e-3,
              *,
              X_val: Optional[torch.Tensor] = None,
              y_val: Optional[torch.Tensor] = None,
              batch_size: int = 128,
              weight_decay: float = 1e-4,
              patience: int = 10,
              use_class_weights: bool = True,
              num_classes: int = 3,
              verbose: bool = False):
    """
    Train a CustomMLP with mini-batch SGD, optional class weighting and early
    stopping on a held-out validation split.

    Backward compatibility
    ----------------------
    If ``X_val`` / ``y_val`` are ``None`` the signature is identical to the
    original training loop (``train_mlp(model, X, y, epochs, lr)``) and the
    return value is ``(model, loss_history)`` — a plain list of per-epoch
    training losses.  Existing unit tests continue to pass.

    When a validation set is supplied, the return value becomes
    ``(model, TrainingHistory)`` containing train/val loss, train/val F1,
    the best epoch, and whether early stopping fired.

    Notes
    -----
    * Uses ``AdamW`` (Adam + decoupled weight decay) rather than plain Adam
      — the regularisation matters when we have only ~17k rows across 27
      features and an imbalanced target.
    * Class weights are inverse-frequency by default so minority grades (B, C)
      are not ignored.  The DOHMH dataset is ~78% A / 16% B / 6% C.
    """

    has_val = X_val is not None and y_val is not None

    if use_class_weights:
        class_weights = _compute_class_weights(y_train, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = TrainingHistory()
    loss_only_history: list[float] = []  # legacy return value

    best_state: Optional[dict] = None
    best_val_f1 = -1.0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        epoch_train_loss = running_loss / max(n_seen, 1)
        loss_only_history.append(epoch_train_loss)
        history.train_loss.append(epoch_train_loss)

        _, train_f1 = _evaluate(model, X_train, y_train, criterion)
        history.train_f1.append(train_f1)

        if has_val:
            val_loss, val_f1 = _evaluate(model, X_val, y_val, criterion)
            history.val_loss.append(val_loss)
            history.val_f1.append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                history.best_epoch = epoch
                history.best_val_f1 = val_f1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    history.stopped_early = True
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1} "
                              f"(best val F1 = {best_val_f1:.4f} "
                              f"@ epoch {history.best_epoch + 1})")
                    break

        if verbose and (epoch + 1) % 10 == 0:
            tail = f", val F1={history.val_f1[-1]:.4f}" if has_val else ""
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"train loss={epoch_train_loss:.4f}, train F1={train_f1:.4f}{tail}")

    # Restore best weights (only meaningful when a val set was provided)
    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    if has_val:
        return model, history
    return model, loss_only_history


def evaluate_mlp(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
                 class_names: Optional[list] = None,
                 return_details: bool = False):
    """
    Compute weighted F1 on the test set.  When ``return_details=True``, also
    returns the confusion matrix, per-class precision/recall/F1, and the
    predicted labels.

    Backward compatibility: default return ``(f1, predictions)`` matches the
    original signature used by ``tests/test_custom_mlp.py``.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()
    f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

    if not return_details:
        return f1, preds

    cm = confusion_matrix(y_true, preds, labels=list(range(3)))
    if class_names is None:
        class_names = ["A", "B", "C"]
    report = classification_report(
        y_true, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "weighted_f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": preds,
        "probabilities": probs,
    }


# ---------------------------------------------------------------------------
# 3. Hyperparameter search (justifies our choice of hidden/lr/dropout)
# ---------------------------------------------------------------------------

def hyperparameter_search(X_train: torch.Tensor,
                          y_train: torch.Tensor,
                          X_val: torch.Tensor,
                          y_val: torch.Tensor,
                          *,
                          hidden_dims=(64, 128, 256),
                          learning_rates=(5e-4, 1e-3, 5e-3),
                          dropouts=(0.1, 0.3, 0.5),
                          epochs: int = 40,
                          batch_size: int = 128,
                          patience: int = 8,
                          seed: int = 42,
                          progress_callback=None) -> list:
    """
    Grid search over (hidden_dim, lr, dropout) on a fixed train/val split.

    Returns a list of result dicts sorted by val F1 (descending).  Each entry
    contains the hyperparameters, best validation F1, best epoch reached, and
    whether early stopping fired.

    This exists specifically so we can justify our chosen hyperparameters to
    the TA rather than hard-coding magic numbers.

    ``progress_callback(i, total, params)`` is invoked between runs so the
    Streamlit page can update a progress bar.
    """
    input_dim = X_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    configs = [
        (h, lr, d)
        for h in hidden_dims
        for lr in learning_rates
        for d in dropouts
    ]

    results = []
    for idx, (h, lr, d) in enumerate(configs):
        if progress_callback is not None:
            progress_callback(idx, len(configs), {"hidden_dim": h, "lr": lr, "dropout": d})

        torch.manual_seed(seed)
        model = CustomMLP(input_dim=input_dim, hidden_dim=h,
                          output_dim=num_classes, dropout=d)
        _, history = train_mlp(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs, lr=lr,
            batch_size=batch_size, patience=patience,
            use_class_weights=True, verbose=False,
        )
        results.append({
            "hidden_dim": h,
            "lr": lr,
            "dropout": d,
            "best_val_f1": history.best_val_f1,
            "best_epoch": history.best_epoch + 1,
            "stopped_early": history.stopped_early,
            "final_train_f1": history.train_f1[-1] if history.train_f1 else float("nan"),
        })

    if progress_callback is not None:
        progress_callback(len(configs), len(configs), None)

    results.sort(key=lambda r: r["best_val_f1"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 4. Counterfactual explanation (what would flip this restaurant to Grade A?)
# ---------------------------------------------------------------------------

def find_counterfactual(model: nn.Module,
                        base_features: torch.Tensor,
                        target_class: int = 0,
                        steps: int = 150,
                        lr: float = 0.1,
                        l2_penalty: float = 0.5,
                        mutable_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Find the minimum perturbation of ``base_features`` required to flip the
    model's prediction to ``target_class``.

    We freeze the model weights and run gradient descent on the *input*
    tensor, minimising cross-entropy to the target class plus an L2 penalty
    keeping the counterfactual close to the original restaurant profile.

    Parameters
    ----------
    mutable_mask : Optional[torch.Tensor]
        Boolean mask of shape ``base_features.shape`` indicating which
        features the optimiser is allowed to change.  One-hot borough /
        cuisine features should typically be held fixed, since changing them
        doesn't correspond to a real-world intervention the restaurant owner
        can make.  If ``None`` all features are treated as mutable.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    opt_features = base_features.clone().detach()
    opt_features.requires_grad = True

    optimizer = optim.Adam([opt_features], lr=lr)
    criterion = nn.CrossEntropyLoss()
    target_tensor = torch.tensor([target_class], dtype=torch.long)

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(opt_features)
        loss = criterion(output, target_tensor)
        reg = l2_penalty * torch.sum((opt_features - base_features) ** 2)
        total = loss + reg
        total.backward()

        if mutable_mask is not None:
            # Zero out gradients on frozen features
            with torch.no_grad():
                opt_features.grad = opt_features.grad * mutable_mask.float()

        optimizer.step()

    for param in model.parameters():
        param.requires_grad = True

    return opt_features.detach()
