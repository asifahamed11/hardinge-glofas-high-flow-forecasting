"""Baseline and neural sequence models."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

# Required by CUDA for deterministic matrix multiplication.  It must be set
# before the first cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        binary_loss = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        target_probability = targets * probabilities + (1 - targets) * (
            1 - probabilities
        )
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        weights = alpha_factor * (1 - target_probability).pow(self.gamma)
        return (weights * binary_loss).mean()


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        return self.classifier(self.normalization(outputs[:, -1])).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.recurrent = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        return self.classifier(self.normalization(outputs[:, -1])).squeeze(-1)


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = max(32, hidden_size)
        self.convolution = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            channels,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        convolved = self.convolution(inputs.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.recurrent(convolved)
        return self.classifier(outputs[:, -1]).squeeze(-1)


class AttentionLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        weights = torch.softmax(self.attention(outputs), dim=1)
        context = (weights * outputs).sum(dim=1)
        return self.classifier(context).squeeze(-1)


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "cnn_lstm": CNNLSTMClassifier,
    "attention_lstm": AttentionLSTMClassifier,
}


@dataclass(frozen=True)
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    validation_average_precision: tuple[float, ...]
    learning_rates: tuple[float, ...]
    best_epoch: int
    parameter_count: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def make_deep_model(
    name: str,
    input_size: int,
    config: dict[str, Any],
    seed: int,
) -> nn.Module:
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported deep model: {name}")
    set_global_seed(seed)
    experiment = config["experiment"]
    return MODEL_CLASSES[name](
        input_size=input_size,
        hidden_size=int(experiment["hidden_size"]),
        layers=int(experiment["recurrent_layers"]),
        dropout=float(experiment["dropout"]),
    )


def make_classical_model(name: str, seed: int) -> Any:
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
            solver="liblinear",
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unsupported classical model: {name}")


def _loss_and_sampler(
    labels: np.ndarray,
    config: dict[str, Any],
    generator: torch.Generator,
) -> tuple[nn.Module, WeightedRandomSampler | None]:
    experiment = config["experiment"]
    strategy = str(experiment["imbalance_strategy"])
    positive = max(int(labels.sum()), 1)
    negative = max(int((labels == 0).sum()), 1)
    sampler = None

    if strategy == "none":
        loss: nn.Module = nn.BCEWithLogitsLoss()
    elif strategy == "pos_weight":
        loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negative / positive], dtype=torch.float32)
        )
    elif strategy == "focal":
        loss = FocalLoss(
            alpha=float(experiment["focal_alpha"]),
            gamma=float(experiment["focal_gamma"]),
        )
    elif strategy == "sampler":
        loss = nn.BCEWithLogitsLoss()
        weights = np.where(labels == 1, negative / positive, 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported imbalance strategy: {strategy}")
    return loss, sampler


def train_deep_model(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> TrainingResult:
    set_global_seed(seed)
    experiment = config["experiment"]
    generator = torch.Generator().manual_seed(seed)
    loss_function, sampler = _loss_and_sampler(
        train_labels,
        config,
        generator,
    )
    loss_function = loss_function.to(device)

    train_dataset = TensorDataset(
        torch.as_tensor(train_inputs, dtype=torch.float32),
        torch.as_tensor(train_labels, dtype=torch.float32),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(experiment["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        num_workers=0,
    )
    validation_x = torch.as_tensor(
        validation_inputs,
        dtype=torch.float32,
        device=device,
    )
    validation_y = torch.as_tensor(
        validation_labels,
        dtype=torch.float32,
        device=device,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(experiment["learning_rate"]),
        weight_decay=float(experiment["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(experiment["patience"]) // 3),
        min_lr=1e-6,
    )
    validation_loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_average_precisions: list[float] = []
    learning_rates: list[float] = []
    stopping_metric = str(
        experiment.get("early_stopping_metric", "average_precision")
    )
    best_score = -float("inf")

    for epoch in range(1, int(experiment["maximum_epochs"]) + 1):
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        model.train()
        batch_losses = []
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(experiment["gradient_clip"]),
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x)
            validation_loss = float(
                validation_loss_function(
                    validation_logits,
                    validation_y,
                ).cpu()
            )
            validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
            validation_average_precision = float(
                average_precision_score(validation_labels, validation_probabilities)
            )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_average_precisions.append(validation_average_precision)
        scheduler.step(validation_loss)

        improved = (
            validation_average_precision > best_score + 1e-6
            if stopping_metric == "average_precision"
            else validation_loss < best_loss - 1e-6
        )
        if improved:
            best_loss = validation_loss
            best_score = validation_average_precision
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(experiment["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Deep model training produced no checkpoint.")
    model.load_state_dict(best_state)
    state_on_cpu = {name: tensor.detach().cpu() for name, tensor in best_state.items()}
    return TrainingResult(
        state_dict=state_on_cpu,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        validation_average_precision=tuple(validation_average_precisions),
        learning_rates=tuple(learning_rates),
        best_epoch=best_epoch,
        parameter_count=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def refit_deep_model(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    learning_rates: tuple[float, ...],
) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
    """Refit a selected neural architecture on the complete training block.

    The inner temporal holdout chooses the number of epochs and learning-rate
    schedule.  This fresh fit then gives the neural model the same complete
    training period used by classical models without consulting validation or
    test outcomes.
    """
    if not learning_rates:
        raise ValueError("At least one refit learning rate is required.")
    set_global_seed(seed)
    experiment = config["experiment"]
    generator = torch.Generator().manual_seed(seed)
    loss_function, sampler = _loss_and_sampler(train_labels, config, generator)
    loss_function = loss_function.to(device)
    dataset = TensorDataset(
        torch.as_tensor(train_inputs, dtype=torch.float32),
        torch.as_tensor(train_labels, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(experiment["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        num_workers=0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rates[0]),
        weight_decay=float(experiment["weight_decay"]),
    )
    losses: list[float] = []
    for learning_rate in learning_rates:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)
        model.train()
        batch_losses = []
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(experiment["gradient_clip"]),
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))

    state_on_cpu = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    return state_on_cpu, tuple(losses)


def predict_deep_probabilities(
    model: nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probabilities).astype(float)
