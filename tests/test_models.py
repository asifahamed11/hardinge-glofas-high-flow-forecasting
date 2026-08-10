from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from hardinge_high_flow.config import load_config
from hardinge_high_flow.models import (
    make_deep_model,
    refit_deep_model,
    train_deep_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_neural_model_is_refit_on_the_complete_training_block() -> None:
    config = copy.deepcopy(load_config(PROJECT_ROOT / "configs" / "default.yaml"))
    config["experiment"].update(
        {
            "hidden_size": 8,
            "recurrent_layers": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "maximum_epochs": 2,
            "patience": 2,
            "imbalance_strategy": "none",
            "early_stopping_metric": "average_precision",
        }
    )
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(24, 4, 3)).astype(np.float32)
    labels = np.tile(np.array([0, 1], dtype=np.int8), 12)
    selection_model = make_deep_model("gru", 3, config, 42)
    selection = train_deep_model(
        selection_model,
        inputs[:16],
        labels[:16],
        inputs[16:],
        labels[16:],
        config,
        42,
        torch.device("cpu"),
    )
    refit_model = make_deep_model("gru", 3, config, 42)
    state, losses = refit_deep_model(
        refit_model,
        inputs,
        labels,
        config,
        42,
        torch.device("cpu"),
        selection.learning_rates[: selection.best_epoch],
    )
    assert state
    assert len(losses) == selection.best_epoch
