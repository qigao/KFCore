from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

FEATURE_COUNT = 78
HIDDEN_SIZE = 64
NUM_LAYERS = 2
GESTURE_COUNT = 8
PHASE_COUNT = 4

GESTURE_CLASSES = (
    "none",
    "wave",
    "swipe_left",
    "swipe_right",
    "grab",
    "release",
    "point",
    "click",
)
PHASE_CLASSES = ("idle", "start", "active", "end")


class TemporalGestureGru(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.gesture_head = nn.Linear(HIDDEN_SIZE, GESTURE_COUNT)
        self.phase_head = nn.Linear(HIDDEN_SIZE, PHASE_COUNT)

    def forward_sequence(
        self, features: torch.Tensor, hidden_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, hidden_out = self.gru(features, hidden_in)
        return (
            self.gesture_head(sequence),
            self.phase_head(sequence),
            hidden_out,
        )

    def forward(
        self, features: torch.Tensor, hidden_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_sequence(features, hidden_in)


class StreamingExportWrapper(nn.Module):
    """Expose the exact fixed-shape KFCore V1 streaming tensor contract."""

    def __init__(self, model: TemporalGestureGru) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, features: torch.Tensor, hidden_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = features.unsqueeze(1)  # [1,78] -> [1,1,78]
        gesture, phase, hidden_out = self.model.forward_sequence(sequence, hidden_in)
        return gesture[:, 0, :], phase[:, 0, :], hidden_out


@dataclass(frozen=True)
class CheckpointMetadata:
    feature_count: int = FEATURE_COUNT
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    gesture_count: int = GESTURE_COUNT
    phase_count: int = PHASE_COUNT
