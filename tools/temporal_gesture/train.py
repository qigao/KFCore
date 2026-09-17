from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from dataset import SequenceExample, load_sequences, subjects
from model import (
    GESTURE_CLASSES,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PHASE_CLASSES,
    TemporalGestureGru,
    checkpoint_config,
)


def _tensorize(example: SequenceExample):
    features = torch.tensor(example.features, dtype=torch.float32).unsqueeze(0)
    gestures = torch.tensor(example.gesture_labels, dtype=torch.long)
    phases = torch.tensor(example.phase_labels, dtype=torch.long)
    hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32)
    return features, gestures, phases, hidden


def evaluate(model: TemporalGestureGru, examples: Sequence[SequenceExample]):
    model.eval()
    total_loss = 0.0
    total_frames = 0
    gesture_correct = 0
    phase_correct = 0
    with torch.no_grad():
        for example in examples:
            features, gestures, phases, hidden = _tensorize(example)
            gesture_logits, phase_logits, _ = model.forward_sequence(features, hidden)
            gesture_logits = gesture_logits[0]
            phase_logits = phase_logits[0]
            loss = F.cross_entropy(gesture_logits, gestures) + F.cross_entropy(phase_logits, phases)
            frames = gestures.numel()
            total_loss += float(loss.item()) * frames
            total_frames += frames
            gesture_correct += int((gesture_logits.argmax(dim=-1) == gestures).sum().item())
            phase_correct += int((phase_logits.argmax(dim=-1) == phases).sum().item())
    if total_frames == 0:
        raise ValueError("evaluation set contains no frames")
    return (
        total_loss / total_frames,
        gesture_correct / total_frames,
        phase_correct / total_frames,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train KFCore Temporal Gesture GRU V1")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    if args.epochs <= 0 or args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise SystemExit("epochs/learning-rate/weight-decay are invalid")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_examples = load_sequences(args.train)
    validation_examples = load_sequences(args.validation)
    if not train_examples or not validation_examples:
        raise SystemExit("train and validation sets must both contain at least one sequence")

    train_subjects = subjects(train_examples)
    validation_subjects = subjects(validation_examples)
    overlap = train_subjects & validation_subjects
    if overlap:
        raise SystemExit(
            "subject leakage between train and validation sets: " + ", ".join(sorted(overlap))
        )
    if not train_subjects or not validation_subjects:
        print("warning: subject_id is missing from at least one split; subject separation cannot be proven")

    model = TemporalGestureGru()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_validation_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_examples)
        running_loss = 0.0
        running_frames = 0
        for example in train_examples:
            features, gestures, phases, hidden = _tensorize(example)
            gesture_logits, phase_logits, _ = model.forward_sequence(features, hidden)
            gesture_logits = gesture_logits[0]
            phase_logits = phase_logits[0]
            loss = F.cross_entropy(gesture_logits, gestures) + F.cross_entropy(phase_logits, phases)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            frames = gestures.numel()
            running_loss += float(loss.item()) * frames
            running_frames += frames

        validation_loss, gesture_accuracy, phase_accuracy = evaluate(
            model, validation_examples
        )
        train_loss = running_loss / max(1, running_frames)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"gesture_accuracy={gesture_accuracy:.4f} phase_accuracy={phase_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": checkpoint_config(),
                    "gesture_classes": list(GESTURE_CLASSES),
                    "phase_classes": list(PHASE_CLASSES),
                    "feature_contract": "kfcore-temporal-gesture-features-v1",
                    "torch_version": torch.__version__,
                    "seed": args.seed,
                    "best_validation_loss": best_validation_loss,
                },
                args.output,
            )

    print(f"best checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
