# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from gr00t.configs.finetune_config import FinetuneConfig


@dataclass
class FinetuneConfigWithEval(FinetuneConfig):
    """Fine-tuning config that enables an episode-level validation split."""

    eval_strategy: str = "steps"
    """Evaluation strategy used by Hugging Face Trainer: no, steps, or epoch."""

    eval_steps: int = 500
    """Run validation every N training steps when eval_strategy='steps'."""

    eval_set_split_ratio: float = 0.2
    """Fraction of episodes reserved for validation."""

    eval_batch_size: int = 2
    """Per-device validation batch size."""

    track_best_eval_loss: bool = True
    """If True, save the best model checkpoint according to eval_loss."""
