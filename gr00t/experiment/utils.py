# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import logging
import math
import shutil

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


def _metric_improved(current: float, best: float, greater_is_better: bool) -> bool:
    return (current > best) if greater_is_better else (current < best)


def _wandb_log_eval_running_best(
    *,
    metric_name: str,
    greater_is_better: bool,
    state: TrainerState,
    current: float | None,
    best_value: float,
    best_step: int | None,
    improved: bool,
) -> None:
    """Log running-best validation metrics for W&B dashboards (same step as HF `global_step`)."""
    if not state.is_world_process_zero or current is None:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    cur_f = float(current)
    if not math.isfinite(cur_f) or not math.isfinite(float(best_value)):
        return
    short = metric_name[5:] if metric_name.startswith("eval_") else metric_name
    gap = (best_value - cur_f) if greater_is_better else (cur_f - best_value)
    payload = {
        f"eval/best_{short}": float(best_value),
        "eval/best_global_step": float(best_step if best_step is not None else state.global_step),
        "eval/new_best": 1.0 if improved else 0.0,
        "eval/gap_to_best": float(gap),
    }
    wandb.log(payload, step=int(state.global_step))


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        processor_dir: Path | None = None,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self,
        metric_name: str,
        greater_is_better: bool = True,
        exp_cfg_dir: Path | None = None,
        log_running_best_to_wandb: bool = True,
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None
        self._best_metric_step: int | None = None
        self.log_running_best_to_wandb = log_running_best_to_wandb

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if not state.is_world_process_zero or metrics is None:
            return

        raw_metric = metrics.get(self.metric_name, None)
        current_metric: float | None = None
        if raw_metric is not None:
            try:
                current_metric = float(raw_metric)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping best checkpoint / W&B extras: %s is not numeric (%r)",
                    self.metric_name,
                    raw_metric,
                )
                return
            if not math.isfinite(current_metric):
                logger.warning(
                    "Skipping best checkpoint / W&B extras: %s is not finite (%s)",
                    self.metric_name,
                    raw_metric,
                )
                return

        improved = False
        if current_metric is not None and _metric_improved(
            current_metric, float(self.best_metric), self.greater_is_better
        ):
            improved = True
            self.best_metric = current_metric
            self._best_metric_step = int(state.global_step)
            best_checkpoint_dir = (
                Path(args.output_dir)
                / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
            )
            best_checkpoint_dir.mkdir(exist_ok=True)
            model.save_pretrained(best_checkpoint_dir)
            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            print(
                f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
            )

            if self._best_checkpoint_dir is not None and Path(self._best_checkpoint_dir).exists():
                shutil.rmtree(self._best_checkpoint_dir)

            self._best_checkpoint_dir = str(best_checkpoint_dir)

        if self.log_running_best_to_wandb and current_metric is not None:
            _wandb_log_eval_running_best(
                metric_name=self.metric_name,
                greater_is_better=self.greater_is_better,
                state=state,
                current=current_metric,
                best_value=float(self.best_metric),
                best_step=self._best_metric_step,
                improved=improved,
            )


class EvalRunningBestWandbCallback(TrainerCallback):
    """Logs running-best validation metrics to W&B when no ``BestMetricCheckpointCallback`` is used."""

    def __init__(
        self,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self._best_metric_step: int | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if metrics is None:
            return
        raw = metrics.get(self.metric_name, None)
        if raw is None:
            return
        try:
            cur_f = float(raw)
        except (TypeError, ValueError):
            return
        if not math.isfinite(cur_f):
            return
        best_f = float(self.best_metric)
        improved = _metric_improved(cur_f, best_f, self.greater_is_better)
        if improved:
            self.best_metric = cur_f
            self._best_metric_step = int(state.global_step)
        _wandb_log_eval_running_best(
            metric_name=self.metric_name,
            greater_is_better=self.greater_is_better,
            state=state,
            current=cur_f,
            best_value=float(self.best_metric),
            best_step=self._best_metric_step,
            improved=improved,
        )
