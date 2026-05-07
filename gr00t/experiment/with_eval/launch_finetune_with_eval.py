# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch N1.7 fine-tuning with an episode-level validation split."""

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run
from gr00t.experiment.with_eval.dataset_factory_with_eval import DatasetFactoryWithEval
from gr00t.experiment.with_eval.finetune_config_with_eval import FinetuneConfigWithEval


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def patch_dataset_factory() -> None:
    """Use the eval-capable factory only for this launcher."""
    import gr00t.model.gr00t_n1d7.setup as n1d7_setup

    n1d7_setup.DatasetFactory = DatasetFactoryWithEval


def configure_wandb_metrics_only() -> None:
    """Keep W&B as a metrics dashboard, not a model/artifact store."""
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(FinetuneConfigWithEval, description=__doc__)
    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    patch_dataset_factory()
    if ft_config.use_wandb:
        configure_wandb_metrics_only()

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.eval_strategy = ft_config.eval_strategy
    config.training.eval_steps = ft_config.eval_steps
    config.training.eval_set_split_ratio = ft_config.eval_set_split_ratio
    config.training.eval_batch_size = ft_config.eval_batch_size
    if ft_config.track_best_eval_loss and ft_config.eval_strategy != "no":
        config.training.save_best_eval_metric_name = "eval_loss"
        config.training.save_best_eval_metric_greater_is_better = False

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
