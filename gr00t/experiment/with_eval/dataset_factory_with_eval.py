# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier
from gr00t.experiment.with_eval.sharded_mixture_dataset_with_eval import (
    ShardedMixtureDatasetWithEval,
)
from gr00t.experiment.with_eval.sharded_single_step_dataset_with_eval import (
    ShardedSingleStepDatasetWithEval,
)


class DatasetFactoryWithEval:
    """Factory that creates train/eval sharded datasets using episode-level splits."""

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDatasetWithEval, ShardedMixtureDatasetWithEval | None]:
        train_datasets = []
        train_weights = []
        eval_datasets = []
        eval_weights = []

        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing train/eval datasets",
        ):
            train_group = []
            eval_group = []

            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"

                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()

                train_episode_indices, eval_episode_indices = self._split_episode_indices(
                    dataset_path
                )

                train_group.append(
                    self._create_dataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        episode_indices=train_episode_indices,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                    )
                )

                if len(eval_episode_indices) > 0:
                    eval_group.append(
                        self._create_dataset(
                            dataset_path=dataset_path,
                            embodiment_tag=EmbodimentTag(embodiment_tag),
                            episode_indices=eval_episode_indices,
                            episode_sampling_rate=1.0,
                        )
                    )

            train_lengths = np.array([len(dataset) for dataset in train_group])
            train_relative_lengths = train_lengths / train_lengths.sum()
            for dataset, relative_length in zip(train_group, train_relative_lengths):
                train_datasets.append(dataset)
                train_weights.append(relative_length * dataset_spec.mix_ratio)

            if eval_group:
                eval_lengths = np.array([len(dataset) for dataset in eval_group])
                eval_relative_lengths = eval_lengths / eval_lengths.sum()
                for dataset, relative_length in zip(eval_group, eval_relative_lengths):
                    eval_datasets.append(dataset)
                    eval_weights.append(relative_length * dataset_spec.mix_ratio)

        train_dataset = ShardedMixtureDatasetWithEval(
            datasets=train_datasets,
            weights=train_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

        eval_dataset = None
        if eval_datasets and self.config.training.eval_strategy != "no":
            eval_dataset = ShardedMixtureDatasetWithEval(
                datasets=eval_datasets,
                weights=eval_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )
            # Keep validation normalized with the training statistics, matching standard
            # offline-validation practice and avoiding leakage from held-out episodes.
            train_stats = train_dataset.get_dataset_statistics()
            processor.set_statistics(
                train_stats,
                override=self.config.data.override_pretraining_statistics,
            )
            for dataset in train_dataset.datasets + eval_dataset.datasets:
                dataset.set_processor(processor)

        return train_dataset, eval_dataset

    def _create_dataset(
        self,
        dataset_path: str,
        embodiment_tag: EmbodimentTag,
        episode_indices: np.ndarray,
        episode_sampling_rate: float,
    ) -> ShardedSingleStepDatasetWithEval:
        return ShardedSingleStepDatasetWithEval(
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_configs=self.config.data.modality_configs[embodiment_tag.value],
            episode_indices=episode_indices,
            video_backend=self.config.data.video_backend,
            shard_size=self.config.data.shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=self.config.data.seed,
            allow_padding=self.config.data.allow_padding,
        )

    def _split_episode_indices(self, dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
        episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
        with open(episodes_path, "r") as f:
            episode_indices = np.array([json.loads(line)["episode_index"] for line in f], dtype=int)

        rng = np.random.default_rng(self.config.data.seed)
        shuffled = rng.permutation(episode_indices)
        eval_ratio = self.config.training.eval_set_split_ratio

        if eval_ratio <= 0 or len(shuffled) < 2:
            return np.sort(shuffled), np.array([], dtype=int)

        eval_count = max(1, int(round(len(shuffled) * eval_ratio)))
        eval_count = min(eval_count, len(shuffled) - 1)
        eval_indices = np.sort(shuffled[:eval_count])
        train_indices = np.sort(shuffled[eval_count:])
        print(
            f"Episode split for {dataset_path}: "
            f"{len(train_indices)} train / {len(eval_indices)} eval"
        )
        return train_indices, eval_indices
