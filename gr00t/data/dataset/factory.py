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

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _split_episode_indices(
        num_episodes: int, eval_split_ratio: float, seed: int
    ) -> tuple[list[int], list[int]]:
        if num_episodes < 2:
            raise ValueError("At least 2 episodes are required to create a validation split.")
        if not 0 < eval_split_ratio < 1:
            raise ValueError(
                "eval_set_split_ratio must be between 0 and 1 when evaluation is enabled."
            )

        eval_count = int(num_episodes * eval_split_ratio + 0.5)
        eval_count = max(1, min(num_episodes - 1, eval_count))

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(num_episodes).tolist()
        eval_indices = sorted(shuffled_indices[:eval_count])
        train_indices = sorted(shuffled_indices[eval_count:])
        return train_indices, eval_indices

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        enable_eval = self.config.training.eval_strategy != "no"

        all_train_datasets = []
        all_train_weights = []
        all_eval_datasets = []
        all_eval_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            train_datasets = []
            eval_datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                modality_configs = self.config.data.modality_configs[embodiment_tag]
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()

                train_episode_indices = None
                eval_episode_indices = None
                if enable_eval:
                    episode_loader = LeRobotEpisodeLoader(
                        dataset_path=dataset_path,
                        modality_configs=modality_configs,
                        video_backend=self.config.data.video_backend,
                    )
                    train_episode_indices, eval_episode_indices = self._split_episode_indices(
                        num_episodes=len(episode_loader),
                        eval_split_ratio=self.config.training.eval_set_split_ratio,
                        seed=self.config.data.seed,
                    )
                    print(
                        f"Split dataset {dataset_path}: {len(train_episode_indices)} train episodes, "
                        f"{len(eval_episode_indices)} validation episodes"
                    )

                train_dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=modality_configs,
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    episode_indices=train_episode_indices,
                )
                train_datasets.append(train_dataset)

                if enable_eval:
                    eval_dataset = ShardedSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=modality_configs,
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        episode_indices=eval_episode_indices,
                    )
                    eval_datasets.append(eval_dataset)

            train_dataset_lengths = np.array([len(dataset) for dataset in train_datasets])
            train_dataset_relative_lengths = train_dataset_lengths / train_dataset_lengths.sum()
            for dataset, relative_length in zip(train_datasets, train_dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_train_datasets.append(dataset)
                all_train_weights.append(weight)

            if enable_eval:
                eval_dataset_lengths = np.array([len(dataset) for dataset in eval_datasets])
                eval_dataset_relative_lengths = eval_dataset_lengths / eval_dataset_lengths.sum()
                for dataset, relative_length in zip(eval_datasets, eval_dataset_relative_lengths):
                    weight = relative_length * dataset_spec.mix_ratio
                    all_eval_datasets.append(dataset)
                    all_eval_weights.append(weight)

        train_dataset = ShardedMixtureDataset(
            datasets=all_train_datasets,
            weights=all_train_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

        eval_dataset = None
        if enable_eval:
            eval_dataset = ShardedMixtureDataset(
                datasets=all_eval_datasets,
                weights=all_eval_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
                set_processor_statistics=False,
            )

        return train_dataset, eval_dataset
