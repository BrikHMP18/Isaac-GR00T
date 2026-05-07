# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import EmbodimentTag, ModalityConfig


class ShardedSingleStepDatasetWithEval(ShardedSingleStepDataset):
    """Single-step sharded dataset restricted to a fixed subset of episodes."""

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        episode_indices: list[int] | np.ndarray,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        self.episode_indices = np.array(sorted(episode_indices), dtype=int)
        super().__init__(
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=allow_padding,
        )

    def shard_dataset(self):
        """Create shards only from the selected episode subset."""
        shuffled_episode_indices = self.rng.permutation(self.episode_indices)
        num_splits = int(1 / self.episode_sampling_rate)

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        total_steps = np.sum(
            [self.get_effective_episode_length(idx) for idx in shuffled_episode_indices]
        ).astype(int)
        assert total_steps > 0, f"No valid timesteps found for dataset {self.dataset_path}"

        num_shards = np.ceil(total_steps / self.shard_size).astype(int)
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        for ep_idx in shuffled_episode_indices:
            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                if len(split_step_indices) == 0:
                    continue
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((int(ep_idx), split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        split_name = "selected"
        print(
            f"Generated {num_shards} {split_name} shards for {self.dataset_path} "
            f"from {len(self.episode_indices)} episodes"
        )
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, "
            f"shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths
