# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from torch.utils.data import get_worker_info

from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset


class ShardedMixtureDatasetWithEval(ShardedMixtureDataset):
    """Mixture dataset whose evaluation iterator is finite."""

    def __iter__(self):
        if self.training:
            yield from super().__iter__()
            return

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        schedule = []
        for i, shard in enumerate(self.shard_sampling_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                schedule.append(shard)

        rng = np.random.default_rng(self.seed)
        for dataset_index, shard_index in schedule:
            shard = self.datasets[dataset_index].get_shard(shard_index)
            indices = np.arange(len(shard))
            rng.shuffle(indices)
            for index in indices:
                yield shard[index]
