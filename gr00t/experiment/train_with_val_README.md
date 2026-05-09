# Fine-Tuning With Validation

This branch adds optional validation to `launch_finetune.py`.

By default, training behaves like the original script. Validation is enabled only when
`--eval-strategy` is set to `steps` or `epoch`.

## Setup

Clone the repository and switch to this branch:

```bash
git clone https://github.com/BrikHMP18/Isaac-GR00T.git
cd Isaac-GR00T
git checkout train-with-val
```

Sync the environment with `uv`:

```bash
uv sync --all-extras
```

Log in to Hugging Face and W&B:

```bash
uv run huggingface-cli login
uv run wandb login
```

Download the `push_peluche` dataset:

```bash
mkdir -p /workspace/datasets/push_peluche

uv run huggingface-cli download NONHUMAN-RESEARCH/push_peluche_processed_and_cleaned \
    --repo-type dataset \
    --local-dir /workspace/datasets/push_peluche
```

## Vast.ai 2-GPU Launch

If you need to clear an old crashed run, stop previous Python training processes first. This
kills every Python process in the container, including notebooks or shells using Python:

```bash
pkill -9 python || true
```

Set the GPU and NCCL environment variables:

```bash
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

Launch with `torchrun` so both GPUs get a distributed worker. BF16 is enabled by default in
the training config, so no extra `--bf16` flag is needed.

```bash
uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /workspace/datasets/push_peluche \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus $NUM_GPUS \
    --output-dir /workspace/output_push_peluche_v6_max \
    --save-total-limit 5 \
    --save-steps 1000 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 32 \
    --eval-strategy steps \
    --eval-steps 1000 \
    --eval-set-split-ratio 0.2 \
    --eval-batch-size 1 \
    --save-best-eval-metric-name eval_loss \
    --no-save-best-eval-metric-greater-is-better \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 16
```

## What This Does

- Splits the dataset by episode, not by frame or timestep.
- Uses `80%` of episodes for training and `20%` for validation in the example above.
- Rounds the validation episode count to the nearest integer.
- Logs validation metrics such as `eval_loss` to WandB when `--use-wandb` is enabled.
- Saves the best checkpoint according to `eval_loss` when the best-checkpoint flags are set.
- Logs running best validation metrics to W&B (`eval/best_loss`, `eval/new_best`,
  `eval/gap_to_best`, and `eval/best_global_step`).

For loss metrics, lower is better, so use:

```bash
--save-best-eval-metric-name eval_loss \
--no-save-best-eval-metric-greater-is-better
```
