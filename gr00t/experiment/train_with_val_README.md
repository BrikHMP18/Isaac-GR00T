# Fine-Tuning With Validation

This branch adds optional validation to `launch_finetune.py`.

By default, training behaves like the original script. Validation is enabled only when
`--eval-strategy` is set to `steps` or `epoch`.

## Example

```bash
export NUM_GPUS=1

uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /workspace/datasets/push_peluche \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus $NUM_GPUS \
    --output-dir /workspace/output_push_peluche \
    --save-total-limit 5 \
    --save-steps 1000 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 16 \
    --eval-strategy steps \
    --eval-steps 1000 \
    --eval-set-split-ratio 0.2 \
    --eval-batch-size 2 \
    --save-best-eval-metric-name eval_loss \
    --no-save-best-eval-metric-greater-is-better \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
```

## What This Does

- Splits the dataset by episode, not by frame or timestep.
- Uses `80%` of episodes for training and `20%` for validation in the example above.
- Rounds the validation episode count to the nearest integer.
- Logs validation metrics such as `eval_loss` to WandB when `--use-wandb` is enabled.
- Saves the best checkpoint according to `eval_loss` when the best-checkpoint flags are set.

For loss metrics, lower is better, so use:

```bash
--save-best-eval-metric-name eval_loss \
--no-save-best-eval-metric-greater-is-better
```

