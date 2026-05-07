# Unitree G1 Mode Approx-Anchor Training on Vast.ai

Runbook for the first GR00T N1.7 training iteration toward:

```text
VLA -> SONIC g1 encoder/body + direct hands
```

This is the **training-only first version**: it uses the fields already present
in the dataset and trains GR00T to predict a future `action.wbc[43]` chunk.
Deployment can later split that chunk into SONIC `g1` body commands and direct
hand commands.

Uses:

- `observation.images.ego_view` -> ego camera
- `annotation.human.task_description` -> language instruction
- `observation.state[43]` -> current robot state
- `observation.projected_gravity[3]` -> gravity/orientation stability signal
- `observation.root_orientation[4]` -> current base quaternion
- `action.wbc[43]` -> future whole-body action chunk labels

Training target:

- `action.wbc[t + k][43]` for `k = 0, 5, 10, ..., 45`

For the next data collection pass, record `action.base_quat_target[4]` so the
deployment-side anchor orientation can be reconstructed from the true target
instead of approximated.

See `examples/UNITREE_G1/README_data_collection_g1_mode.md` for the data contract.

## 1. Vast.ai Instance

1. Add credit and upload your SSH public key in Vast.ai.
2. Rent a CUDA image with enough VRAM. **40GB+ is recommended** for fine-tuning; 24GB may require smaller batches.
3. SSH into the instance using the command from Vast.ai.
4. Optional, for large downloads:

```bash
export HF_HOME=/path/to/big_disk/.cache/huggingface
```

## 2. System Dependencies

Assume a clean SSH server with Python only:

```bash
apt-get update
apt-get install -y git git-lfs curl ca-certificates ffmpeg build-essential
git lfs install
```

Check GPU/CUDA:

```bash
nvidia-smi
```

## 3. Clone Repo

```bash
git clone --recurse-submodules https://github.com/BrikHMP18/Isaac-GR00T.git
cd Isaac-GR00T
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 4. Install Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv sync --python 3.10
```

Verify:

```bash
uv run python -c "import gr00t; print('GR00T installed successfully')"
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If fine-tuning fails with `CUDA_HOME is unset`:

```bash
export CUDA_HOME=/usr/local/cuda
```

## 5. Download Dataset

```bash
uv run hf download --repo-type dataset NONHUMAN-RESEARCH/push_peluche_processed_and_cleaned \
  --local-dir ./data/push_peluche_processed_and_cleaned
```

If private/gated:

```bash
uv run huggingface-cli login
```

## 6. Patch Dataset Modality

Use the checked-in whole-body WBC modality. No derived dataset is needed for this
training iteration; `g1_wholebody_config.py` samples the future `action.wbc`
chunk with 10 frames at step 5.

```bash
cp examples/UNITREE_G1/modality_wholebody_wbc.json \
  ./data/push_peluche_processed_and_cleaned/meta/modality.json
```

## 7. Generate Stats

```bash
uv run python gr00t/data/stats.py \
  --dataset-path ./data/push_peluche_processed_and_cleaned \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UNITREE_G1/g1_wholebody_config.py
```

## 8. Weights & Biases

The with-eval launcher logs to W&B only when `--use-wandb` is set.
W&B is used only for metrics and run monitoring. Checkpoints stay local under
`--output-dir`; models are not uploaded as W&B artifacts.

```bash
export WANDB_API_KEY=<your_key>
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
```

To disable W&B, remove `--use-wandb` from the fine-tuning command.

## 9. Fine-Tune with 80/20 Validation

```bash
uv run python gr00t/experiment/with_eval/launch_finetune_with_eval.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./data/push_peluche_processed_and_cleaned \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UNITREE_G1/g1_wholebody_config.py \
  --output-dir ./outputs/g1_mode_approx_anchor \
  --num-gpus 1 \
  --global-batch-size 32 \
  --max-steps 10000 \
  --save-steps 1000 \
  --use-wandb \
  --eval-strategy steps \
  --eval-steps 500 \
  --eval-set-split-ratio 0.2 \
  --eval-batch-size 2 \
  --track-best-eval-loss
```

This command is the with-eval training entrypoint. It trains on whole-body
`action.wbc[43]` chunks; deployment later decides how to route body and hand
slices.

This reports Hugging Face Trainer `eval_loss`: the same native GR00T training loss,
computed on the 20% validation episodes with `model.eval()` and no backward pass.
The best model by lowest `eval_loss` is saved under `checkpoint-*-best-eval_loss_*`.
Regular checkpoints are local and saved every `1000` steps by `--save-steps 1000`.

Expected W&B charts:

- `loss`: train loss
- `eval_loss`: validation loss on held-out episodes
- `learning_rate`
- `grad_norm`
- `train_accuracy` when token labels are present
