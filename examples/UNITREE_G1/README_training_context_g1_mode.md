# G1 Mode Training Context

This note summarizes the current training decision for the first Unitree G1
iteration.

The goal is training first, not deployment yet:

```text
GR00T VLA -> future whole-body WBC chunk
```

Deployment can later decide how to route that output:

```text
body slice -> SONIC g1 encoder/body path
hand slice -> direct hand command
```

## What The VLA Predicts

The VLA is fine-tuned to predict a future chunk of `action.wbc[43]`:

```text
action.wbc[t + 0]
action.wbc[t + 5]
action.wbc[t + 10]
...
action.wbc[t + 45]
```

That is:

```text
target shape = [10, 43]
```

The dataset is 50 FPS, so the last predicted action is `0.9s` in the future.

The 43 DoF action is split as:

```text
0:6    left_leg
6:12   right_leg
12:15  waist
15:22  left_arm
22:29  left_hand
29:36  right_arm
36:43  right_hand
```

For training, legs, waist, and arms use relative joint actions. Hands use
absolute joint actions.

## Dataset Columns Used

Inputs:

```text
observation.images.ego_view        video, 480 x 640 x 3
task_index                         mapped to annotation.human.task_description
observation.state[43]              current whole-body joint state
observation.root_orientation[4]    current base quaternion
observation.projected_gravity[3]   gravity/orientation stability signal
```

Target:

```text
action.wbc[43]                     future whole-body WBC actions
```

Not used in this first training iteration:

```text
observation.eef_state
observation.cpp_rotation_offset
observation.init_base_quat
action.motion_token
teleop.*
```

## Dataset Double Check

The local dataset checked was:

```text
outputs/push_peluche_processed_and_cleaned
```

Observed metadata:

```text
episodes: 48
frames: 35,816
fps: 50
ego video: 480 x 640 x 3
observation.state: 43
action.wbc: 43
observation.root_orientation: 4
observation.projected_gravity: 3
```

With `delta_indices = [0, 5, 10, ..., 45]`, each episode still has enough
frames. The shortest episode has 276 frames, and the effective usable total is
33,656 timesteps.

With seed `42` and `eval_set_split_ratio = 0.2`, the split is:

```text
train episodes: 38
eval episodes: 10
```

The split is episode-level, not frame-level, so validation episodes are held
out cleanly.

Important caveat: before training, copy the checked-in WBC modality file into
the dataset:

```bash
cp examples/UNITREE_G1/modality_wholebody_wbc.json \
  ./data/push_peluche_processed_and_cleaned/meta/modality.json
```

The original dataset `meta/modality.json` still contains many `teleop.*`
action mappings. The WBC modality file is what makes GR00T read `action.wbc`
as `left_leg`, `right_leg`, `waist`, arms, and hands.

## Model And Dimensionality

The base model to fine-tune is:

```text
nvidia/GR00T-N1.7-3B
```

In this repo's NVIDIA N1.7 model config:

```text
max_state_dim  = 132
max_action_dim = 132
action_horizon = 40
```

Our configured training dimensions are:

```text
state_dim  = 50
action_dim = 43
horizon    = 10
```

So we are safely under the model limits:

```text
50 < 132
43 < 132
10 < 40
```

GR00T pads smaller embodiments to the model maximum internally:

```text
state real:  [1, 50]   -> padded to [1, 132]
action real: [10, 43]  -> padded to [40, 132]
```

The action mask ensures loss is computed only on the valid region:

```text
valid:   action[0:10, 0:43]
padding: ignored
```

If a downloaded checkpoint or processor ever came with an older
`max_action_dim`, training would fail early. Confirm after launch by checking
the saved `final_processor_config.json` in the output directory.

## Current Verdict

For this first training iteration, the setup is coherent:

```text
input:
  ego image + task text + robot state/orientation context

target:
  future whole-body action.wbc chunk [10, 43]

validation:
  80/20 held-out episode split with eval_loss

model:
  nvidia/GR00T-N1.7-3B

dimensionality:
  safe under N1.7 max_state_dim, max_action_dim, and action_horizon
```

This does not yet train `motion_anchor_orientation_10frame_step5` or
`action.base_quat_target`. Those remain deployment/future-data-collection
topics.
