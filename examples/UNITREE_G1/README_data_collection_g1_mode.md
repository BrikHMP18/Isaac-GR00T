# G1 Mode Data Collection Notes

This note documents the data needed for the first SONIC-aware GR00T plan:

```text
VLA -> g1 encoder/body + direct hands
```

The body motion is routed through SONIC `g1` mode for stability. The hands are predicted
directly from the VLA and concatenated with the body action after the SONIC control decoder.

## Current Dataset Fields

Ignoring `teleop.*` fields that are not needed for the first formulation, the current dataset has:

```text
observation.state[43]
observation.eef_state[14]
action.wbc[43]
observation.root_orientation[4]
observation.projected_gravity[3]
observation.cpp_rotation_offset[4]
observation.init_base_quat[4]
teleop.delta_heading[1]
action.motion_token[64]
timestamp
frame_index
episode_index
index
task_index
```

`teleop.delta_heading` is technically under `teleop.*`, but it should be kept if we want to
reconstruct SONIC-style heading correction.

## Target Architecture

```text
ego image + language + current robot observation
        |
        v
      GR00T VLA
        |
        +--> body future motion command
        |      - motion_joint_positions_10frame_step5
        |      - motion_joint_velocities_10frame_step5
        |      - motion_anchor_orientation_10frame_step5
        |      v
        |    SONIC g1 encoder -> universal token -> robot control decoder -> body_action[29]
        |
        +--> hand_action[14]

final_action[43] = concat(body_action[29], hand_action[14])
```

The `g1` encoder expects:

```text
encoder_mode_4
motion_joint_positions_10frame_step5
motion_joint_velocities_10frame_step5
motion_anchor_orientation_10frame_step5
```

## What We Already Have

```text
✅ observation.state[43]
   Sirve para:
   - estado actual del robot
   - body_q[29] = state[:29]
   - left_hand_q[7] = state[22:29]
   - right_hand_q[7] = state[36:43]

✅ action.wbc[43]
   Sirve para construir labels:
   - motion_joint_positions_10frame_step5 = action.wbc[t+k][:29]
   - left_hand_action[7] = action.wbc[22:29]
   - right_hand_action[7] = action.wbc[36:43]
   - hand_action[14] = concat(action.wbc[22:29], action.wbc[36:43])

✅ observation.root_orientation[4]
   Sirve como base_quat actual.
   También permite aproximar future root quat con root_orientation[t+k].

✅ observation.projected_gravity[3]
   Sirve como orientación/estabilidad para input del VLA.

✅ observation.cpp_rotation_offset[4]
   Sirve para reconstruir mejor el heading correction estilo SONIC.

✅ observation.init_base_quat[4]
   Sirve para reconstruir mejor el heading correction estilo SONIC.

✅ teleop.delta_heading[1]
   Aunque sea teleop.*, sí conviene conservarlo para heading correction.

✅ action.motion_token[64]
   Ya existe, pero no es necesario para la primera formulación VLA -> g1 encoder.

✅ encoder_mode_4
   No hace falta grabarlo.
   Para modo g1 es fijo: [1, 0, 0, 0].

✅ motion_joint_velocities_10frame_step5
   No hace falta grabarlo inicialmente.
   Se deriva desde motion_joint_positions_10frame_step5.
```

## What We Can Derive

For each frame `t`, use future indices:

```text
k = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
```

Build body motion labels:

```text
motion_joint_positions_10frame_step5 =
  action.wbc[t + k][:29]
```

Build velocity labels from positions:

```text
motion_joint_velocities_10frame_step5 =
  finite_difference(motion_joint_positions_10frame_step5)
```

Build hand labels:

```text
hand_action[14] =
  concat(action.wbc[t][22:29], action.wbc[t][36:43])
```

## Orientation Match

There is no direct 1:1 dimensional match like:

```text
observation.state[43] <-> action.wbc[43]
```

For orientation, the current observation and SONIC command use different representations:

```text
observation.root_orientation[4]
= current absolute base quaternion
= [qw, qx, qy, qz]

motion_anchor_orientation[6]
= relative target orientation in 6D rotation format
= first two columns of a rotation matrix
```

The conceptual match is:

```text
current_base_quat = observation.root_orientation[t]
target_root_quat  = action.base_quat_target[t]

relative_quat = inverse(current_base_quat) * target_root_quat
motion_anchor_orientation[6] = quat_to_rot6d(relative_quat)
```

In simple terms:

```text
root_orientation = cómo está orientado el robot ahora
motion_anchor_orientation = hacia qué orientación relativa quiero llevarlo
```

## Minimum Additional Field To Record

For the first serious `g1` mode iteration, the minimum additional field is:

```text
❌ action.base_quat_target[4]
   Target root/base orientation del motion/control.
   Sirve para construir motion_anchor_orientation_10frame_step5 sin depender de
   root_orientation[t+k] como aproximación.
```

With this field, we can construct:

```text
motion_anchor_orientation_10frame_step5[60]
```

offline:

```text
current_base_quat = observation.root_orientation[t]
target_root_quat  = action.base_quat_target[t+k]

anchor_quat[k] = inverse(current_base_quat) * target_root_quat
anchor_6d[k] = quat_to_rot6d(anchor_quat[k])
```

## Optional Fields

These are useful, but not required for the minimal first iteration:

```text
❌ action.motion_anchor_orientation[6]
   Anchor orientation ya calculado para el frame actual.
   Es opcional si grabamos action.base_quat_target[4].

❌ action.motion_anchor_orientation_10frame_step5[60]
   Ideal, pero no obligatorio.
   Si lo grabamos directo, evitamos reconstruir ventanas offline.
```

## Not Recording For Now

These can be derived or are already embedded in existing fields:

```text
➖ observation.body_dq[29]
   Se puede derivar desde posiciones para esta primera iteración.

➖ observation.left_hand_dq[7]
   No crítico para primera iteración.

➖ observation.right_hand_dq[7]
   No crítico para primera iteración.

➖ observation.base_ang_vel[3]
   Útil para control decoder puro, pero no imprescindible para construir input g1 encoder.

➖ action.body_q_target[29]
   Ya está en action.wbc[:29].

➖ action.left_hand[7]
   Ya está en action.wbc[22:29].

➖ action.right_hand[7]
   Ya está en action.wbc[36:43].
```

## Future Exporter Upgrade

Use this section as context for a later session that modifies:

```text
external_dependencies/GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py
```

The relevant function is:

```text
GrootDataCollector._add_data_frame_sonic()
```

Today it builds:

```text
observation.state = body_q + left_hand_q + right_hand_q
action.wbc = last_action + last_left_hand_action + last_right_hand_action
```

and then calls:

```text
GrootDataCollector._add_cpp_state_features(frame_data, proprio)
```

where `observation.root_orientation`, `observation.projected_gravity`,
`observation.cpp_rotation_offset`, `observation.init_base_quat`,
`teleop.delta_heading`, and `action.motion_token` are added.

For the next collection pass, add the minimum missing field:

```text
❌ action.base_quat_target[4]
```

The C++ ZMQ stream already documents/publishes a target orientation concept as:

```text
base_quat_target
```

so the Python exporter should read it from `proprio` when available:

```python
if "base_quat_target" in proprio:
    frame_data["action.base_quat_target"] = np.asarray(
        proprio["base_quat_target"], dtype=np.float64
    )
else:
    frame_data["action.base_quat_target"] = frame_data["observation.root_orientation"].copy()
```

Preferred behavior:

```text
✅ If base_quat_target exists, save it as action.base_quat_target[4].
✅ If it does not exist, either fail loudly during collection or use current root orientation
   only as a temporary fallback for debugging.
```

After adding this field, update dataset metadata/modality so the derived dataset
can build:

```text
motion_anchor_orientation_10frame_step5[60]
```

from:

```text
current_base_quat = observation.root_orientation[t]
target_root_quat = action.base_quat_target[t+k]
anchor_quat[k] = inverse(current_base_quat) * target_root_quat
anchor_6d[k] = quat_to_rot6d(anchor_quat[k])
```

Optional exporter fields if we want to avoid offline reconstruction later:

```text
➖ action.motion_anchor_orientation[6]
➖ action.motion_anchor_orientation_10frame_step5[60]
```

Do not prioritize these before `action.base_quat_target[4]`; the quaternion target
is easier to inspect, validate, and convert into the SONIC 6D representation.

## Minimum Summary

```text
✅ Con lo actual podemos construir casi todo para modo g1.
✅ Velocidades se pueden derivar desde posiciones.
✅ encoder_mode_4 es fijo para g1.
✅ motion_joint_positions_10frame_step5 sale de action.wbc[t+k][:29].
✅ manos salen directo de action.wbc[22:29] y action.wbc[36:43].
❌ Lo único realmente mínimo que conviene grabar adicionalmente es action.base_quat_target[4].
```

