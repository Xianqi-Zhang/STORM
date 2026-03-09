# STORM Assets

This directory stores embodiment assets used by STORM.

## Layout
- `robots/`: full robot bodies (humanoids and complete mobile manipulators).
- `hands/`: standalone dexterous hands (URDF + optional MJCF hand-only models).

## Current robot assets
- `robots/unitree_g1`
- `robots/unitree_h1_2`
- `robots/booster_t1`
- `robots/fourier_n1`

## Unitree G1 canonical files
- `robots/unitree_g1/g1.urdf`: base G1 URDF for IsaacLab.
- `robots/unitree_g1/g1_inspire_hand.urdf`: G1 with mounted Inspire left/right hands (URDF).
- `robots/unitree_g1/g1_with_hands.xml`: MJCF for dynamic HOI with articulated fingers.
- `robots/unitree_g1/g1_nohands.xml`: MJCF without articulated fingers.

## G1 + Inspire mount convention
- Hand is attached to wrist with fixed joints:
  - `lhand_mount_joint`: parent `left_wrist_yaw_link`
  - `rhand_mount_joint`: parent `right_wrist_yaw_link`
- Current calibrated mount rotations in `g1_inspire_hand.urdf`:
  - left: `rpy = 0 0 1.5707963`
  - right: `rpy = 3.1415926 0 -1.5707963`
- `lhand_base_joint` and `rhand_base_joint` are kept at `rpy = 0 0 0`.

Note:
Wrist and palm frames are rigidly connected. This is valid for dynamics/simulation,
but coordinate-transform-sensitive modules (IK, retarget, grasp planners) must use the
same `wrist -> palm` fixed transform convention.

## Hand assets
- URDF hand library from dex-urdf is under `hands/*`.
- MJCF hand-only models from spider currently kept under:
  - `hands/mjcf/metahand`

## Inspire Hand (Current Asset Spec)
Source file:
- `hands/inspire_hand/inspire_hand_right.urdf`

Kinematics (from current URDF):
- `active_joint`: `6` (independent, directly controlled revolute joints)
- `passive_joint`: `6` (mimic revolute joints, coupled to active joints)
- `fixed_joint`: `6` (structural fixed joints)
- `total_joint`: `18`

Relation:
- `active_joint + passive_joint + fixed_joint = total_joint`
- `6 + 6 + 6 = 18`

Active joint names:
- `thumb_proximal_yaw_joint`
- `thumb_proximal_pitch_joint`
- `index_proximal_joint`
- `middle_proximal_joint`
- `ring_proximal_joint`
- `pinky_proximal_joint`

Mimic mapping (`passive_joint` <- `active_joint`):
- `thumb_intermediate_joint = 1.334 * thumb_proximal_pitch_joint + 0`
- `thumb_distal_joint = 0.667 * thumb_proximal_pitch_joint + 0`
- `index_intermediate_joint = 1.06399 * index_proximal_joint - 0.04545`
- `middle_intermediate_joint = 1.06399 * middle_proximal_joint - 0.04545`
- `ring_intermediate_joint = 1.06399 * ring_proximal_joint - 0.04545`
- `pinky_intermediate_joint = 1.06399 * pinky_proximal_joint - 0.04545`

Practical note:
- In some notes/scripts, `activate_joint` is used as a typo/alias of `active_joint`.
- STORM uses `active_joint` (6-DoF) as policy output; `passive_joint` is expanded by the above mapping.

Control policy used in STORM:
- Default hand control space: `6`-D (`active_joint` only).
- Runtime execution expands to 12 revolute joints by mimic mapping:
  - `q_hand_full = Expand(q_hand_active)`
- `12`-D independent hand control is treated as a simulation-only ablation,
  not the default setting for sim2real-oriented training.

Size (zero pose, collision-based AABB, meters):
- `x ≈ 0.0763 m`
- `y ≈ 0.1584 m`
- `z ≈ 0.2181 m`
- Equivalent axis extents in cm: `7.63 × 15.84 × 21.81`

Note:
- The above size is AABB in the current URDF frame at zero pose (frame-dependent).
- For planner/controller thresholds, prefer link-level contact geometry and fingertip distances over raw global AABB.

Assets are sourced from:
- https://github.com/facebookresearch/spider/tree/main/spider/assets
- https://github.com/dexsuite/dex-urdf
