# STORM: Structured Task-Oriented Human-Robot Motion Co-Generation for Executable Human-Object Interaction

## Title
**STORM: Structured Task-Oriented Human-Robot Motion Co-Generation for Executable Human-Object Interaction**

## Acronym
**STORM** = **S**tructured **T**ask-**O**riented Human-**R**obot **M**otion Co-Generation

## Document Scope
This document provides the implementation and training details of STORM.

## Title-Keyword to Module Mapping

- **Structured** -> `Interaction Reasoner` + `Embodiment Graph Encoder`:
  explicit phase/contact/state structure rather than pose-only trajectory fitting.
- **Task-Oriented** -> objective design in `L_interaction`, `L_outcome`, and `L_robot`:
  optimize contact feasibility, timing correctness, and outcome consistency instead of pure kinematic similarity.
- **Human-Robot Motion Co-Generation** -> `Human-Object Motion Generator` + `Human-Robot Co-Generation Head`:
  jointly produce human-object trajectories and robot-trackable latent targets under shared conditioning.
- **Executable Human-Object Interaction** -> `Controller-in-the-loop Simulation` + `L_physics`/`L_robot`:
  enforce stability, collision safety, and execution success in Isaac Lab / MuJoCo.


## 1. Scope

This document defines the concrete implementation of STORM, with a focus on:
- Human-Robot Co-Generation as the primary novelty,
- dynamic HOI feasibility (not only pose similarity),
- embodiment-aware training with robot description inputs,
- practical training and evaluation procedures.

## 2. Input and Representation

### 2.1 Inputs
- Text instruction `x_txt`.
- Initial object state `x_obj0` (pose, velocity if available).
- Optional initial human state `x_h0`.
- Robot embodiment description `x_robot` parsed from URDF/MJCF/USD.

### 2.2 Embodiment Encoding (URDF/MJCF/USD)
We normalize robot description files into a unified **Embodiment Graph**:
- Node attributes: link mass/inertia, collision shape, visual shape, center of mass.
- Edge attributes: joint type, joint axis, limits, damping/friction, parent-child relation.
- Global attributes: base type, gravity convention, control timestep.

Recommended implementation:
1. Keep URDF as the canonical source for robot morphology.
2. Parse MJCF/USD through adapters into the same graph schema.
3. Feed the graph to a lightweight GNN/Transformer encoder to produce robot tokens `z_robot`.

### 2.3 Motion State
At each timestep `t`, represent:
- Human state `s_h(t)` (root pose, body joints, velocities),
- Object state `s_o(t)` (pose, velocity, angular velocity),
- Robot state `s_r(t)` (joint positions/velocities, root/base state).

## 3. Model Architecture

### 3.1 Main Method (3 Core Modules)

1. **Embodiment Graph Encoder**
- Input: robot description (`URDF/MJCF/USD`) converted to a unified graph.
- Output: embodiment tokens `z_robot`.

2. **Outcome-Centric Interaction Generator**
- Backbone: conditional sequence/diffusion generator.
- Input: `z_txt`, `x_obj0`, optional `x_h0`, `z_robot`.
- Output:
  - human trajectory `s_h(1:T)`
  - object trajectory `s_o(1:T)`
  - robot-trackable latent `y_r(1:T)`
  - outcome targets `o_hat = {t_c, contact_state, post_object_state, stability_state}`

3. **Object-Centric Interaction Field Encoder**
- Builds interaction-invariant descriptors in object coordinates:
  - `r_hand_obj(t) = R_obj(t)^T (p_hand(t)-p_obj(t))`
  - `v_rel(t) = R_obj(t)^T (v_hand(t)-v_obj(t))`
- Provides structured supervision for dynamic-object interaction feasibility.

### 3.2 Optional Interaction Reasoner
- The reasoner is **optional** and kept only if it predicts high-value signals
  (`contact anchors`, `intercept windows`, confidence), not only phase ids.
- If it only predicts phase labels, it should be removed to avoid redundancy.

### 3.3 Decoupled Robot Control (Body vs Hand)

To avoid a brittle monolithic controller, STORM uses **two separate control modules**:

1. **Body Control Module**
- Controls base/torso/leg/arm (up to wrist) joints.
- Optimizes balance, whole-body reachability, tracking, and locomotion stability.

2. **Hand Control Module**
- Controls finger/hand articulation for contact and grasp.
- Optimizes local contact quality, closure timing, and object-holding stability.
- **Default control dimensionality: 6-DoF** (`active_joint` only, sim2real-oriented).
- 12-joint execution is obtained via mimic expansion at runtime (`q_hand_full = Expand(q_hand_active)`).
- 12-DoF independent hand control is only used as an ablation baseline.
- Hand action space convention:
  - `q_hand_active` (6-DoF) = `[thumb_proximal_yaw, thumb_proximal_pitch, index_proximal, middle_proximal, ring_proximal, pinky_proximal]`
  - `q_hand_full` (12 revolute joints) is obtained by mimic expansion from `q_hand_active`.

Interface between the two modules:
- Wrist pose/velocity commands from body module.
- Contact intent/outcome signals from generation module.
- Hand module solves local dexterous behavior under the wrist trajectory.

Design decision:
- **Not** a single end-to-end controller for all joints.
- Body and hand are optimized separately with coordinated interface constraints.

## 4. Training Objective

We use a compact 5-term objective:

```
L_total = λ1 * L_motion
        + λ2 * L_interaction
        + λ3 * L_outcome
        + λ4 * L_robot
        + λ5 * L_physics
```

- `L_motion`: reconstruction/denoising quality for human-object trajectories.
- `L_interaction`: contact timing/consistency and interaction field alignment (`r_hand_obj`, `v_rel`).
- `L_outcome`: outcome-layer supervision (`t_c`, contact state, post-object state, post-contact stability).
- `L_robot`: embodiment feasibility (joint limits, reachability, trackability).
- `L_physics`: penetration, foot-sliding, and balance constraints.

Design principle: keep losses minimal and non-overlapping; avoid excessive handcrafted terms in the main method.

### 4.1 Robot Feasibility Decomposition

`L_robot` is decomposed as:

```
L_robot = w_b * L_robot_body
        + w_h * L_robot_hand
        + w_i * L_interface
```

- `L_robot_body`: joint limits, body trackability, balance, and reachability.
- `L_robot_hand`: finger limits, grasp/contact consistency, and local dexterity stability.
- `L_interface`: wrist-hand consistency (hand behavior conditioned on wrist trajectory and contact intent).

This enforces body-hand coordination without collapsing into a single monolithic end-to-end control objective.

Hand-control implementation convention:
- policy output: `q_hand_active` (6-DoF)
- execution state: `q_hand_full` (12 revolute joints after mimic expansion)
- training default uses 6-DoF action space for sim2real consistency
- mimic expansion follows the URDF coupling equations (documented in `src/storm/assets/README.md`).

## 5. Why Co-Generation Instead of Retarget-Only

Retarget-centric pipelines primarily optimize kinematic similarity and can fail dynamic HOI tasks.
In tasks like ball interaction, correct execution depends on:
- where the object is,
- when contact occurs,
- whether the robot can physically reach and control that contact.

STORM therefore introduces robot embodiment and object-state constraints **during generation**, not only as a post-hoc correction.

## 6. Data Pipeline

### 6.1 Training Data
- Primary: InterAct / InterAct-X.
- Auxiliary: OMOMO.
- Optional dynamic skill references: SkillMimic-style basketball trajectories.

### 6.2 Preprocessing
1. Normalize coordinate systems and frame rates.
2. Build contact annotations (human-object contact points and timing).
3. Convert object geometry to simulation-friendly representations.
4. Parse robot description files into Embodiment Graph cache.

## 7. Training Stages

### Stage A: HOI prior pretraining
- Train text-conditioned HOI generator with `L_motion + L_interaction + L_physics`.

### Stage B: Co-generation alignment
- Add robot encoder and co-generation head.
- Enable decoupled body/hand controllers with interface constraints.
- Train with full objective, emphasizing `L_robot_body`, `L_robot_hand`, and `L_interface` on dynamic-object samples.

### Stage C: Simulation-in-the-loop refinement
- Roll out generated sequences in simulator.
- Use rollout statistics to adapt weights for feasibility/stability terms.

Recommended curriculum:
1. static/simple objects,
2. articulated objects,
3. dynamic objects (ball-like interactions).

## 8. Evaluation and Reporting

### 8.1 Core metrics
- Motion quality: realism/diversity/text alignment.
- Interaction quality: contact timing and synchronization error.
- Physical validity: penetration, foot sliding, balance violations.
- Robot executability: success rate, tracking error, stability/fall rate.

### 8.2 REIS
```
REIS = α * Q_motion
     + β * S_exec
     + γ * (1 - E_track)
     + δ * P_stability
```
Report sensitivity to `α, β, γ, δ`.

### 8.3 Required ablations
1. Remove robot embodiment input.
2. Replace co-generation with generate-then-retarget baseline.
3. Remove object-conditioned reachability term in `L_robot`.
4. Remove contact timing feasibility term.
5. Evaluate dynamic-object subset separately.
6. Single monolithic controller vs decoupled body/hand controllers.
7. 6-DoF active-hand control (mimic expansion) vs 12-DoF independent hand control.

## 9. Implementation Notes

- Use a shared internal embodiment schema to decouple file format from model training.
- Keep parsers deterministic and cached (`URDF/MJCF/USD -> graph tensor`).
- Start from one reference robot (e.g., Unitree G1), then add multi-embodiment training.
- Prioritize robust simulator validation before real robot transfer.



## 10. System Extensions

The following modules are useful in practice but are not part of the core STORM method:

1. **Critical-Window Constrained Refinement**
- Post-generation trajectory optimization on key contact windows only.
- Role: execution boost under hard constraints; not a core learning innovation.

2. **Failure-Aware Recovery Head**
- Handles drops/misses/contact loss in long-horizon deployment.
- Role: system robustness extension; not required for core generation claims.

3. **Motion-Matching Skill Chaining**
- Long-horizon skill composition for downstream control.
- Role: deployment/control layer; not required for main STORM contribution.

## 11. Key Experiment: Co-Generation vs Retarget

To validate the core claim, dynamic-object tasks must include direct comparison:

- STORM (co-generation)
- generate -> retarget
- generate -> retarget -> optimization

Recommended tasks:
- ball catch
- shuttlecock hit
- dynamic intercept-like interactions

Required metrics:
- contact success rate
- task completion rate
- object trajectory error
- stability/fall rate

Core thesis to test:
**Pose similarity is not sufficient for interaction feasibility.**
Co-generation should outperform retarget pipelines on dynamic-object execution metrics.

## 12. Reference

Only the ideas that are explicitly integrated into STORM are listed here.

### 12.1 URDFormer
- Reference: *URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images*  
  https://arxiv.org/abs/2405.11656
- Borrowed idea:
  1. Structured embodiment description as model input.
  2. Unified parser path (`URDF/MJCF/USD -> internal graph`).
  3. Deterministic, cached description-to-simulation pipeline.
- Not borrowed:
  1. Image-to-URDF scene reconstruction pipeline.

### 12.2 OmniRetarget
- Reference: *OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction*  
  https://arxiv.org/abs/2509.26633
- Borrowed idea:
  1. Interaction-preserving relative geometry prior (mesh/Laplacian-style descriptor).
  2. Hard-constraint local refinement around critical contact windows.
- Not borrowed:
  1. Retarget-first full pipeline as STORM main method.

### 12.3 Pro-HOI
- Reference: *Pro-HOI: Perceptive Root-guided Humanoid-Object Interaction*  
  https://arxiv.org/abs/2603.01126
- Borrowed idea:
  1. Root-guided high-level task interface.
  2. Failure recovery loop for dropped/lost objects.

### 12.4 Perceptive Humanoid Parkour (PHP)
- Reference: *Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching*  
  https://arxiv.org/abs/2602.15827
- Borrowed idea:
  1. Motion matching for long-horizon dynamic skill chaining.
  2. Teacher-student distillation with RL correction for multi-skill policy unification.

### 12.5 MeshMimic
- Reference: *MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction*  
  https://arxiv.org/abs/2602.15733
- Borrowed idea:
  1. Scene/contact-coupled correction priors for cleaner interaction references.
  2. Geometry-aware penetration/sliding cleanup before policy learning.