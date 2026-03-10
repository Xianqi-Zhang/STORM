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

### 1.1 One-Paragraph Summary
Given a task condition `x` (text, scene/object context, and robot embodiment), STORM jointly generates:
- human trajectory `s_h`,
- object trajectory `s_o`,
- robot-trackable targets `y_r`.

The key idea is to model the joint process `p(s_h, s_o, y_r | x)` under a shared interaction intent with compatibility constraints learned during training, then use an Embodiment Bridge to align HOI dynamics with robot-executable targets. This avoids the common generate-then-retarget mismatch.

### 1.2 Symbol Guide
- `x`: task condition (`x_txt`, `x_scene`, `x_obj`, `x_robot`)
- `z_int`: shared interaction latent
- `T_int`: structured interaction tokens generated from `z_int`
- `s_h`, `s_o`: generated human/object trajectories
- `y_r`: generated robot targets in `Y_trackable`
- `B_phi`: Embodiment Bridge mapping HOI trajectories to robot-trackable targets
- `y_r_gt`: grounded robot targets from projection/IK

### 1.3 Paper-Ready Contribution Paragraphs
This subsection is for manuscript writing reuse and is not required for implementation.

**Abstract-style paragraph**
STORM is a compatibility-constrained generative framework for embodied interaction synthesis. Instead of generating human-object motion and retargeting afterward, STORM jointly models human motion, object dynamics, and robot-trackable interaction targets under shared interaction intent: `p(s_h, s_o, y_r | x)`. The method introduces a robot-agnostic trackable interaction interface and cross-view consistency learning between HOI dynamics and robot control targets, improving executable alignment during generation. Compatibility is enforced through soft training constraints over reachability, controller trackability, HOI geometry consistency, and bridge-based cross-view alignment, yielding interactions that are simultaneously human-realistic, object-consistent, and robot-executable.

**Introduction-style paragraph**
The central contribution of STORM is a shift from pipeline integration to compatibility-constrained interaction modeling. We formulate embodied HOI synthesis as joint conditional generation over `(s_h, s_o, y_r)` rather than a generate-then-retarget decomposition, so embodiment constraints influence interaction generation directly. We further define a robot-agnostic trackable interaction representation that preserves interaction semantics while remaining controller-executable across embodiments. Finally, we introduce cross-view consistency learning that aligns robot targets inferred from HOI dynamics with robot targets produced by the generative stream, providing stable embodiment-aware supervision during training.

**Method-opener paragraph**
STORM models embodied interaction as joint generation with shared intent and soft compatibility constraints. Given condition `x` (task text, scene/object context, embodiment), the model samples interaction intent and jointly predicts human trajectory `s_h`, object trajectory `s_o`, and robot-trackable targets `y_r`. Training combines motion, interaction, outcome, robot-feasibility, and physics objectives with a cross-view bridge objective and grounded bridge supervision. This design treats compatibility not as post-hoc correction but as a learned constraint: reachability and trackability are enforced in `L_robot`, HOI geometric consistency in `L_interaction`, and HOI-robot alignment in `L_bridge/L_bridge_gt`.

## 2. Input and Representation

### 2.1 Inputs
- Text instruction `x_txt`.
- Scene/object context `x_scene`, `x_obj`.
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
3. Feed the graph to a lightweight encoder to produce embodiment embedding `x_robot`.

### 2.3 Motion State
At each timestep `t`, represent:
- Human state `s_h(t)` (root pose, body joints, velocities),
- Object state `s_o(t)` (pose, velocity, angular velocity),
- Robot state `s_r(t)` (joint positions/velocities, root/base state).

### 2.4 Robot Trackable Space
To make co-generation constraints explicit, STORM does not align HOI to full robot pose space.
Instead, we define a robot-trackable target space:

```
Y_trackable = {
  root trajectory,
  end-effector targets,
  contact anchors,
  temporal phase tokens
}
```

`y_r(t) ∈ Y_trackable` is the shared executable interface between HOI generation and robot control.
This is the key modeling choice behind the Embodiment Bridge.

Interface convention:
- `Y_trackable` is robot-agnostic at the schema level (supports humanoid, single-arm, dual-arm, and multi-effector settings).
- The interface dimensionality is fixed **within a training schema** for stable conditioning and batching.
- This does not imply identical raw actuator spaces across different robots.

## 3. Model Architecture

### 3.1 Main Method (6 Core Modules)

1. **Embodiment Graph Encoder**
- Input: robot description (`URDF/MJCF/USD`) converted to a unified graph.
- Output: embodiment embedding `x_robot`.

2. **Shared Interaction Latent Predictor**
- Predict a unified interaction latent `z_int` from `x_txt`, `x_scene`, `x_obj`, optional `x_h0`.
- `z_int` contains contact anchors, object-relative goals, phase timing, and interaction mode.

3. **Structured Interaction Token Generator**
- Generate a fixed token set `T_int = {t_1, ..., t_K}`, `t_k ∈ R^d`, from `z_int`.
- Tokens are latent interaction anchors (contact cues, object-relative goals, phase/timing cues).
- `K` is fixed across tasks; each token occupies a dedicated input slot.

4. **Cross-Embodiment Co-Generation Transformer (CE-CGT)**
- Joint generator with shared conditioning (`x`, `z_int`, `T_int`):
  - human/object stream: `s_h(1:T)`, `s_o(1:T)`
  - robot stream: `y_r(1:T) ∈ Y_trackable`.

5. **Embodiment Bridge**
- Define bridge as `B_phi(s_h, s_o) -> Y_trackable`.
- During training, bridge acts as an embodiment-alignment stabilizer, not the core model identity.

6. **Object-Centric Interaction Field Encoder**
- Builds interaction-invariant descriptors in object coordinates:
  - `r_hand_obj(t) = R_obj(t)^T (p_hand(t)-p_obj(t))`
  - `v_rel(t) = R_obj(t)^T (v_hand(t)-v_obj(t))`
- Provides interaction-invariant supervision for dynamic-object feasibility.

### 3.2 End-to-End Dataflow
The algorithm can be read as:

```
x -> z_int -> T_int -> {s_h, s_o, y_r}
                    \-> B_phi(s_h, s_o) for training-time alignment
```

Interpretation:
- `z_int` and `T_int` define shared interaction intent.
- `{s_h, s_o, y_r}` are generated jointly, not sequentially.
- `B_phi` is a training stabilizer that enforces HOI-to-robot consistency.

### 3.3 Interaction Token Protocol
- Tokens are generated by `T_int = f_theta(z_int)`.
- Token semantics are not manually labeled; roles emerge through task losses and cross-stream consistency.
- Required ablations: `K=0` (no tokens), reduced `K`, full `K`, token shuffle, and token masking at test time.

### 3.4 Reachability-Aware Interaction Prior
We explicitly model embodiment reachability at anchor level:

```
R(anchor, x_robot) -> [0, 1]
```

where `R` predicts whether a contact anchor is reachable by the current embodiment.
This prior can be used both as:
- a sub-term in `L_robot` during training,
- and a sampling-time feasibility filter.

### 3.5 Optional Interaction Reasoner
- The reasoner is **optional** and kept only if it predicts high-value signals
  (`contact anchors`, `intercept windows`, confidence), not only phase ids.
- If it only predicts phase labels, it should be removed to avoid redundancy.

### 3.6 Decoupled Robot Control (Body vs Hand)

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

## 4. Algorithm Specification

This section provides one linear implementation view: forward pass, objective, optimization protocol, and schedule.

### 4.1 Probabilistic Identity

Let `x = {x_txt, x_scene, x_obj, x_robot}`. STORM uses a single joint factorization:

```
p(s_h, s_o, y_r | x) = ∫ p(z_int | x) p(s_h, s_o, y_r | z_int, x) dz_int
```

where `y_r ∈ Y_trackable`.  
This is the only probabilistic factorization used in this document.

### 4.2 Forward Pass (Implementation Form)

```
Input: x = {x_txt, x_scene, x_obj, x_robot}
z_int = InteractionLatent(x)
T_int = TokenGenerator(z_int)                  # fixed K slots
s_h_hat, s_o_hat, y_r_hat = CoGenerator(x, z_int, T_int)
```

Two-view bridge variables:
- `y_r_gen = y_r_hat` (generative view)
- `y_r_hoi = B_phi(s_h_hat, s_o_hat)` (HOI dynamics view)

### 4.3 Unified Training Objective

Base objective:

```
L_base = L_motion + L_interaction + L_physics + L_outcome + L_robot
```

Bridge objective:

```
L_bridge = (1/T) * Σ_t || y_r_gen[t] - sg(y_r_hoi[t]) ||^2
```

Grounded bridge supervision:

```
y_r_gt = Pi_robot(s_h_gt, s_o_gt)
L_bridge_gt = (1/T) * Σ_t c[t] * || B_phi(s_h_gt[t], s_o_gt[t]) - y_r_gt[t] ||^2
```

Total training objective:

```
L_total = L_base + λ_b * L_bridge + λ_g * L_bridge_gt
```

Compatibility interpretation:
- STORM uses soft compatibility constraints (not hard-constrained optimization).
- Formal score (for analysis):

```
C(s_h, s_o, y_r) = - (a1 * L_interaction + a2 * L_robot + a3 * L_bridge + a4 * L_bridge_gt)
```

Higher `C` indicates better compatibility across HOI geometry, embodiment feasibility, and cross-view alignment.

### 4.4 Loss Semantics and Decompositions

Loss-role summary:

| Loss | Purpose |
| --- | --- |
| `L_motion` | trajectory realism and reconstruction quality |
| `L_interaction` | HOI structure consistency |
| `L_outcome` | task success and contact timing correctness |
| `L_robot` | embodiment feasibility (including reachability) |
| `L_physics` | dynamic plausibility and safety |
| `L_bridge` | cross-view alignment |
| `L_bridge_gt` | bridge grounding to projected executable targets |

`L_outcome` decomposition:

```
L_outcome = u1 * L_contact_time
          + u2 * L_contact_state
          + u3 * L_post_object
          + u4 * L_post_stability
```

`L_robot` decomposition:

```
L_robot = w_b * L_robot_body
        + w_h * L_robot_hand
        + w_i * L_interface
```

Notes:
- Reachability is implemented as a sub-term of `L_robot`.
- Controller tracking feasibility is enforced by `L_robot` and rollout-based evaluation.

### 4.5 Optimization Protocol

Gradient flow:
- `L_bridge` updates generator parameters through `y_r_gen` (HOI branch is stop-gradient).
- `L_bridge_gt` updates bridge parameters via grounded supervision.

Projection reliability handling:
- `c[t]` is a projection-validity/confidence weight from IK/retargeting checks.
- Failed projection frames use `c[t]=0` and are excluded from `L_bridge_gt`.

Mixed conditioning for bridge robustness:

```
(s_h_in, s_o_in) =
  (s_h_gt, s_o_gt) with prob p
  (s_h_hat, s_o_hat) with prob 1-p
```

### 4.6 Training Schedule

1. Bridge warmup:
- optimize `L_bridge_gt` to stabilize HOI->robot mapping.

2. Joint training:
- optimize `L_total` with ramped `λ_b`.

3. Simulation refinement:
- use rollout statistics to tune feasibility/stability weights.

### 4.7 Alignment and Batching Rules

- Resample human/object/robot trajectories to a shared frame rate (`30 Hz` default).
- Linear interpolation for dense states; nearest-neighbor for discrete contact states.
- Use padding masks for variable-length sequences.
- Drop or mask invalid synchronized frames before loss accumulation.

### 4.8 Planned Innovation Upgrades (Next Iteration)

1. **Feasibility prediction head (primary)**
- `F(s_h, s_o, y_r, x_robot) -> [0, 1]`
- `L_feas = BCE(F(...), success_label)`
- `L_total_next = L_total + λ_f * L_feas`

2. **Cross-view contrastive consistency**
- Add contrastive positives/negatives on HOI-robot pairs to strengthen shared representation.

3. **Contact intervention robustness (optional)**
- Perturb contact timing/anchors/sequence during training.
- Train for stable outcomes under intervention-style perturbations (not a formal causal-identification claim).

## 5. Why Co-Generation Instead of Retarget-Only

Retarget-centric pipelines primarily optimize kinematic similarity and can fail dynamic HOI tasks.
In tasks like ball interaction, correct execution depends on:
- where the object is,
- when contact occurs,
- whether the robot can physically reach and control that contact.

Human motions can be infeasible for a given robot (e.g., required reach exceeds arm length or joint limits), so the human trajectory itself must be **shaped by robot feasibility**. This is the core necessity for co-generation: robot constraints must influence human motion during generation, not only after the fact.

STORM therefore introduces robot embodiment and object-state constraints **during generation**, not only as a post-hoc correction. The model explicitly targets the joint distribution `p(s_h, s_o, y_r | x)` rather than a two-stage `p(s_h, s_o | x) p(y_r | s_h, s_o)`.

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
- Train shared interaction latent + human/object stream with `L_motion + L_interaction + L_physics`.

### Stage B: Co-generation alignment
- Add robot stream and Embodiment Bridge.
- Enable decoupled body/hand controllers with interface constraints.
- Warm up bridge with `L_bridge_gt`, then train jointly with `L_total`.
- Emphasize `L_bridge`, `L_bridge_gt`, `L_robot_body`, `L_robot_hand`, and `L_interface` on dynamic-object samples.

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
3. Remove Embodiment Bridge (`Bridge`).
4. Replace `Y_trackable` with direct full-pose robot prediction.
5. Token ablation: `K=0` (no tokens), reduced `K`, full `K`.
6. Token shuffle perturbation at test time.
7. Token masking perturbation at test time.
8. Remove bridge grounding (`L_bridge_gt`) and keep `L_bridge` only.
9. Remove reachability term from `L_robot`.
10. Remove explicit contact-time term (`L_contact_time`).
11. Evaluate dynamic-object subset separately.
12. Single monolithic controller vs decoupled body/hand controllers.
13. 6-DoF active-hand control (mimic expansion) vs 12-DoF independent hand control.

## 9. Implementation Notes

- Use a shared internal embodiment schema to decouple file format from model training.
- Keep parsers deterministic and cached (`URDF/MJCF/USD -> graph tensor`).
- Start from one reference robot (e.g., Unitree G1), then add multi-embodiment training.
- Prioritize robust simulator validation before real robot transfer.
- Keep the main method section focused on learning formulation; detailed controller/simulator mechanics should be placed in system or appendix sections.



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

4. **Contact-Consistent Object-Pose Augmentation (InterReal-style)**
- Use IK to preserve hand-object contact while perturbing object pose to generate augmented HOI references.
- Role: robustness to object pose noise and sim2real disturbance; optional data augmentation, not a core modeling change.

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

### 12.6 InterReal
- Reference: *InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills*  
  https://arxiv.org/abs/2603.07516
- Borrowed idea:
  1. Contact-consistent object-pose augmentation via IK for robustness.
- Not borrowed:
  1. Meta-policy automatic reward learner as a default training component.
  2. Full imitation-RL tracking pipeline as the main STORM method.
  2. Geometry-aware penetration/sliding cleanup before policy learning.
