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
- **Task-Oriented** -> objective design in `L_interaction` and `L_robot`:
  optimize contact feasibility, timing correctness, and task completion instead of pure kinematic similarity.
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

### 3.1 High-Level Structure
`{x_txt, x_obj0, x_h0, x_robot} -> Interaction Reasoning -> Co-Generation -> Simulation Check`

### 3.2 Modules
1. Text encoder
- Encodes instruction to language token sequence `z_txt`.

2. Interaction Reasoner
- Predicts phase/contact plan `p(t)` and interaction anchors.
- Output includes contact timing priors and target contact regions.

3. Human-Object Motion Generator
- Diffusion/sequence backbone generating `s_h(1:T), s_o(1:T)`.
- Conditioned on `z_txt`, phase plan `p(t)`, and robot tokens `z_robot`.

4. Human-Robot Co-Generation Head
- Predicts robot-trackable latent targets `y_r(1:T)` jointly with HOI motion.
- Enforces object-conditioned reachability constraints during generation.
- Uses minimal projection to robot joint space (no heavy post-hoc retarget optimization).

5. Controller-in-the-loop Simulation
- Executes projected trajectories in Isaac Lab / MuJoCo.
- Returns tracking, stability, and task-contact feedback.

## 4. Training Objective

Overall objective:
```
L = λ1 * L_motion
  + λ2 * L_interaction
  + λ3 * L_physics
  + λ4 * L_robot
```

### 4.1 `L_motion`
- Reconstruction/denoising loss on human-object trajectories.
- Includes velocity/acceleration smoothness regularizers.

### 4.2 `L_interaction`
- Phase transition consistency.
- Contact timing alignment.
- Relative pose consistency between human end-effectors and object interaction sites.

### 4.3 `L_physics`
- Penetration penalty (human-object, self, ground).
- Contact consistency and non-floating constraints.
- Balance and foot-sliding penalties.

### 4.4 `L_robot`
- Joint-limit and actuation-feasibility constraints.
- Object-conditioned reachability (robot-to-object contact viability at required phases).
- Contact timing feasibility (robot can realize required contact schedule).
- Trackability/stability terms from simulation rollout.

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
- Train with full objective, emphasizing `L_robot` on dynamic-object samples.

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

## 9. Implementation Notes

- Use a shared internal embodiment schema to decouple file format from model training.
- Keep parsers deterministic and cached (`URDF/MJCF/USD -> graph tensor`).
- Start from one reference robot (e.g., Unitree G1), then add multi-embodiment training.
- Prioritize robust simulator validation before real robot transfer.



## 11. Interaction Representation for Dynamic Objects

### 11.1 Design Principle
For dynamic HOI (e.g., ball catching), we do not enforce identical robot poses across embodiments.
Instead, we enforce consistent **interaction outcomes** with embodiment-specific feasible motions.

This section borrows from:
- URDFormer: structured embodiment representation.
- OmniRetarget: interaction-preserving relative geometry (adapted, not copied as a retarget pipeline).

### 11.2 Two-Layer Interaction Representation

1. Outcome Layer (cross-embodiment invariant)
- Intercept/catch time `t_c`.
- Contact state at `t_c` (contact/no-contact, contact region).
- Relative contact geometry (end-effector to object center, contact normal).
- Post-contact object state (velocity damping / stabilized possession).
- Stability state after contact (no-fall, bounded trunk/root deviation).

2. Embodiment Layer (robot-specific)
- Joint trajectory `q(1:T)` and `dq(1:T)`.
- Foot contact sequence.
- Center-of-mass / support relation.
- Actuation and joint/velocity feasibility.

Different robots may realize different `q(1:T)`, but should match Outcome Layer targets.

### 11.3 Relative Interaction Descriptor
To preserve interaction semantics without forcing pose identity, we use relative descriptors in object-centric coordinates:
- `r_hand_obj(t) = R_obj(t)^T * (p_hand(t) - p_obj(t))`
- `v_rel(t) = R_obj(t)^T * (v_hand(t) - v_obj(t))`
- optional local interaction graph/Laplacian descriptor over selected body-object-environment points.

Note: Laplacian-style descriptors are used as a relative-geometry prior, not as a full retargeting objective.

### 11.4 Catch/Intercept Process (Ball Example)
1. Predict object trajectory and uncertainty for horizon `T`.
2. Compute feasible intercept window candidates `{(x_i, t_i)}` under embodiment constraints.
3. Select target window maximizing feasibility + stability margin.
4. Co-generate robot motion conditioned on selected outcome target.
5. Enforce post-contact stabilization (object velocity reduction and body recovery).

### 11.5 Loss Terms for Dynamic Interaction
Add to the base objective:

- `L_outcome`:
  match `t_c`, contact state, and post-contact object state.
- `L_rel`:
  align object-centric relative descriptors (`r_hand_obj`, `v_rel`, optional graph prior).
- `L_intercept_feas`:
  penalize infeasible intercept windows (timing/reachability mismatch).
- `L_stab_post`:
  enforce stability margin after contact (root tilt, support consistency, no-fall).

Updated objective (conceptually):
```
L_total = L_motion + L_interaction + L_physics + L_robot
        + w1 * L_outcome
        + w2 * L_rel
        + w3 * L_intercept_feas
        + w4 * L_stab_post
```

### 11.6 Multi-Embodiment Training Target
For robots with different sizes/structures:
- share Outcome Layer supervision,
- keep Embodiment Layer robot-specific,
- evaluate consistency by task success and stability, not pose similarity.

This ensures "different poses, same successful interaction result" across embodiments.

## 12. Retarget Literature Takeaways for STORM

### 12.1 Interaction Representation (What to Borrow)
From recent retarget works, the most useful idea is to preserve **interaction relations** rather than only joint/keypoint similarity.

1. Interaction mesh + Laplacian prior (OmniRetarget-style)
- Preserve relative geometry among body/object/terrain points.
- Use as a soft prior in STORM (`L_rel`), not as the main objective.

2. Contact-area/distribution transfer (contact-rich hand retargeting)
- Represent contact as region/distribution instead of single-point matching.
- Useful for stable hand-object or hand-ball interaction under morphology changes.

3. Contact guidance during optimization (SPIDER-style)
- Add guidance signals when contact is ambiguous in dynamic tasks.
- Can be used in late-stage refinement for hard contact windows.

### 12.2 Single Human to Multiple Robot Embodiments
For multi-embodiment consistency, use shared interaction semantics with embodiment-specific realization.

1. Graph-conditioned embodiment encoding (G-DReaM-style)
- Encode robot morphology as graph features and correspondence masks.
- Condition generation on this graph to support heterogeneous robots.

2. Shared backbone + embodiment prompts/heads (AdaMorph-style)
- Shared model learns interaction intent.
- Embodiment-specific adapters decode feasible robot trajectories.
- Matches STORM target: different poses, similar task outcomes.

3. Fast geometric warm-start (closed-form upper-body methods)
- Useful for initialization and low-latency upper-body mapping.
- Not sufficient alone for full-body dynamic HOI.

### 12.3 Trajectory Optimization Components to Reuse
Retarget literature suggests a practical hybrid strategy: generation first, constrained refinement second.

1. Hard-constraint local refinement (OmniRetarget-style)
- Enforce joint/velocity limits, foot stability, and collision constraints.
- Apply on key interaction windows instead of full-sequence heavy optimization.

2. Sampling-based trajectory optimization (DynaRetarget-style)
- Start from kinematic/co-generated motion and refine toward dynamic feasibility.
- Incremental-horizon strategy is useful for long-horizon dynamic interactions.

3. Parallel sampling + contact curriculum (SPIDER-style)
- Improve robustness in contact-rich phases.
- Scales better for large generated datasets and diverse objects.

### 12.4 Integration Plan in STORM
1. Keep co-generation as the main pipeline.
2. Add object-centric relative descriptor + optional Laplacian prior in `L_rel`.
3. Condition model on embodiment graph (`URDF/MJCF/USD -> unified graph`).
4. Use lightweight constrained refinement only on critical contact phases.
5. Evaluate cross-embodiment by outcome consistency and stability, not pose similarity.

This preserves STORM's core novelty (Human-Robot Co-Generation) while incorporating robust retarget-inspired priors and optimization tools.

## 13. Selective Borrowing from Recent Robot Interaction Papers

Only ideas with clear expected gains for STORM are included below.

### 13.1 High-Value Additions

1. Root-guided task interface (from Pro-HOI)
- Use desired root trajectory as the high-level control interface.
- Keep full-body reference mainly in training losses, not mandatory online input.
- Benefit: cleaner planner integration and better cross-embodiment transfer.

2. Failure-recovery loop for object interaction (from Pro-HOI)
- Add explicit failure states: object drop, contact loss, out-of-view.
- Trigger recovery behaviors (re-acquire, re-grasp, re-stabilize).
- Benefit: long-horizon robustness for real deployments.

3. Motion matching for long-horizon dynamic skill chaining (from PHP Parkour)
- Compose atomic interaction skills into long-horizon references with feature-space retrieval.
- Use teacher-student distillation plus RL correction for a unified multi-skill policy.
- Benefit: improves continuity and diversity for dynamic interaction sequences.

4. Scene/contact-coupled retarget prior (from MeshMimic)
- Preserve motion-scene contact consistency during data preparation/refinement.
- Use geometry-aware penetration/sliding corrections before policy learning.
- Benefit: cleaner references and reduced artifact compensation burden in RL.

### 13.2 Explicitly Not Adopted

- Omni-Manip style panoramic LiDAR policy as a core method component.
Reason: mainly perception-FOV expansion; limited direct gain for the current core contribution
(Human-Robot co-generation with static+dynamic object interaction constraints).
Can be considered later as a deployment module if needed.


## 14. STORM V1 Frozen Spec (1 Core + 3 Supporting Modules)

This section is the implementation contract for the first complete STORM version.

### 14.1 Core Innovation Module: Outcome-Centric Co-Generation

Goal:
- Generate interaction trajectories that satisfy task outcomes, not pose mimicry.
- Allow different embodiments to realize different postures with similar interaction outcomes.

Network:
- Backbone: conditional sequence/diffusion generator.
- Inputs:
  - text tokens `z_txt`
  - initial object state `x_obj0`
  - optional initial human state `x_h0`
  - embodiment tokens `z_robot` from Embodiment Graph encoder
- Outputs:
  - human trajectory `s_h(1:T)`
  - object trajectory `s_o(1:T)`
  - robot-trackable latent `y_r(1:T)`
  - phase/contact logits `p(1:T)`
  - outcome predictions `o_hat = {t_c, c_state, post_obj_state, stab_state}`

Primary losses:
- `L_motion`: trajectory reconstruction/denoising.
- `L_interaction`: phase/contact consistency.
- `L_outcome`: outcome consistency (`t_c`, contact state, post-contact object state, stability state).

### 14.2 Supporting Module A: Object-Centric Interaction Field Encoder

Goal:
- Encode interaction geometry in object-centric coordinates for dynamic-object robustness.

Inputs:
- `s_h(t), s_o(t)`, selected contact/body anchor points, optional local scene points.

Outputs:
- relative descriptor sequence:
  - `r_hand_obj(t) = R_obj(t)^T (p_hand(t)-p_obj(t))`
  - `v_rel(t) = R_obj(t)^T (v_hand(t)-v_obj(t))`
  - optional Laplacian-style local geometry descriptor `g_lap(t)`

Losses:
- `L_rel = ||r_hand_obj - r*_hand_obj|| + ||v_rel - v*_rel|| + beta * ||g_lap - g*_lap||`

Notes:
- Laplacian descriptor is a soft interaction prior only.

### 14.3 Supporting Module B: Critical-Window Constrained Refinement

Goal:
- Refine only high-impact contact windows instead of full-trajectory heavy optimization.

Inputs:
- preliminary trajectories `s_h, s_o, y_r`
- predicted key windows `W = {(t_start, t_end)}`
- embodiment constraints (joint limits, velocity limits, collision pairs, foot constraints)

Outputs:
- refined robot trajectory `q_refined(1:T)`
- refined contact timing in key windows

Optimization objective in each window:
- minimize interaction deviation + tracking deviation
- subject to hard constraints:
  - `q_min <= q <= q_max`
  - `v_min <= dq <= v_max`
  - collision-free
  - stance stability constraints

Loss proxy for end-to-end training:
- `L_intercept_feas` (timing/reachability feasibility)
- `L_robot` (trackability/stability after refinement)

### 14.4 Supporting Module C: Failure-Aware Recovery Head

Goal:
- Improve long-horizon robustness (drop, miss, contact loss).

Inputs:
- short-horizon state history: object visibility/confidence, contact state, root/base state, balance indicators.

Outputs:
- failure state logits `f_hat`:
  - `normal`, `contact_loss`, `object_drop`, `unstable`
- recovery action mode `a_rec`:
  - `re_acquire`, `re_grasp`, `re_stabilize`

Losses:
- `L_fail_cls` (failure classification)
- `L_recovery` (recovery success supervision / rollout success)

### 14.5 Unified Objective (V1)

```
L_total = λ1 L_motion
        + λ2 L_interaction
        + λ3 L_physics
        + λ4 L_robot
        + λ5 L_outcome
        + λ6 L_rel
        + λ7 L_intercept_feas
        + λ8 L_stab_post
        + λ9 L_fail_cls
        + λ10 L_recovery
```

Recommended initial weights:
- `λ1..λ4 = 1.0`
- `λ5 = 1.0`
- `λ6 = 0.5`
- `λ7 = 0.7`
- `λ8 = 0.7`
- `λ9 = 0.3`
- `λ10 = 0.5`

### 14.6 Training Schedule (Frozen)

1. Stage A (base generation)
- Train core module + Module A with `L_motion/L_interaction/L_physics/L_outcome/L_rel`.

2. Stage B (feasibility)
- Enable Module B and optimize with `L_robot/L_intercept_feas/L_stab_post`.

3. Stage C (robustness)
- Enable Module C and optimize with `L_fail_cls/L_recovery` under perturbation rollouts.

### 14.7 Must-Pass Ablations

- Remove Outcome-Centric head (core novelty test).
- Remove Module A (`L_rel`) on dynamic-object split.
- Remove Module B (no window refinement).
- Remove Module C (no failure recovery).
- Compare against generate-then-retarget baseline.

The model is considered valid only if outcome consistency and stability improve without requiring pose similarity across embodiments.


## 10. Reference

Only the ideas that are explicitly integrated into STORM are listed here.

### 10.1 URDFormer
- Reference: *URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images*  
  https://arxiv.org/abs/2405.11656
- Borrowed idea:
  1. Structured embodiment description as model input.
  2. Unified parser path (`URDF/MJCF/USD -> internal graph`).
  3. Deterministic, cached description-to-simulation pipeline.
- Not borrowed:
  1. Image-to-URDF scene reconstruction pipeline.

### 10.2 OmniRetarget
- Reference: *OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction*  
  https://arxiv.org/abs/2509.26633
- Borrowed idea:
  1. Interaction-preserving relative geometry prior (mesh/Laplacian-style descriptor).
  2. Hard-constraint local refinement around critical contact windows.
- Not borrowed:
  1. Retarget-first full pipeline as STORM main method.

### 10.3 Pro-HOI
- Reference: *Pro-HOI: Perceptive Root-guided Humanoid-Object Interaction*  
  https://arxiv.org/abs/2603.01126
- Borrowed idea:
  1. Root-guided high-level task interface.
  2. Failure recovery loop for dropped/lost objects.

### 10.4 Perceptive Humanoid Parkour (PHP)
- Reference: *Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching*  
  https://arxiv.org/abs/2602.15827
- Borrowed idea:
  1. Motion matching for long-horizon dynamic skill chaining.
  2. Teacher-student distillation with RL correction for multi-skill policy unification.

### 10.5 MeshMimic
- Reference: *MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction*  
  https://arxiv.org/abs/2602.15733
- Borrowed idea:
  1. Scene/contact-coupled correction priors for cleaner interaction references.
  2. Geometry-aware penetration/sliding cleanup before policy learning.