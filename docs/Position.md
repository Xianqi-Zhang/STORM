# STORM: Structured Task-Oriented Human-Robot Motion Co-Generation for Executable Human-Object Interaction

## 1. Motivation

Recent text-conditioned motion generation methods produce visually plausible motions, but many outputs are not directly usable for robot mimic learning. Common failures include foot sliding, unstable balance, unrealistic human-object penetration, and robot-infeasible poses.

Most existing deployment pipelines follow a generate-then-retarget paradigm. This introduces secondary errors and extra runtime, and often ignores task-level interaction feasibility. In dynamic HOI scenarios (e.g., hitting or dribbling a ball), pose similarity alone is insufficient: the policy must also account for object state, contact timing, and reachability to ensure successful interaction.

For practical robot deployment, we focus on **single-robot mimic** of **human-object interaction (HOI)** motions, instead of two-human interaction mimic with two robots.


## 2. Problem Setting

### Input
- A text instruction describing human-object interaction.
- Optional initial human pose and object state.

Example:
`"A person approaches a box, lifts it, and places it on a table."`

### Output
- A physically valid human-object motion sequence that is executable by a humanoid robot in simulation.

## 3. Core Contributions

1. **Structured Interaction Modeling**
   We represent HOI with explicit phase/contact/object-state structure via an interaction reasoner and embodiment-aware encoding, instead of direct pose-only generation.

2. **Task-Oriented Objective Design**
   We optimize task completion signals (contact feasibility, timing correctness, and outcome consistency) rather than only kinematic similarity.

3. **Human-Robot Motion Co-Generation (Primary Novelty)**
   We jointly generate human-object trajectories and robot-executable targets via a **Cross-Embodiment Co-Generation** mechanism with a shared interaction latent and an Embodiment Bridge in a robot-trackable target space `Y_trackable`, modeling `p(s_h, s_o, y_r | x)` rather than `p(y_r | s_h, s_o)`.  
   **Method-level novelty**: bidirectional embodiment alignment where robot-trackable targets constrain HOI generation and HOI trajectories constrain robot targets, not a post-hoc retarget step.
   **Necessity**: human motions are often infeasible for a specific robot (e.g., reach distance exceeds arm length), so human generation must be influenced by robot feasibility; co-generation enforces this coupling during generation instead of a late-stage correction.

4. **Executable HOI via Physics and Closed-Loop Evaluation**
   We enforce physical and robot-feasibility constraints during training and validate executability through simulation rollout and REIS.

## 4. Method Overview

Main method (6 core modules):

1. **Embodiment Graph Encoder**
- `URDF/MJCF/USD -> unified graph -> x_robot`.

2. **Shared Interaction Latent Predictor**
- Predict unified interaction latent `z_int` (contact anchors, object goals, phase timing, interaction mode).

3. **Structured Interaction Token Generator**
- Generate fixed token slots `T_int={t_1,...,t_K}` from `z_int` for shared interaction conditioning.

4. **Cross-Embodiment Co-Generation Transformer**
- Joint generator for human/object (`s_h`, `s_o`) and robot-trackable targets (`y_r ∈ Y_trackable`) conditioned on `x`, `z_int`, and `T_int`.

5. **Embodiment Bridge**
- `Bridge(s_h, s_o) -> Y_trackable` aligns generated interaction dynamics with robot-trackable targets during training.

6. **Object-Centric Interaction Field**
- Relative descriptors (`r_hand_obj`, `v_rel`) for interaction-invariant supervision.

Additional constraint:
- **Reachability-aware interaction prior** `R(anchor, x_robot)` supervises whether sampled contact anchors are executable for the current embodiment.

Optional module:
- **Interaction Reasoner** is only retained when it predicts high-value signals
  (contact anchors/intercept windows), not only phase labels.
- Hand-control convention: default `6`-DoF `active_joint` policy output with runtime mimic expansion to `12` revolute joints; `12`-DoF independent hand control is ablation-only.

Implementation details are documented in [STORM.md](./STORM.md).

At-a-glance pipeline:

```
x -> z_int -> T_int -> {s_h, s_o, y_r in Y_trackable}
                     \-> B_phi(s_h, s_o) (training-time alignment)
```

This structure separates:
- method identity: joint co-generation under shared intent,
- optimization support: bridge-based embodiment alignment during training.

## 5. Training Objective

Method-level formulation:

```
p(s_h, s_o, y_r | x) = ∫ p(z_int | x) p(s_h, s_o, y_r | z_int, x) dz_int
```

This is the only probabilistic factorization used in this document.

Optimization-level objective:

```
L_base = L_motion + L_interaction + L_outcome + L_robot + L_physics
L_total = L_base + λ_b * L_bridge + λ_g * L_bridge_gt
```

Interpretation:
- `L_base` learns motion quality, interaction structure, physical plausibility, and robot feasibility.
- `L_bridge` couples generated HOI and generated robot targets.
- `L_bridge_gt` grounds bridge predictions to physically valid projected targets.

- `L_motion`: trajectory reconstruction/denoising quality.
- `L_interaction`: contact timing consistency + object-centric relation alignment.
- `L_outcome`: task outcome consistency with explicit contact-time supervision (`t_contact`, `t_release`, contact state, post-object state, stability).
- `L_robot`: robot feasibility (joint limits, trackability, and reachability). Reachability is implemented as a sub-term of `L_robot` in this formulation.
- `L_physics`: penetration, foot-sliding, and balance constraints.
- `L_bridge`: consistency `||y_hat_r - sg(B_phi(s_hat_h, s_hat_o))||^2` (updates generator side).
- `L_bridge_gt`: grounded bridge supervision `||B_phi(s_h^gt, s_o^gt) - y_r^gt||^2` (updates bridge side).

Detailed schedules and implementation notes are specified in [STORM.md](./STORM.md).

## 6. Datasets and Baselines

### Datasets
- **Primary text-to-HOI dataset**: [InterAct / InterAct-X](https://github.com/wzyabcas/InterAct)  
  Dataset access form: <https://docs.google.com/forms/d/e/1FAIpQLScMCfdd8BXzDBZ3iw0x5zA3KSTlD1F2GTaO8ylDG9Cj1upaPw/viewform?usp=sharing>
- **Auxiliary HOI dataset**: [OMOMO](https://github.com/lijiaman/omomo_release) (for physical prior and robot compatibility pretraining)  
  Dataset download: <https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing>

### Baselines
- [InterAct](https://github.com/wzyabcas/InterAct) (*InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation*).
- [InterMimic](https://github.com/Sirui-Xu/InterMimic) (*InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions*).
- [HOI-Diff](https://github.com/neu-vi/HOI-Diff) (*HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models*).
- [InterDiff](https://github.com/Sirui-Xu/InterDiff) (*InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion*) (physics-oriented HOI baseline).
- [SkillMimic](https://arxiv.org/abs/2408.15270) (*SkillMimic: Learning Basketball Interaction Skills from Demonstrations*).
- Generate-then-Retarget pipeline (strong robot-execution baseline): [HOI-Diff](https://github.com/neu-vi/HOI-Diff) (*HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models*) + [InterMimic](https://github.com/Sirui-Xu/InterMimic) (*InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions*).

## 7. Evaluation Protocol

### Simulation Environments
- NVIDIA Isaac Lab
- MuJoCo

### Evaluation Pipeline
```
Generated Human-Object Motion
            ↓
Minimal Kinematic Projection
            ↓
Robot Mimic Controller
            ↓
Simulation Rollout
```

### Metrics
- Motion Quality: realism, diversity, text alignment.
- Interaction Quality: contact timing accuracy, synchronization error.
- Physical Validity: foot sliding, penetration rate, balance violation.
- Robot Executability: success rate, tracking error, stability/fall rate.

## 8. Robot-Executable Interaction Score (REIS)

We define:
```
REIS = α * Q_motion
     + β * S_exec
     + γ * (1 - E_track)
     + δ * P_stability
```

All components are normalized to `[0, 1]`.
We also report sensitivity analysis over `α, β, γ, δ`.


## 9. Related Work

### 9.1 Text-Conditioned and Diffusion-Based HOI Generation
Recent HOI generation methods mainly focus on human-object motion realism under text or historical-motion conditions.  
[HOI-Diff](https://github.com/neu-vi/HOI-Diff) (*HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models*) decomposes generation into human/object motion diffusion and affordance-guided interaction correction, improving contact plausibility.  
[InterDiff](https://github.com/Sirui-Xu/InterDiff) (*InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion*) targets future HOI prediction and introduces physics-informed correction during denoising to reduce floating and penetration artifacts.  
These methods provide strong kinematic HOI priors, but they do not directly optimize robot executability in closed-loop control.

### 9.2 HOI Datasets and Benchmarks
Data quality and scale are key bottlenecks for HOI modeling.  
[InterAct](https://github.com/wzyabcas/InterAct) (*InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation*) consolidates multiple HOI datasets, applies unified correction/augmentation, and defines six benchmark tasks for versatile HOI generation.  
Its contribution is mainly on data infrastructure and multi-task evaluation; however, benchmark objectives still prioritize HOI generation quality rather than direct humanoid execution.

### 9.3 Physics-Based Mimic and Humanoid Control
Physics-based imitation methods improve motion deployability on robots, typically through retarget-then-track pipelines.  
[InterMimic](https://github.com/Sirui-Xu/InterMimic) (*InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions*) uses a teacher-student RL framework to recover and imitate noisy HOI motions in simulation.  
[InterReal](*InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills*, arXiv:2603.07516) is a recent example that targets real-world HOI tracking with contact-consistent motion augmentation and automatic reward weighting, emphasizing deployment robustness rather than text-conditioned generation.  
[SkillMimic](https://arxiv.org/abs/2408.15270) (*SkillMimic: Learning Basketball Interaction Skills from Demonstrations*) learns basketball interaction skills from demonstrations via a unified HOI imitation reward and a contact-graph reward, improving dynamic-object skill composition through high-level control.  
[FRoM-W1](https://openmoss.github.io/FRoM-W1) (*FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions*), [GenMimic](https://genmimic.github.io/) (*From Generated Human Videos to Physically Plausible Robot Trajectories*), and [RLPF](https://beingbeyond.github.io/RLPF/) (*RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control*) further push language/video-driven humanoid control, but they still rely on multi-stage conversion and retargeting pipelines in most settings.  
[RoboGhost](https://arxiv.org/abs/2510.14952) (*From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance*) explores retargeting-free latent-conditioned control for locomotion-focused tasks, showing the value of tighter coupling between generation and control.

### 9.4 Positioning of STORM
Existing HOI works are strong at interaction generation but weak on robot executability; existing robot-control works are strong at tracking/deployment but often weak on HOI-specific interaction reasoning.  
STORM targets this gap with a unified formulation that combines:
- interaction reasoning over human-object phase/contact dynamics,
- shared interaction latent + structured token conditioning,
- interaction-aware physical constraints during training,
- human-robot co-generation in shared robot-trackable space,
- and a robot-executable evaluation protocol (REIS) for end-to-end assessment.

Compared with generate-then-retarget baselines, STORM reduces pipeline fragmentation and accumulated errors while preserving text alignment, HOI realism, and simulation-level robot feasibility.  
Unlike retarget-centric pipelines that mainly optimize kinematic similarity, STORM explicitly models object-state and contact-feasibility constraints. This is particularly important for dynamic HOI (e.g., ball interaction), where post-hoc remapping is fragile and hard to fix via simple post-processing or domain randomization.

Reviewer-facing summary sentence:
**STORM jointly generates human-object interaction and robot-executable motion targets under shared interaction intent and embodiment constraints.**
