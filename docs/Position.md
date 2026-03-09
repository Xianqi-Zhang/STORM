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
   We jointly generate human-object trajectories and robot-trackable motion targets in a shared conditional space, with minimal post-hoc projection to reduce retarget-induced error accumulation and latency.

4. **Executable HOI via Physics and Closed-Loop Evaluation**
   We enforce physical and robot-feasibility constraints during training and validate executability through simulation rollout and REIS.

## 4. Method Overview

High-level pipeline:
```
Text + Object State + Robot Embodiment
            ↓
Embodiment Graph Encoder + Interaction Reasoner
            ↓
 Human-Object Generator + Co-Generation Head
            ↓
Task-Oriented Losses (Interaction/Physics/Robot)
            ↓
   Controller-in-the-loop Simulation + REIS
```

Implementation details (network blocks, embodiment encoding, losses, and training stages) are documented in [STORM.md](./STORM.md).

## 5. Training Objective

We optimize motion quality, interaction consistency, physical plausibility, and robot feasibility jointly.

Detailed objective terms, stage-wise schedules, and practical hyper-parameter defaults are specified in [STORM.md](./STORM.md).


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
[SkillMimic](https://arxiv.org/abs/2408.15270) (*SkillMimic: Learning Basketball Interaction Skills from Demonstrations*) learns basketball interaction skills from demonstrations via a unified HOI imitation reward and a contact-graph reward, improving dynamic-object skill composition through high-level control.  
[FRoM-W1](https://openmoss.github.io/FRoM-W1) (*FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions*), [GenMimic](https://genmimic.github.io/) (*From Generated Human Videos to Physically Plausible Robot Trajectories*), and [RLPF](https://beingbeyond.github.io/RLPF/) (*RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control*) further push language/video-driven humanoid control, but they still rely on multi-stage conversion and retargeting pipelines in most settings.  
[RoboGhost](https://arxiv.org/abs/2510.14952) (*From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance*) explores retargeting-free latent-conditioned control for locomotion-focused tasks, showing the value of tighter coupling between generation and control.

### 9.4 Positioning of STORM
Existing HOI works are strong at interaction generation but weak on robot executability; existing robot-control works are strong at tracking/deployment but often weak on HOI-specific interaction reasoning.  
STORM targets this gap with a unified formulation that combines:
- interaction reasoning over human-object phase/contact dynamics,
- interaction-aware physical constraints during training,
- human-robot co-generation with minimal post-hoc projection,
- and a robot-executable evaluation protocol (REIS) for end-to-end assessment.

Compared with generate-then-retarget baselines, STORM reduces pipeline fragmentation and accumulated errors while preserving text alignment, HOI realism, and simulation-level robot feasibility.  
Unlike retarget-centric pipelines that mainly optimize kinematic similarity, STORM explicitly models object-state and contact-feasibility constraints. This is particularly important for dynamic HOI (e.g., ball interaction), where post-hoc remapping is fragile and hard to fix via simple post-processing or domain randomization.
