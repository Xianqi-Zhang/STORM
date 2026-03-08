# STORM: Structured Task-Oriented Object-Interaction Motion for Robot Mimicry

## 1. Motivation

Recent text-conditioned motion generation methods produce visually plausible motions, but many outputs are not directly usable for robot mimic learning. Common failures include foot sliding, unstable balance, unrealistic human-object penetration, and robot-infeasible poses.

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

1. **Interaction Reasoning for HOI**
   We model initiation-response dynamics between the human body and object state, including contact timing and phase transitions.

2. **Interaction-Aware Physical Modeling**
   We enforce differentiable HOI physics constraints during training, including contact consistency, collision avoidance, and balance stability.

3. **Human-Robot Co-Generation**
   We generate motion in a shared kinematic representation and apply minimal kinematic projection to robot joint space, avoiding expensive optimization-based post-hoc retargeting.

4. **Robot-Executable Evaluation Protocol**
   We define a simulation-based protocol and a composite score (REIS) for robot executability.

## 4. Method Overview

Pipeline:
```
Text Instruction + Object State
            ↓
        Text Encoder
            ↓
   HOI Interaction Reasoning
            ↓
   HOI Motion Generator (Human + Object)
            ↓
   Physical Constraint Training Losses
            ↓
  Human-Robot Co-Generation / Projection
            ↓
    Robot Mimic Controller (Simulation)
```

## 5. Training Objective

We use a multi-term loss:
```
L = λ1 * L_motion
  + λ2 * L_interaction
  + λ3 * L_physics
  + λ4 * L_robot
```

Where:
- `L_motion`: reconstruction / denoising quality.
- `L_interaction`: interaction reasoning consistency (phase/contact timing).
- `L_physics`: contact consistency + collision penalty + balance constraint.
- `L_robot`: robot joint-limit, reachability, and trackability regularization.

## 6. Datasets and Baselines

### Datasets
- **Primary text-to-HOI dataset**: [InterAct / InterAct-X](https://github.com/wzyabcas/InterAct)  
  Dataset access form: <https://docs.google.com/forms/d/e/1FAIpQLScMCfdd8BXzDBZ3iw0x5zA3KSTlD1F2GTaO8ylDG9Cj1upaPw/viewform?usp=sharing>
- **Auxiliary HOI dataset**: [OMOMO](https://github.com/lijiaman/omomo_release) (for physical prior and robot compatibility pretraining)  
  Dataset download: <https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing>

### Baselines
- [InterAct](https://github.com/wzyabcas/InterAct).
- [InterMimic](https://github.com/Sirui-Xu/InterMimic).
- [HOI-Diff](https://github.com/neu-vi/HOI-Diff).
- [InterDiff](https://github.com/Sirui-Xu/InterDiff) (physics-oriented HOI baseline).
- Generate-then-Retarget pipeline (strong robot-execution baseline): [HOI-Diff](https://github.com/neu-vi/HOI-Diff) + [InterMimic](https://github.com/Sirui-Xu/InterMimic).

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
