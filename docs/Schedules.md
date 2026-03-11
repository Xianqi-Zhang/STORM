# STORM Schedule (Execution Checklist)

## Status Legend
- `[x]` done
- `[~]` in progress / partial
- `[ ]` not started

## Phase 0: Environment and Data Readiness

### 0.1 Enviroment
- [x] Verify Python environment and dependencies (`torch`, `numpy`, `pyyaml`).
- [ ] Verify dataset locations (`DATASETS/OMOMO`, InterAct assets if available).
- [ ] Run baseline data conversion smoke test (InterAct -> InterMimic replay path).
- [x] Freeze one robot embodiment for V1 (Unitree G1).

### 0.2 Datasets
- [x] behave
- [x] grab
- [] intercap
- [x] omomo
- [x] parahome
- [x] arctic

## Phase 1: Baseline Smoke Benchmark
- [ ] Run minimal HOI-Diff sampling on a small prompt set.
- [ ] Replay samples through InterMimic tracking pipeline.
- [ ] Record baseline metrics: success rate, penetration ratio, contact timing error, fall rate.
- [ ] Export a compact baseline report under `Doc/`.

## Phase 2: STORM V1 Implementation
- [x] Implement Outcome-Centric Co-Generation core module (scaffold).
- [x] Implement Object-Centric Interaction Field module (scaffold).
- [x] Implement Critical-Window Constrained Refinement module (scaffold).
- [x] Implement Failure-Aware Recovery module (scaffold).
- [x] Implement unified STORM V1 objective and config (scaffold).
- [x] Add training entrypoint and synthetic smoke dataset.
- [ ] Define and implement `Y_trackable` target schema (root/EE/contact/phase).
- [ ] Add interaction token generator (`T_int`) with fixed slot count `K`.
- [ ] Add bridge stabilization losses: `L_bridge` (stop-gradient consistency) + `L_bridge_gt` (grounded supervision).
- [ ] Add mixed GT/generated conditioning for bridge inputs.
- [ ] Add projection validity/confidence weighting for `y_r^gt` supervision.
- [ ] Add reachability scorer `R(anchor, x_robot)` and integrate as sub-term in `L_robot`.
- [ ] Add explicit contact-time supervision (`t_contact`, `t_release`) in `L_outcome`.
- [ ] Replace synthetic dataset with real InterAct/OMOMO dataloader.
- [ ] Replace random embodiment graph with URDF/MJCF/USD parser.
- [ ] Implement true critical-window constrained optimizer (not proxy smoothing).
- [ ] Implement recovery execution loop in simulation (not classifier-only).

## Phase 3: STORM V1 Training and Ablation
- [ ] Train Stage A (core + interaction field) on real data.
- [ ] Train Stage B (enable constrained refinement) on real data.
- [ ] Train Stage C (enable failure recovery) with perturbation rollouts.
- [ ] Run required ablations:
  - [ ] No `Y_trackable` (direct full-pose robot prediction)
  - [ ] No Embodiment Bridge
  - [ ] No interaction tokens (`K=0`)
  - [ ] Reduced token count (`K=4`) vs full (`K=8`)
  - [ ] Token shuffle perturbation at test time
  - [ ] No `L_bridge_gt` (consistency-only bridge)
  - [ ] No reachability prior (`L_reach`)
  - [ ] No explicit contact-time term (`L_contact_time`)
  - [ ] No Outcome-Centric head
  - [ ] No interaction field loss (`L_rel`)
  - [ ] No critical-window refinement
  - [ ] No failure recovery head
- [ ] Compare against generate-then-retarget baseline.

## Optional Innovation Upgrades (Post-V1)
- [ ] Add feasibility predictor head `F(s_h, s_o, y_r, x_robot)` and `L_feas`.
- [ ] Add contrastive cross-view consistency (positive/negative HOI-robot pairs).
- [ ] Add intervention-style contact robustness training (timing/anchor perturbations; no formal causal-identification claim).
- [ ] Run novelty-focused ablations:
  - [ ] `+L_feas` vs baseline STORM
  - [ ] L2-only bridge vs contrastive bridge
  - [ ] without vs with contact intervention robustness

## Phase 4: Report and Next Iteration Gate
- [ ] Report REIS and component metrics.
- [ ] Confirm "different poses, same outcome" on static and dynamic tasks.
- [ ] Decide whether to add perception-heavy modules in V2.

## Immediate Next Steps (Execution Order)
1. [ ] Install dependencies and run synthetic smoke training (`scripts/train_storm.py`).
2. [ ] Implement `Y_trackable` API and logging in `src/storm/models/`.
3. [ ] Build InterAct/OMOMO dataset adapter in `src/storm/data/`.
4. [ ] Implement interaction-token module (`K` fixed, slot-based conditioning).
5. [ ] Implement embodiment parser from `src/storm/assets/robots/unitree_g1/*.xml` into graph tensors.
6. [ ] Add bridge warmup stage (`L_bridge_gt`) then joint stage (`L_total`) with `λ_b` ramp.
7. [ ] Add reachability scorer and contact-time labels to training batches.
8. [ ] Add baseline smoke scripts and metric logging.
9. [ ] Launch Stage A on real data and validate losses/metrics.
10. [ ] (Optional Post-V1) Implement feasibility head and novelty ablations.

## Current V1 Scope Freeze
- Tasks: box carrying (static object), ball catch/intercept (dynamic object).
- Simulator: single simulator only.
- Embodiments: single robot first, multi-embodiment after V1 stability.
- Method: 1 core innovation + 3 supporting modules (as defined in `Doc/STORM.md`).
