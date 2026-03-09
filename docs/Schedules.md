# STORM Schedule (Execution Checklist)

## Status Legend
- `[x]` done
- `[~]` in progress / partial
- `[ ]` not started

## Phase 0: Environment and Data Readiness
- [ ] Verify Python environment and dependencies (`torch`, `numpy`, `pyyaml`).
- [ ] Verify dataset locations (`DATASETS/OMOMO`, InterAct assets if available).
- [ ] Run baseline data conversion smoke test (InterAct -> InterMimic replay path).
- [x] Freeze one robot embodiment for V1 (Unitree G1).

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
- [ ] Replace synthetic dataset with real InterAct/OMOMO dataloader.
- [ ] Replace random embodiment graph with URDF/MJCF/USD parser.
- [ ] Implement true critical-window constrained optimizer (not proxy smoothing).
- [ ] Implement recovery execution loop in simulation (not classifier-only).

## Phase 3: STORM V1 Training and Ablation
- [ ] Train Stage A (core + interaction field) on real data.
- [ ] Train Stage B (enable constrained refinement) on real data.
- [ ] Train Stage C (enable failure recovery) with perturbation rollouts.
- [ ] Run required ablations:
  - [ ] No Outcome-Centric head
  - [ ] No interaction field loss (`L_rel`)
  - [ ] No critical-window refinement
  - [ ] No failure recovery head
- [ ] Compare against generate-then-retarget baseline.

## Phase 4: Report and Next Iteration Gate
- [ ] Report REIS and component metrics.
- [ ] Confirm "different poses, same outcome" on static and dynamic tasks.
- [ ] Decide whether to add perception-heavy modules in V2.

## Immediate Next Steps (Execution Order)
1. [ ] Install dependencies and run synthetic smoke training (`scripts/train_storm.py`).
2. [ ] Build InterAct/OMOMO dataset adapter in `src/storm/data/`.
3. [ ] Implement embodiment parser from `src/assets/unitree_g1/*.xml` into graph tensors.
4. [ ] Add baseline smoke scripts and metric logging.
5. [ ] Launch Stage A on real data and validate losses/metrics.

## Current V1 Scope Freeze
- Tasks: box carrying (static object), ball catch/intercept (dynamic object).
- Simulator: single simulator only.
- Embodiments: single robot first, multi-embodiment after V1 stability.
- Method: 1 core innovation + 3 supporting modules (as defined in `Doc/STORM.md`).
