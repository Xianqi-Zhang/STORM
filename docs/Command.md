


## 1. InterMimic

### Data Replay
```bash
./isaaclab/scripts/run_data_replay.sh --num-envs 8 --motion-dir InterAct/OMOMO_new
```

### Train

```bash
# Uses all available GPUs by default
sh isaacgym/scripts/train_teacher_multigpu.sh

# High-fidelity simulation variant
sh isaacgym/scripts/train_teacher_new_multigpu.sh

# MLP-based student policy
sh isaacgym/scripts/train_student_multigpu.sh

# Transformer-based student policy
sh isaacgym/scripts/train_student_transformer_multigpu.sh
```

