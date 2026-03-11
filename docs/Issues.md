


## Problem Issues

### InterMimic

#### Import Error.
- File: InterMimic/isaaclab/src/intermimic_lab/intermimic_env.py
```
# from isaaclab.sim.utils.prims import bind_visual_material, bind_physics_material, is_prim_path_valid

from isaaclab.sim.utils.prims import bind_visual_material, bind_physics_material
from isaaclab.sim.utils.legacy import is_prim_path_valid
```

- File: InterMimic/isaacgym/src/intermimic/env/tasks/intermimic_all.py
```
# from torch.func import vmap

from functorch import vmap
```

#### Training Out of Memory.

- File: InterMimic/isaacgym/train_teacher_multigpu.sh
```sh
#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Safer defaults for memory-heavy InterMimic training.
# You can override via environment variables, e.g.:
#   NUM_GPUS=1 NUM_ENVS=256 HORIZON_LENGTH=16 MINIBATCH_SIZE=4096 sh ...
NUM_GPUS=${NUM_GPUS:-1}
NUM_ENVS=${NUM_ENVS:-256}
HORIZON_LENGTH=${HORIZON_LENGTH:-16}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-4096}

if [ "$NUM_GPUS" -gt 1 ]; then
    MULTI_GPU_FLAG="--multi_gpu"
else
    MULTI_GPU_FLAG=""
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --num_envs "$NUM_ENVS" \
    --horizon_length "$HORIZON_LENGTH" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --headless \
    --output checkpoints \
    $MULTI_GPU_FLAG \
    "$@"
```

---


### InterAct

#### Data Process Error. 
```bash
python process/process_grab.py
Traceback (most recent call last):
  File "/media/zhangxq/Workspace/STORM/storm/baselines/InterAct/process/process_grab.py", line 7, in <module>
    from render.mesh_utils import Mesh
ModuleNotFoundError: No module named 'render.mesh_utils'; 'render' is not a package
```

```python
# process/process_grab.py

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT2INTERACTION_DIR = os.path.join(ROOT_DIR, "text2interaction")
if TEXT2INTERACTION_DIR not in sys.path:
    sys.path.insert(0, TEXT2INTERACTION_DIR)
from render.mesh_utils import Mesh
```


