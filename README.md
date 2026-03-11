# STORM

**STORM: Structured Task-Oriented Human-Robot Motion Co-Generation for Executable Human-Object Interaction**

**Acronym**  
**STORM** = **S**tructured **T**ask-**O**riented Human-**R**obot **M**otion Co-Generation

## Setup

- Create conda env.
```bash
conda create -n storm python=3.11
conda activate storm
```

- Install packages.
```bash
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge libgcc-ng
conda install -c conda-forge cmake ninja

# * PyTorch will be installed during the IsaacLab installation process..
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

conda install -c conda-forge openmesh

pip install -r requirements.txt
```

- Install IsaacLab
  * Please do not use `pip install`; InterMimic requires installation from source (it requires isaaclab.sh).
```bash
# pip install "isaaclab[isaacsim,all]==2.3.1" --extra-index-url https://pypi.nvidia.com
# pip install git+https://github.com/isaac-sim/rl_games.git@python3.11
```

```bash
git clone git@github.com:isaac-sim/IsaacLab.git

cd IsaacLab
./isaaclab.sh -i

# Add to .zshrc or .bashrc
export ISAACLAB_PATH=/path/to/your/IsaacLab
```

```bash
$ISAACLAB_PATH/isaaclab.sh -p -m pip install --upgrade imageio imageio-ffmpeg
```

## Conda Env for InterMimic

```
conda create -n isaacgym python=3.8
conda activate isaacgym
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install functorch==1.13.1

# * Download IsaacGym from https://developer.nvidia.com/isaac-gym/download
cd isaacgym/python
pip install -e .

cd InterMimic
pip install -r requirement.txt
```


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




