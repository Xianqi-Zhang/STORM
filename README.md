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

## Problem Issues

### InterMimic

- File: intermimic_env.py
```
# from isaaclab.sim.utils.prims import bind_visual_material, bind_physics_material, is_prim_path_valid

from isaaclab.sim.utils.prims import bind_visual_material, bind_physics_material
from isaaclab.sim.utils.legacy import is_prim_path_valid
```


