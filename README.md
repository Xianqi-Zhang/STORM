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







