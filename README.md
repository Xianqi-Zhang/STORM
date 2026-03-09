# STORM

**STORM: Structured Task-Oriented Human-Robot Motion Co-Generation for Executable Human-Object Interaction**

**Acronym**  
**STORM** = **S**tructured **T**ask-**O**riented Human-**R**obot **M**otion Co-Generation

## Setup

```bash
conda create -n storm python=3.11
conda activate storm

conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge libgcc-ng
conda install -c conda-forge cmake ninja

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

conda install -c conda-forge openmesh

pip install -r requirements.txt
```
