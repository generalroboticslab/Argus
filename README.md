# Extreme Symmetry Enables Omnidirectional and Multifunctional Robotics
### Project Website: tbd

This repo includes the code for the Argus robot policy training.
<div align="left">
    <img src="visualization/Argus_multi-functional.png" width="400">
</div>

## Installation
Please install [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and following packages.
```
conda create --name argus python=3.8
conda activate argus
pip install -r requirements.txt --no-cache-dir
```

## Quick Start!
Run following checkpoint to check how Argus moves!
### Flat ground rolling

`bash run.sh argus_base -pk`

🎮 Keyboard Controls  
- ⬆️ `i` → forward (+0.05m/s)  
- ⬇️ `k` → backward (-0.05m/s)  
- ⬅️ `j` → left (+0.05m/s)  
- ➡️ `l` → right (-0.05m/s)  
---
### Object interaction
Object pushing: `bash run.sh argus_object_pushing_IL -p` \
Object tracking: `bash run.sh argus_object_tracking_IL -p` 

Object pushing rendering:
<div align="left">
    <video src="visualization/object_pushing.mp4" width="640" controls autoplay loop>
    </video>
</div>


## Training Instructions
All training command can be found in [Train.md](Train.md)

## BibTeX

If you find our paper or codebase helpful, please consider citing:


## Acknowledgement
`This work is supported by DARPA FoundSci program under award HR00112490372, DARPA TIAMAT program under award HR00112490419, ARO under award W911NF2410405, ARL STRONG program under awards W911NF2320182, W911NF2220113, and W911NF242021, and by gift supports from BMW.`