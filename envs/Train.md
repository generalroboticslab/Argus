
# Train

## Flat plane
`bash run.sh argus_base`

## Disable leg
Argus: `bash run.sh argus_disable_leg_dof_20_const_vel` \
12 legs version: `bash run.sh argus_disable_leg_dof_12_const_vel`  
32 legs version: `bash run.sh argus_disable_leg_dof_32_const_vel` 

## Carry object
Argus: `bash run.sh argus_carry_object_dof_20_const_vel` \
12 legs version: `bash run.sh argus_carry_object_dof_12_const_vel`\
32 legs version: `bash run.sh argus_carry_object_dof_32_const_vel`

## Push rejection
`bash run.sh argus_push`

## Discrete terrain
Argus:`bash run.sh argus_terrain_dof_20_const_vel`\
32 legs version:`bash run.sh argus_terrain_dof_32_const_vel`\
12 legs version:`bash run.sh argus_terrain_dof_12_const_vel`

## Object tracking/Pushing

### step1: base policy training with object states
Object pushing: `bash run.sh argus_object_pushing_base` \
Oobject tracking: `bash run.sh argus_object_tracking_base`
### step2: off-line point cloud data collection
Object pushing: `bash run.sh  argus_object_pushing_IL` \
Object tracking: `bash run.sh argus_object_tracking_IL`
### step3: point cloud encoder training
Object pushing: `python train_point_could_encoder.py --task object_pushing` \
Object tracking: `python train_point_could_encoder.py --task object_tracking`


## Object tracking - 32 legs with different number of perception units
### step1: base policy training with object states
`bash run.sh argus_object_tracking_base_32legs`
### step2: off-line point cloud data collection
0.5m cube:`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5`  
0.25m cube:`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25` \
asymmetric peception units - 50cm cube:`bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry` \
asymmetric peception units - 25cm cube:`bash run.sh argus_object_tracking_IL_offline_cube025_asymmetry`

### step3: point cloud encoder training
*Data are collected with 32 perception units, but down sampled for 20 and 12 perception units during supervised learning.\
`python train_point_could_encoder.py --task object_tracking --num_perception_units <select one:32/20/12>`


# Play your checkpoint
## Locomotion and Object pushing
To play your checkpoint, add `-p` after the bash command. For example, `bash run.sh argus_base -p` for the flat plane rolling. You also need to specify your checkpoint path in `argus_base` under `PLAY_ARGS` in the `exp.sh` file.

## Object tracking
`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5 -p`\
`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25 -p` \
*please specify your encoder checkpoint and the number of peception units in the `exp.sh`
file. The trained encoder checkpoint are listed under `assets/checkpoint/argus_object_tracking`.


# Evaluation

Evaluation is based on the same code used for training and playing. The relavent evaluation data will be saved in `.json` file under `envs/eval` folder. 

## Flat plane
`bash run.sh argus_base_eval -p`

## Disable leg
`bash run.sh argus_disable_leg_dof_20_const_vel_eval -p` \
`bash run.sh argus_disable_leg_dof_32_const_vel_eval -p` \
`bash run.sh argus_disable_leg_dof_12_const_vel_eval -p`

## Carry object
`bash run.sh argus_carry_object_dof_32_const_vel_eval -p` \
`bash run.sh argus_carry_object_dof_20_const_vel_eval -p` \
`bash run.sh argus_carry_object_dof_12_const_vel_eval -p` 

## Push rejection
`bash run.sh argus_push_eval -p`

## Discrete terrain
`bash run.sh argus_terrain_dof_20_const_vel_eval -p` \
`bash run.sh argus_terrain_dof_32_const_vel_eval -p` \
`bash run.sh argus_terrain_dof_12_const_vel_eval -p`

## Object pushing

### base
`bash run.sh argus_object_pushing_base_eval -p`
### with point cloud observation
`bash run.sh argus_object_pushing_eval -p`

## Object tracking

### base
`bash run.sh argus_object_tracking_base_eval -p`
### with point cloud
`bash run.sh argus_object_tracking_eval -p`

## Object tracking - 32 legs
### base
`bash run.sh argus_object_tracking_base_32legs_eval -p`
### 12 perception units
`bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_5_eval -p` \
`bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval -p`
### 20 perception units
`bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_5_eval -p` \
`bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval -p`
### 32 perception units
`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval -p` \
`bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval -p`
## Object tracking asymmetric perception
*please specify the asymmetric design and the corredponding encoder under the `PLAY_ARGS`. \
`bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry_eval -p` \
`bash run.sh argus_object_tracking_IL_offline_cube025_asymmetry_eval -p`
