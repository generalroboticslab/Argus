
# Train
```bash

`flat ground`: bash run.sh argus_base

## point cloud encoder
### step1: base policy training with object states
`object pushing`: bash run.sh  argus_object_pushing_base \
`object tracking`: bash run.sh argus_object_tracking_base
### step2: off-line point cloud data collection
`object pushing`: bash run.sh  argus_object_pushing_IL \
`object tracking`: bash run.sh argus_object_tracking_IL
### step3: point cloud encoder training
`object pushing`: python train_point_could_encoder.py --task object_pushing \
`object tracking`: python train_point_could_encoder.py --task object_tracking



# baseline
bash run.sh argus_base

# disable leg
bash run.sh argus_disable_leg_dof_20_const_vel
bash run.sh argus_disable_leg_dof_32_const_vel
bash run.sh argus_disable_leg_dof_12_const_vel

# carry object
bash run.sh argus_carry_object_dof_32_const_vel
bash run.sh argus_carry_object_dof_20_const_vel
bash run.sh argus_carry_object_dof_12_const_vel

# push rejection
bash run.sh argus_push

# discrete terrain
bash run.sh argus_terrain_dof_20_const_vel
bash run.sh argus_terrain_dof_32_const_vel
bash run.sh argus_terrain_dof_12_const_vel

#object tracking
##base
bash run.sh argus_object_tracking_base
##IL collect
bash run.sh argus_object_tracking_IL

##32 legs
#base
bash run.sh argus_object_tracking_base_debug_32legs
#IL collect
#0.5m cube
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5 
#0.25m cube
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect 

#asymmetry
#IL collect
bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry_collect

#object pushing
#base
bash run.sh  argus_object_pushing_base
#IL
bash run.sh  argus_object_pushing_IL
```

# eval
```bash
# baseline
bash run.sh argus_base_eval -p

# disable leg
bash run.sh argus_disable_leg_dof_20_const_vel_eval -p
bash run.sh argus_disable_leg_dof_32_const_vel_eval -p
bash run.sh argus_disable_leg_dof_12_const_vel_eval -p

# carry object
bash run.sh argus_carry_object_dof_32_const_vel_eval -p
bash run.sh argus_carry_object_dof_20_const_vel_eval -p
bash run.sh argus_carry_object_dof_12_const_vel_eval -p

# push rejection
bash run.sh argus_push_eval -p

# discrete terrain
bash run.sh argus_terrain_dof_20_const_vel_eval -p
bash run.sh argus_terrain_dof_32_const_vel_eval -p
bash run.sh argus_terrain_dof_12_const_vel_eval -p

#object pushing
##base
bash run.sh argus_object_pushing_base_eval_template -p
## with point cloud
bash run.sh argus_object_pushing_eval_template -p

#object tracking
##base
bash run.sh argus_object_tracking_base_eval_template -p
##with point cloud
bash run.sh argus_object_tracking_eval_template -p

#32 legs
#base
bash run.sh argus_object_tracking_base_debug_32legs_eval -p
#12 perception units
bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval -p
#20 perception units
bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval -p
#32 perception units
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval -p
```