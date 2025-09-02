
argus_base(){
# bash run.sh argus_base -pk
    ENTRY_POINT=ppo_isaacgym.py
    PLAY_ARGS=(
        --checkpoint=../assets/checkpoint/flat_plane.pt
        --train_mode=play
        --num_envs=1
        --headless=False
        --track=False
        task.env.dataPublisher.enable=True
        ++task.env.renderFPS=50
        task.env.randomize.control_pd.enable=false
        task.env.randomize.baseMass.enable=false
        task.env.randomize.baseInertiaOrigin.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.dof_strength.enable=false
        task.env.randomize.default_dof_pos.enable=false
        task.env.randomize.link_mass.enable=false
        task.env.randomize.link_inertia.enable=false
        task.env.randomize.body_force.enable=false
    )
    TRAIN_ARGS=(
        --train_mode=train
        --headless=True
        --track=True
        --bounds_loss_coef=0
    )
    BASE_ARGS=(
        --num_envs=8192
        --num_steps=8
        --task_name=argus
        --wandb_entity=grl_argus
        --wandb_run_name=baseline
        --agent_name=baseline
        "++task.env.defaultJointPositions=0"
        "++task.env.initialJointPositions=-0.105"
        "++task.env.desiredJointPositions=-0.105"
        task.env.randomCommandVelocityRanges.linear_x=[-0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[-0.8,0.8]
    )
    KEYBOARD_ARGS=(
        task.env.viewer.keyboardOperator=True
    )
}

argus_disable_leg(){
    argus_base
    PLAY_ARGS+=(
        --checkpoint=''
        --num_envs=16
        task.env.dataPublisher.enable=False
    )
    BASE_ARGS+=(
        --wandb_run_name=disable_leg
        task.env.randomize.dof_disable.enable=True
        task.env.randomize.dof_disable.probability=0.1
        task.env.learn.reward.action.scale=0.1 # smaller
    )
}

argus_disable_leg_dof_20_const_vel(){
    # bash run.sh argus_disable_leg_dof_20_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_disable_leg/argus_disable_leg_dof_20_const_vel.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_20_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_20_const_vel
    ) 
}

argus_disable_leg_dof_12_const_vel(){
    # bash run.sh argus_disable_leg_dof_12_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_disable_leg/argus_disable_leg_dof_12_const_vel.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_12_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_12_const_vel
    )
}


argus_disable_leg_dof_32_const_vel(){
    # bash run.sh argus_disable_leg_dof_32_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_disable_leg/argus_disable_leg_dof_32_const_vel.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_32_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_32_const_vel
    )
} 


argus_carry_object(){
    # bash run.sh argus_carry_object -p
    argus_base
    PLAY_ARGS+=(
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/train/carry_object_20250419_222447/carry_object.pt
        --num_envs=1
        ++task.env.evaluate.enable=True
        --num_envs=8192
        --headless=True
        task.env.learn.episodeLength_s=3
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.urdfAsset.env.AssetOptions.override_inertia=True
        task.env.randomize.baseInertiaOrigin.enable=True
        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[0,40]
        task.env.terrain.terrainType=plane # use plane for evaluation
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        task.env.randomize.baseInertiaOrigin.range=[[-0.05,0.05],[-0.05,0.05],[-0.05,0.05]]
        task.env.randomize.baseMass.range=[0,10]
        ++task.env.evaluate.filename=eval/argus_carry_object.json
        --wandb_run_name=argus_carry_object
    )
}


argus_carry_object_dof_20_const_vel(){
    # bash run.sh argus_carry_object_dof_20_const_vel -p
    argus_carry_object
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_carry_object/argus_carry_object_dof_20_const_vel.pt
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_20_const_vel.json
        --headless=False
        --num_envs=16
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_20_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_20_const_vel
    )
}


argus_carry_object_dof_12_const_vel(){
    # bash run.sh argus_carry_object_dof_12_const_vel -p
    argus_carry_object
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_carry_object/argus_carry_object_dof_12_const_vel.pt
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        --headless=False
        --num_envs=16
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_12_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_12_const_vel
    )
}


argus_carry_object_dof_32_const_vel(){
    # bash run.sh argus_carry_object_dof_32_const_vel -p
    argus_carry_object
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_carry_object/argus_carry_object_dof_32_const_vel.pt
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        --headless=False
        --num_envs=16
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_32_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_32_const_vel
    )
}


argus_push(){
    # bash run.sh argus_push -p
    argus_base
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_push/push.pt
        task.env.randomCommandVelocityRanges.linear_x=[0,0]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.baseHeightOffset=1
        task.env.randomize.push.velMin=[1.5,0,0,0,0,0]
        task.env.randomize.push.velMax=[1.5,0,0,0,0,0]
        task.env.baseHeightOffset=1
        --num_envs=16
        task.env.learn.episodeLength_s=5
        task.env.dataPublisher.enable=False
    )
    BASE_ARGS+=(
        --wandb_run_name=push
        task.env.randomize.push.enable_at_reset=True
        task.env.baseHeightOffset=0.5
        task.env.randomize.push.velMin=[-2,-2,-2,-2,-2,-2]
        task.env.randomize.push.velMax=[2,2,2,2,2,2]
    )
}


argus_terrain(){
    # bash run.sh argus_terrain -pk
    argus_base
    PLAY_ARGS+=(
        task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
        task.env.terrain.discrete.height=0.1
        task.env.terrain.numTerrains=1
        task.env.terrain.numLevels=5
        task.env.dataPublisher.enable=False
        task.env.learn.episodeLength_s=15
        --num_envs=8
        --checkpoint=../envs/runs/argus_debug/train/discrete_terrain_dof20_20250614_174701/discrete_terrain_dof20_newest.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=discrete_terrain_dof20
        task.env.terrain.terrainType=heightfield
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        --num_envs=4096
        --num_steps=16
    )
}

argus_terrain_dof_20_const_vel(){
    # bash run.sh argus_terrain_dof_20_const_vel -p
    argus_terrain
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/discrete_terrain/argus_terrain_dof_20_const_vel.pt
        --num_envs=64
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_terrain_dof_20_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        --wandb_run_name=argus_terrain_dof_20_const_vel
    ) 
}

argus_terrain_dof_32_const_vel(){
    # bash run.sh argus_terrain_dof_32_const_vel -p
    argus_terrain
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/discrete_terrain/argus_terrain_dof_32_const_vel.pt
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_terrain_dof_32_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        --wandb_run_name=argus_terrain_dof_32_const_vel
    )
} 

argus_terrain_dof_12_const_vel(){
    # bash run.sh argus_terrain_dof_12_const_vel -p
    argus_terrain
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/discrete_terrain/argus_terrain_dof_12_const_vel.pt
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_terrain_dof_12_const_vel.json
        --wandb_run_name=argus_terrain_dof_12_const_vel
    )
}



#object pushing 

argus_object_pushing_IL(){
    # bash run.sh argus_object_pushing_IL -p
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1
        --encoder_checkpoint=../assets/checkpoint/argus_object_pushing/object_pushing_encoder.pt
        --checkpoint=../assets/checkpoint/argus_object_pushing/argus_object_pushing_base.pt
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_orientation.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        --agent_name=mixed_ray
        task.env.ray_obs.visualize_ray_point_cloud=True
        --headless=False
        task.env.ray_obs.random_dropout=0.0
        task.env.ray_obs.random_noise=0.0
        task.env.ray_obs.num_step_to_delay_ray=0
        # task.env.save_blender_trajectory=True
    )

    TRAIN_ARGS+=(
        --num_envs=16
        --train_mode=collect
        --checkpoint=../assets/checkpoint/argus_object_pushing/argus_object_pushing_base.pt
        --seed=42
    )

    BASE_ARGS+=(
        --wandb_run_name=argus_object_pushing
        task.env.ray_obs.enable=True
        task.env.objectPushing.enable=True
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_goal_velocity,object_orientation]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,object_velocity,object_goal_velocity,robot_root_position,object_root_state]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.learn.reward.robot_object_distance.scale=0.8
        task.env.terrain.staticFriction=0.1  # [-]
        task.env.terrain.dynamicFriction=0.1  # [-]
        task.env.randomize.friction.range=[0.1,0.3]
        task.env.objectPushing.cube_y_position=1.5
        task.env.learn.reward.dof_force_target.scale=0.04
        task.env.learn.reward.dof_vel.scale=0.01
        task.env.learn.reward.action.scale=0.02
        task.env.learn.reward.action_rate.scale=0.02
        task.env.randomize.base_init_pos.range=[[-0.4,0.4],[-0.6,0.3],[0,0]]
        task.env.learn.episodeLength_s=8
        task.env.learn.reward.lin_vel.exp_scale=-12.0
        task.env.learn.reward.orientation_along_command_direction.scale=2
        task.env.objectPushing.goal_vel=0.6
        )
}

argus_object_pushing_base(){
    # bash run.sh argus_object_pushing_base -p
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1
        --checkpoint=../assets/checkpoint/argus_object_pushing/argus_object_pushing_base.pt
        task.env.dataPublisher.enable=false
    )

    TRAIN_ARGS+=(
        --num_envs=16384
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_pushing
        task.env.ray_obs.enable=False
        task.env.objectPushing.enable=True
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_goal_velocity,object_orientation]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,object_velocity,object_goal_velocity,robot_root_position,object_root_state]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.learn.reward.robot_object_distance.scale=0.8
        task.env.terrain.staticFriction=0.1  # [-]
        task.env.terrain.dynamicFriction=0.1  # [-]
        task.env.randomize.friction.range=[0.1,0.3]
        task.env.objectPushing.cube_y_position=1.5
        task.env.learn.reward.dof_force_target.scale=0.04
        task.env.learn.reward.dof_vel.scale=0.01
        task.env.learn.reward.action.scale=0.02
        task.env.learn.reward.action_rate.scale=0.02
        task.env.randomize.base_init_pos.range=[[-0.4,0.4],[-0.6,0.3],[0,0]]
        task.env.learn.episodeLength_s=15
        task.env.learn.reward.lin_vel.exp_scale=-12.0 
        task.env.learn.reward.orientation_along_command_direction.scale=2
        task.env.objectPushing.goal_vel=0.6
    )
}

#object tracking

argus_object_tracking_IL(){
    # bash run.sh argus_object_tracking_IL -p
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --headless=False
        --num_envs=8
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=True
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base.pt
        --encoder_checkpoint=../assets/checkpoint/argus_object_tracking/object_tracking_encoder.pt
        --agent_name=mixed_ray
    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=object_tracking
        task.env.ray_obs.enable=True
        task.env.objectTracking.enable=True
        task.env.objectTracking.velocity_range=[0.5,0.8]
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_velocity]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position,object_velocity]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        task.env.control.actionScale=0.1
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
    )
}


argus_object_tracking_base(){
    # bash run.sh argus_object_tracking_base -p
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base.pt
        task.env.dataPublisher.enable=false
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking
        task.env.ray_obs.enable=False
        task.env.objectTracking.enable=True
        task.env.objectTracking.velocity_range=[0.5,0.8]
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_velocity]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position,object_velocity]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        task.env.control.actionScale=0.1
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
    )
}



argus_object_tracking_base_32legs(){
    # bash run.sh argus_object_tracking_base_32legs -p
    argus_base
    PLAY_ARGS+=(
        --num_envs=1
        task.env.objectTracking.velocity_range=[0.5,0.8]
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_orientation.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
    )
    TRAIN_ARGS+=(
        --num_envs=16384
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs
        task.env.ray_obs.enable=False
        task.env.objectTracking.enable=True
        task.env.objectTracking.velocity_range=[0.5,0.8]
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_velocity]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position,object_velocity]"
        task.env.envSpacing=2
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.urdfAsset.file='urdf/argus/argus_dof32_minimum.urdf'
        task.env.learn.reward.dof_force_target.scale=0.0
        task.env.learn.reward.dof_vel.scale=0.0
        task.env.learn.reward.dof_vel_computed.scale=0.0
        task.env.learn.reward.dof_acc_computed.scale=0.0
        task.env.learn.reward.dof_pow.scale=0.0
        task.env.learn.reward.dof_absolute_position.scale=0.0
        task.env.learn.reward.dof_limit.scale=0.0
        task.env.learn.reward.impact.scale=0.0
        task.env.learn.reward.slip.scale=0.0
        task.env.learn.reward.action.scale=0.0
        task.env.learn.reward.action_rate.scale=0.0

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_5(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5 -p
    argus_object_tracking_base_32legs
    PLAY_ARGS+=(
        --num_envs=4
        --seed=43
        --headless=False
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base_32legs.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=True

        # to specify which encoder to use
        --encoder_checkpoint=../assets/checkpoint/argus_object_tracking/immitation_20perception_05cube_encoder/imitation_model_best_val.pt
        task.env.ray_obs.num_perception_units=20
    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base_32legs.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_IL_offline_32legs_32percetion_cube0_5
        task.env.ray_obs.enable=True
        task.env.ray_obs.num_perception_units=32
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.objectTracking.cube_size=0.5
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s50cm_d20.urdf'

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_25(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25 
    argus_base
    argus_object_tracking_base_32legs
        PLAY_ARGS+=(
        # ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base_32legs.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        # ++task.env.evaluate.filename=eval/argus_object_tracking_32perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        --seed=43

        # to specify which encoder to use
        --encoder_checkpoint=../assets/checkpoint/argus_object_tracking/immitation_20perception_025cube_encoder/imitation_model_best_val.pt
        task.env.ray_obs.num_perception_units=20

    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=False
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base_32legs.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_IL_offline_32legs_32percetion_cube0_25
        task.env.ray_obs.enable=True
        task.env.ray_obs.num_perception_units=32
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.objectTracking.cube_size=0.25
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s25cm_d20.urdf'
    )
}


argus_object_tracking_IL_offline_cube05_asymmetry(){
    # bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry
    argus_base
    argus_sim2real_dynamics_setup_template
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        --total_timesteps=51200
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base.pt
        --seed=42
        task.env.perception_asymmetry_experiment.data_collection=True
        #specific asymmetry urdf
        task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0230_argus_dof20_minimum.urdf'
        --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0230_argus_dof20_minimum
    )
    BASE_ARGS+=(
        task.env.ray_obs.enable=True
        task.env.objectTracking.enable=True
        task.env.objectTracking.velocity_range=[0.5,0.8]
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_velocity]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position,object_velocity]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        task.env.control.actionScale=0.1
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.perception_asymmetry_experiment.enable=True
        task.env.objectTracking.cube_size=0.5
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s50cm_d20.urdf'
        task.env.objectTracking.objectRobotInitialDistance=1.25
    )
}

argus_object_tracking_IL_offline_cube025_asymmetry(){
    # bash run.sh argus_object_tracking_IL_offline_cube025_asymmetry
    argus_base
    argus_sim2real_dynamics_setup_template
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        --total_timesteps=51200
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base.pt
        --seed=42
        task.env.perception_asymmetry_experiment.data_collection=True
        #specific asymmetry urdf
        task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0230_argus_dof20_minimum.urdf'
        --wandb_run_name=argus_object_tracking_IL_offline_cube025_asymmetry_collect_sim_rand_joint_0230_argus_dof20_minimum
    )
    BASE_ARGS+=(
        task.env.ray_obs.enable=True
        task.env.objectTracking.enable=True
        task.env.objectTracking.velocity_range=[0.5,0.8]
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_velocity]"
        "task.env.stateNames=[linearVelocity,worldSpaceAngularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position,object_velocity]"
        task.env.envSpacing=2
        task.env.control.decimation=10
        task.env.control.actionScale=0.1
        "task.env.num_stacked_obs_frame=1"
        "task.env.num_stacked_state_frame=1"
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.perception_asymmetry_experiment.enable=True
        task.env.objectTracking.cube_size=0.5
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s25cm_d20.urdf'
        task.env.objectTracking.objectRobotInitialDistance=1.25
    )
}



argus_base_eval(){
    # bash run.sh argus_base_eval -p
    argus_base
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.filename=eval/argus_base.json
        --num_envs=8192
        task.env.learn.episodeLength_s=3
        --total_timesteps=12288000 # equivalent to 81920 sample
        --headless=True
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[-0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[-0.8,0.8]
        "task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf"
    )
}

argus_disable_leg_eval_template(){
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=8192
        --headless=True
        task.env.learn.episodeLength_s=5
        --total_timesteps=20480000
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
    )
}

argus_disable_leg_dof_20_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_20_const_vel_eval -p
    argus_disable_leg_dof_20_const_vel
    argus_disable_leg_eval_template
}

argus_disable_leg_dof_32_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_32_const_vel_eval -p
    argus_disable_leg_dof_32_const_vel
    argus_disable_leg_eval_template
}

argus_disable_leg_dof_12_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_12_const_vel_eval -p
    argus_disable_leg_dof_12_const_vel
    argus_disable_leg_eval_template
}

argus_carry_object_eval_template(){
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=8192
        --headless=True
        task.env.learn.episodeLength_s=3
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.urdfAsset.env.AssetOptions.override_inertia=True
        task.env.randomize.baseInertiaOrigin.enable=True
        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[0,40]
        task.env.terrain.terrainType=plane # use plane for evaluation
    )
}

argus_carry_object_dof_12_const_vel_eval(){
    # bash run.sh argus_carry_object_dof_12_const_vel_eval -p
    argus_carry_object_dof_12_const_vel
    argus_carry_object_eval_template
}

argus_carry_object_dof_20_const_vel_eval(){
    # bash run.sh argus_carry_object_dof_20_const_vel_eval -p
    argus_carry_object_dof_20_const_vel
    argus_carry_object_eval_template
}

argus_carry_object_dof_32_const_vel_eval(){
    # bash run.sh argus_carry_object_dof_32_const_vel_eval -p
    argus_carry_object_dof_32_const_vel
    argus_carry_object_eval_template
}


argus_push_eval(){
    # bash run.sh argus_push_eval -p
    argus_push
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.filename=argus_push.json
        --num_envs=8192
        --headless=True
        task.env.learn.episodeLength_s=5
        task.env.baseHeightOffset=0
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0,0]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomize.push.velMin=[-5,-5,0,0,0,0]
        task.env.randomize.push.velMax=[5,5,0,0,0,0]
    )
}


argus_terrain_eval_template(){
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        task.env.terrain.minInitMapLevel=0
        task.env.terrain.maxInitMapLevel=10
        task.env.terrain.numLevels=10
        task.env.terrain.numTerrains=80
        task.env.terrain.discrete.height=0.1
        --num_envs=1024
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        --total_timesteps=2048000
    )
}

argus_terrain_dof_12_const_vel_eval(){
    # bash run.sh argus_terrain_dof_12_const_vel_eval -p
    argus_terrain_dof_12_const_vel
    argus_terrain_eval_template
}

argus_terrain_dof_32_const_vel_eval(){
    # bash run.sh argus_terrain_dof_32_const_vel_eval -p
    argus_terrain_dof_32_const_vel
    argus_terrain_eval_template
}

argus_terrain_dof_20_const_vel_eval(){
    # bash run.sh argus_terrain_dof_20_const_vel_eval -p
    argus_terrain_dof_20_const_vel
    argus_terrain_eval_template
}


argus_object_pushing_base_eval(){
    # bash run.sh argus_object_pushing_base_eval -p
    argus_object_pushing_base
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.shape_evaluate=True
        --num_envs=512
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_pushing_base.json
        task.env.ray_obs.visualize_ray_point_cloud=False

    )
}

argus_object_pushing_eval(){
    # bash run.sh argus_object_pushing_eval -p
    argus_object_pushing_IL
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_pushing.json
        task.env.ray_obs.visualize_ray_point_cloud=False

    )
}

argus_object_tracking_base_eval(){
    # bash run.sh argus_object_tracking_base_eval -p
    argus_object_tracking_base
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_base.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.objectTracking.distance_reset_threshold=5
        task.env.objectTracking.velocity_range=[0.6,0.6]
    )
}


argus_object_tracking_eval(){
    # bash run.sh argus_object_tracking_eval -p
    argus_object_tracking_IL
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.objectTracking.distance_reset_threshold=5
        task.env.objectTracking.velocity_range=[0.6,0.6]
    )
}

argus_object_tracking_base_32legs_eval(){
    # bash run.sh argus_object_tracking_base_32legs_eval -p
    argus_object_tracking_base_32legs
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --checkpoint=../assets/checkpoint/argus_object_tracking/argus_object_tracking_base_32legs.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_base_32perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
    )
}

argus_object_tracking_IL_offline_32legs_12percetion_cube0_5_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_5_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_5
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_12perception_50cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=12
        --seed=43
    )
}

argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_12perception_25cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=12
        --seed=43
    )
}



argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_20perception_25cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=20
        --seed=43

    )
}

argus_object_tracking_IL_offline_32legs_20percetion_cube0_5_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_5_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_5
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_20perception_50cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=20
        --seed=43

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_32perception_25cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=32
        --seed=43

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_5
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_32perception_50cube.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=32
        --seed=43

    )
}


argus_object_tracking_IL_offline_cube05_asymmetry_eval(){
    # bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry_eval -p
    argus_object_tracking_IL_offline_cube05_asymmetry
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        task.env.learn.episodeLength_s=5
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=False
        --seed=43

        #specify the asymmetric design to eval
        --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/5x5_pointcloud_immitation__baseline__20250731_230922_20perception_05cube_aymmetrydesign_073/imitation_model_best_val.pt # To change
        ++task.env.evaluate.filename=eval/argus_object_tracking_asymmetry_073.json # To change
        task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0073_argus_dof20_minimum.urdf' # To change
        --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0073_argus_dof20_minimum # To change
    )
}


argus_object_tracking_IL_offline_cube025_asymmetry_eval(){
    # bash run.sh argus_object_tracking_IL_offline_cube025_asymmetry_eval -p
    argus_object_tracking_IL_offline_cube025_asymmetry
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/5x5_pointcloud_immitation__baseline__20250731_230922_20perception_05cube_aymmetrydesign_073/imitation_model_best_val.pt # To change
        task.env.learn.episodeLength_s=5
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=False
        --seed=43

        #specify the asymmetric design to eval
        --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/5x5_pointcloud_immitation__baseline__20250731_230922_20perception_05cube_aymmetrydesign_073/imitation_model_best_val.pt # To change
        ++task.env.evaluate.filename=eval/argus_object_tracking_asymmetry_073.json # To change
        task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0073_argus_dof20_minimum.urdf' # To change
        --wandb_run_name=argus_object_tracking_IL_offline_cube025_asymmetry_collect_sim_rand_joint_0073_argus_dof20_minimum # To change
    )
}