argus_climbup_smooth_wall_real_fixed_base_01(){ # in sim it is moving fast. 🤖➡️🚶
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01 -p
    argus_climbup
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_20250722_191421/argus_climbup_smooth_wall_real_fixed_base_01_newest.pt

        # ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls_2m_long.npz'
        # # task.env.sim2realDataPublisher.enable=true
        # task.env.dataReceiver.enable=true
        # task.env.randomize.base_init_pos.range=[[0,0],[0,0],[0,0]]
        # task.env.gravity_support.end_acc=6 # TEST ONLY
        # task.env.learn.episodeLength_s=999
        # task.env.baseInitState.rot=[-0.5210,+0.0218,+0.2495,0.815961]

        # task.env.randomize.friction.enable=false
        # task.env.randomize.control_pd.enable=false
        # task.env.randomize.baseMass.enable=false
        # task.env.randomize.baseInertiaOrigin.enable=false
        # task.env.randomize.push.enable=false
        # task.env.randomize.initDofPos.enable=false
        # task.env.randomize.initDofVel.enable=false
        # task.env.randomize.erfi.enable=false
        # task.env.randomize.dof_strength.enable=false
        # task.env.randomize.default_dof_pos.enable=false
        # task.env.randomize.link_mass.enable=false
        # task.env.randomize.link_inertia.enable=false
        # task.env.randomize.body_force.enable=false
        # task.env.learn.addNoise=false

        task.env.dataPublisher.enable=true
        task.env.gravity_support.end_acc=7.1
        
        --num_envs=1
        ++task.env.renderFPS=25
        # task.env.randomize.base_init_pos.range=[[-0,0],[0,0],[0,0]]
    )
    # TRAIN_ARGS+=(
    #     --headless=False
    # )

    BASE_ARGS+=(     
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01

        # task.env.control.dynamic_action_bound.enable=false
        task.env.control.dynamic_action_bound.enable=true
        task.env.control.dynamic_action_bound.alpha=0.95
        # task.env.control.dynamic_action_bound.bound=0.1
        task.env.control.dynamic_action_bound.bound=0.15 # higher range.

        # task.env.control.actionScale=0.25
        # task.env.control.limit=300


        task.env.gravity_support.start_acc=8
        task.env.gravity_support.end_acc=6.8

        task.env.learn.reward.dof_vel_non_contact.scale=0
        task.env.learn.reward.dof_limit.scale=-2

        task.env.learn.addNoise=True # noisy observation
        task.env.max_observation_delay_steps=1 # 1 step delay
        task.env.randomize.erfi.enable=true
        task.env.randomize.default_dof_pos.enable=true

        task.env.num_stacked_obs_frame=1
        task.env.control.decimation=8
        # task.env.assetDofProperties.damping=20
        # task.env.control.damping=100
        # task.env.control.integral=100

        # task.env.baseInitState.rot=[0.0,0.258815,0.0,0.965959]
        # task.env.baseInitState.rot=[-0.1206,0.2670,-0.2462,0.9239]

        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls_0.98_1.0_1.2m_long.npz'
        task.env.randomize.base_init_orientation.enable=False # added
        task.env.randomize.base_init_pos.range=[[-5000,5000],[-0.05,0.05],[0,3]]

        task.env.randomize.action_delay.enable=false
        # task.env.randomize.action_delay.range=[0.1,0.1]


    )
}

argus_climbup_smooth_wall_real_fixed_base_01_gentle(){ 
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_gentle -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_gentle_20250722_221125/argus_climbup_smooth_wall_real_fixed_base_01_gentle.pt   
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_gentle
        task.env.control.dynamic_action_bound.bound=0.11
    )
}

argus_climbup_smooth_wall_real_fixed_base_01_gentle_vel_constraint(){ 
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_gentle_vel_constraint -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_gentle_vel_constraint_20250722_223752/argus_climbup_smooth_wall_real_fixed_base_01_gentle_vel_constraint_newest.pt

    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_gentle_vel_constraint
        task.env.control.dynamic_action_bound.bound=0.12
        task.env.learn.reward.lin_vel.normalize_by=[0.05,0.05,1]
    )
}


argus_climbup_smooth_wall_real_fixed_base_01_with_vel_constraint(){ # 
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_with_vel_constraint -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_with_vel_constraint_20250722_221158/argus_climbup_smooth_wall_real_fixed_base_01_with_vel_constraint.pt

    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_with_vel_constraint
        task.env.learn.reward.lin_vel.normalize_by=[0.05,0.05,1]
    )
}

argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact(){ # OK, SLIGHTLY aggressive 🤖➡️🚶
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact_20250718_122936/argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact_newest.pt
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact_20250718_122936/argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_vel_constraint_dof_vel_non_contact
        task.env.learn.reward.lin_vel.normalize_by=[0.1,0.1,1]
        task.env.learn.reward.dof_vel_non_contact.scale=0.1
    )
}

argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact(){ # 🤖➡️🚶
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact_20250718_153430/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact_newest.pt
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact_20250718_154328/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact.pt
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact_20250718_154328/argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_random_dof_strength_vel_constraint_dof_vel_non_contact
        task.env.learn.reward.lin_vel.normalize_by=[0.1,0.1,1]
        task.env.learn.reward.dof_vel_non_contact.scale=0.1
        task.env.randomize.dof_strength.range=[0.95,1.0] # will make it more aggressive
    )
}


argus_climbup_smooth_wall_real_fixed_base_01_multi_obs(){ # ok in sim, a bit aggressive, TODO test in real 🤖➡️🚶
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_multi_obs -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_20250718_010703/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_newest.pt
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_20250718_010703/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_newest.pt
    )
    BASE_ARGS+=(
        task.env.num_stacked_obs_frame=5
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_multi_obs
    )
}

argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_with_vel_constraint(){ # aggresive
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_with_vel_constraint -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_with_vel_constraint_20250718_010624/argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_with_vel_constraint_newest.pt
    )
    BASE_ARGS+=(
        task.env.num_stacked_obs_frame=5
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_multi_obs_with_vel_constraint
        task.env.learn.reward.lin_vel.normalize_by=[0.5,0.5,1]
    )
}


argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2(){
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2 -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2_20250716_205802/argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2_newest.pt
        
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2_20250716_211534/argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2_newest.pt
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2_newest.pt
        
        task.env.gravity_support.end_acc=7.1 # moon
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_gravity_mars_control_damping_100_gentle_2
        task.env.learn.reward.action.scale=0.2
        task.env.learn.reward.action_rate.scale=0.2
        # task.env.learn.reward.dof_vel.scale=0.2 # 
        task.env.learn.reward.dof_vel.scale=1 # 
        task.env.learn.reward.lin_vel.scale=4 #
        # task.env.learn.reward.lin_vel.scale=2 # too small
        task.env.gravity_support.end_acc=6.1
        task.env.gravity_support.curriculum_threshold=0.1

        # firction randomization 
        task.env.randomize.friction.enable=True
        # task.env.control.actionScale=0.2


    )
}


argus_climbup_smooth_wall_real_fixed_base_01_gravity_5_control_damping_100_gentle_3(){
    # bash run.sh argus_climbup_smooth_wall_real_fixed_base_01_gravity_5_control_damping_100_gentle_3 -p
    argus_climbup_smooth_wall_real_fixed_base_01
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/argus_climbup_smooth_wall_real_fixed_base_01_gravity_5_control_damping_100_gentle_3_20250716_020846/argus_climbup_smooth_wall_real_fixed_base_01_gravity_5_control_damping_100_gentle_3_newest.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_fixed_base_01_gravity_5_control_damping_100_gentle_3
        task.env.learn.reward.action.scale=0.2
        task.env.learn.reward.action_rate.scale=0.2
        task.env.learn.reward.dof_vel.scale=0.2 # 
        task.env.learn.reward.lin_vel.scale=4 #
        # task.env.learn.reward.lin_vel.scale=2 # too small
        task.env.gravity_support.end_acc=5
        task.env.gravity_support.curriculum_threshold=0.1
        # ++task.physx.sim.max_depenetration_velocity=5 # BAD
    )
}


####################################################3

argus_object_pushing_base_eval_template(){
    # bash run.sh argus_object_pushing_base_eval_template -p
    argus_object_pushing_base_debug_2
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

argus_object_tracking_base_eval_template(){
    # bash run.sh argus_object_tracking_base_eval_template -p
    argus_object_tracking_base_debug_1
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

argus_object_pushing_eval_template(){
    # bash run.sh argus_object_pushing_eval_template -p
    argus_object_pushing_IL_debug_2
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


argus_object_tracking_eval_template(){
    # bash run.sh argus_object_tracking_eval_template -p
    argus_object_tracking_IL_offline_debug_1
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


#final version
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


argus_object_pushing_base_simulation(){
    # bash run.sh argus_object_pushing_base_simulation -p
    argus_base
    # argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1

        --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_debug/argus_object_pushing_20250703_163723/argus_object_pushing.pt
        task.env.dataPublisher.enable=false
        # task.env.randomize.base_init_orientation.enable=false
        # task.env.randomize.base_init_pos.enable=false
        # task.env.randomize.initDofPos.enable=false
        # task.env.randomize.initDofVel.enable=false

    )
    TRAIN_ARGS+=(
        --num_envs=16384
        # --track=False
        # --headless=False

    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_pushing
        task.env.ray_obs.enable=False
        task.env.objectPushing.enable=True
        "task.env.observationNames=[worldSpaceAngularVelocity,dofPosition,dofVelocity,actions,base_rotation_matrix_filtered,object_goal_velocity,object_velocity,object_orientation,contact]"
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
        task.env.randomize.base_init_pos.range=[[-0.4,0.4],[-0.0,0.3],[0,0]]
        task.env.learn.episodeLength_s=10
        task.env.learn.reward.lin_vel.exp_scale=-12.0 #-16 work for goal_vel 0.5
        task.env.learn.reward.orientation_along_command_direction.scale=0.5
        task.env.objectPushing.goal_vel=0.6

    )
}


argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250722_210955_12perception_025cube/imitation_model_best_val.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_12perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=12
        --seed=43
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
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250716_100613_12perception_05cube/imitation_model_0_1200_0.006321416236460209.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_12perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=12
        --seed=43


    )
}

argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250724_100020_20perception_025cube/imitation_model_best_val.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_20perception.json
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
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250718_100114_20perception_05cube/imitation_model_0_1200_0.006348154507577419.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_20perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=20
        --seed=43

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250724_100020_20perception_025cube/imitation_model_best_val.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_32perception.json
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
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250720_141754_32perception_05cube/imitation_model_best_val.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_32perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        task.env.ray_obs.num_perception_units=32
        --seed=43

    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval_blendner(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval_blendner -p
    argus_object_tracking_IL_offline_32legs_32percetion_cube0_5
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=False
        --num_envs=1
        --headless=False
        --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=../assets/checkpoint/5x5_pointcloud_immitation__baseline__20250720_141754_32perception_05cube/imitation_model_best_val.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=False
        task.env.randomize.base_init_pos.enable=False
        task.env.randomize.initDofPos.enable=False
        task.env.randomize.initDofVel.enable=False
        task.env.ray_obs.visualize_ray_point_cloud=True
        task.env.ray_obs.num_perception_units=12
        --seed=43
        task.env.save_blender_trajectory=True
        task.env.ray_obs.static_debug=True
        task.env.ray_obs.random_dropout=0
        task.env.ray_obs.apply_noise_on_point_cloud=False
        task.env.randomize.base_init_pos.range=[[-0.0,0.0],[-0.0,0.0],[0,0]]
    )
}



argus_object_tracking_base_debug_32legs_eval(){
    # bash run.sh argus_object_tracking_base_debug_32legs_eval -p
    argus_object_tracking_base_debug_32legs
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        --num_envs=512
        --headless=True
        # --agent_name=mixed_ray
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        # --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/5x5_pointcloud_immitation__baseline__20250713_005945_32perception/imitation_model_best.pt
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomize.base_init_pos.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        ++task.env.evaluate.filename=eval/argus_object_tracking_base_32perception.json
        task.env.ray_obs.visualize_ray_point_cloud=False
        # task.env.objectTracking.velocity_range=[0.6,0.6]

        # task.env.ray_obs.num_perception_units=32


    )
}

argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect 
    argus_base
    argus_object_tracking_base_debug_32legs
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=False
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_collect
        task.env.ray_obs.enable=True
        task.env.ray_obs.num_perception_units=32
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.objectTracking.cube_size=0.25
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s25cm_d20.urdf'
    )
}


argus_object_tracking_IL_offline_32legs_32percetion_cube0_75(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_75 -p
    argus_base
    argus_object_tracking_base_debug_32legs
    PLAY_ARGS+=(
        --headless=False
        --num_envs=8
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=True
        --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/5x5_pointcloud_immitation__baseline__20250722_001405_32perception_075cube/imitation_model_best_val.pt
        --agent_name=mixed_ray
    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        # --headless=False
        # task.env.dataPublisher.enable=false
        # task.env.ray_obs.visualize_ray_point_cloud=True
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_IL_offline_32legs_32percetion_cube0_75
        task.env.ray_obs.enable=True
        task.env.ray_obs.num_perception_units=32
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]
        task.env.objectTracking.cube_size=0.75
        task.env.urdfAsset.cube_asset='urdf/cube/track_cube_s75cm_d20.urdf'

    )
}


argus_object_tracking_IL_offline_32legs_32percetion_cube0_5(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5 -p
    argus_base
    argus_object_tracking_base_debug_32legs
    PLAY_ARGS+=(
        --headless=False
        --num_envs=8
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=True
        # --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/immitation__baseline__20250620_144614/imitation_model_199.pt
        # --checkpoint=/home/jl1099/vrobot_env/assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --encoder_checkpoint=runs/argus_debug/imitation_model_2_4500_0.0048337699845433235.pt #large pointnet
        # --encoder_checkpoint=../assets/checkpoint/imitation_model_0_600_0.03113037347793579.pt  # small pointnet
        --agent_name=mixed_ray
        # task.env.randomize.base_init_orientation.enable=false
        # task.env.randomize.base_init_pos.enable=false
        # task.env.randomize.initDofPos.enable=false
        # task.env.randomize.initDofVel.enable=false
        task.env.objectTracking.distance_reset_threshold=5
        task.env.ray_obs.num_perception_units=12

    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        # --headless=False
        # task.env.dataPublisher.enable=false
        # task.env.ray_obs.visualize_ray_point_cloud=True
        # --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
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


argus_object_tracking_IL_offline_32legs_32percetion(){
    # bash run.sh argus_object_tracking_IL_offline_32legs_32percetion -p
    argus_base
    argus_object_tracking_base_debug_32legs
    PLAY_ARGS+=(
        --headless=False
        --num_envs=8
        task.env.dataPublisher.enable=false
        task.env.ray_obs.visualize_ray_point_cloud=True
        # --encoder_checkpoint=/home/jl1099/vrobot_env/envs/runs/immitation__baseline__20250620_144614/imitation_model_199.pt
        # --checkpoint=/home/jl1099/vrobot_env/assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_20250628_105707/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs.pt
        --encoder_checkpoint=runs/argus_debug/imitation_model_2_4500_0.0048337699845433235.pt #large pointnet
        # --encoder_checkpoint=../assets/checkpoint/imitation_model_0_600_0.03113037347793579.pt  # small pointnet
        --agent_name=mixed_ray
        # task.env.randomize.base_init_orientation.enable=false
        # task.env.randomize.base_init_pos.enable=false
        # task.env.randomize.initDofPos.enable=false
        # task.env.randomize.initDofVel.enable=false
        task.env.objectTracking.distance_reset_threshold=5
        task.env.ray_obs.num_perception_units=12

    )
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        # --headless=False
        # task.env.dataPublisher.enable=false
        # task.env.ray_obs.visualize_ray_point_cloud=True
        # --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt
        --seed=42
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_object_tracking_IL_offline_32legs_32percetion_cube_1_0
        task.env.ray_obs.enable=True
        task.env.ray_obs.num_perception_units=32
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-0.25,0.25],[-0.25,0.25],[0,0]]

    )
}





argus_object_tracking_base_debug_32legs(){
    # bash run.sh argus_object_tracking_base_debug_32legs -p
    argus_base
    # argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1
        task.env.objectTracking.velocity_range=[0.5,0.8]
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_20250628_105707/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs.pt
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


argus_object_tracking_IL_offline_cube05_asymmetry_eval(){
    # bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry_eval -p
    argus_object_tracking_IL_offline_cube05_asymmetry_collect
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

        ++task.env.evaluate.filename=eval/argus_object_tracking_asymmetry_073.json # To change
        task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0073_argus_dof20_minimum.urdf' # To change
        --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0073_argus_dof20_minimum # To change
    )
}

# Bin Index  Bin Range  Sampled Index  Potential Energy collect  IL    Eval
                                    # 511     LOWEST    ✅      ✅    ✅
                                    # 0       Argus     ✅      ✅    ✅
#          0 [150, 160)            230        153.337118 collecting on server1
#          0 [150, 160)            158        152.025556 collecting on server1
#          0 [150, 160)            312        157.148825 collecting on server1
#          1 [160, 170)            350        160.119511✅      ✅    ✅
#          1 [160, 170)             52        160.268208✅      ✅    ✅  
#          1 [160, 170)             22        166.835847✅      ✅    ✅
#          2 [170, 180)            255        176.363857✅      ✅    ✅
#          2 [170, 180)            143        170.041157✅      ✅    ✅
#          2 [170, 180)            323        176.662258✅      ✅    ✅
#          3 [180, 190)            436        180.708670✅      ✅    ✅
#          3 [180, 190)              2        189.575208✅      ✅    ✅
#          3 [180, 190)            406        184.468209✅      ✅    ✅
#          4 [190, 200)            441        193.810098✅      ✅    ✅
#          4 [190, 200)             10        194.790787✅      ✅    ✅
#          4 [190, 200)            419        198.708055✅      ✅    ✅
#          5 [200, 210)            286        202.860039✅      ✅    ✅
#          5 [200, 210)            199        200.311460✅      ✅    ✅
#          5 [200, 210)            126        201.206067✅      ✅   ✅
#          6 [210, 220)            291        214.321646✅      ✅  ✅
#          6 [210, 220)             40        216.052540✅      ✅    ✅
#          6 [210, 220)             73        218.250071✅      ✅    ✅

argus_object_tracking_IL_offline_cube05_asymmetry_collect(){
    # bash run.sh argus_object_tracking_IL_offline_cube05_asymmetry_collect
    argus_base
    argus_sim2real_dynamics_setup_template
    TRAIN_ARGS+=(
        --num_envs=512
        --train_mode=collect
        --total_timesteps=51200
        # --headless=False
        # --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --checkpoint=../assets/checkpoint/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_20250614_095931/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe.pt
        --seed=42
        task.env.perception_asymmetry_experiment.data_collection=True

        # task.env.ray_obs.visualize_ray_point_cloud=True

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0511_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0511_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0255_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0255_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0441_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0441_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0436_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0436_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0286_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0286_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0000_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0000_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0350_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0350_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0002_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0002_argus_dof20_minimum

        # task.env.urdfAsset.asymmetry_design_asset='urdf/argus_sim_rand_joint/sim_rand_joint_0052_argus_dof20_minimum.urdf'
        # --wandb_run_name=argus_object_tracking_IL_offline_cube05_asymmetry_collect_sim_rand_joint_0052_argus_dof20_minimum

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


# 🟢 testing this one
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
        --num_envs=16
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
    TRAIN_ARGS+=(

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





argus_disable_leg_debug_2_const_vel_06_08(){ # OK IN SIM, check in real
    # bash run.sh argus_disable_leg_debug_2 -p
    argus_disable_leg_debug_1
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/disable_leg_debug_2.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=disable_leg_debug_2
        task.env.control.decimation=8
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
    )
}






argus_terrain_const_vel_debug_2(){ # GOOD verified in real 👍
    # bash run.sh argus_terrain_const_vel_debug_2 -p
    argus_terrain
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        # # good
        # --checkpoint=../assets/checkpoint/argus_terrain_const_vel_debug_2_20250607_213102/argus_terrain_const_vel_debug_2.pt
        # --checkpoint=../envs/runs/argus_debug/train/argus_terrain_const_vel_debug_2_20250610_134412/argus_terrain_const_vel_debug_2_newest.pt
        --checkpoint=../assets/checkpoint/argus_terrain_const_vel_debug_2_newest.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_terrain_const_vel_debug_2
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
    )
}


argus_terrain_debug_2(){ # 
    # bash run.sh argus_terrain_debug_2 -p
    argus_terrain
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_debug/train/argus_terrain_debug_2_20250610_140046/argus_terrain_debug_2_newest.pt
        # task.env.dataPublisher.enable=True
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_terrain_debug_2
    )
}


#######

argus_terrain_combined_dof_12_const_vel_eval(){
    # bash run.sh argus_terrain_combined_dof_12_const_vel_eval -p
    argus_terrain_combined_dof_12_const_vel
    argus_terrain_eval_template
}

argus_terrain_combined_dof_32_const_vel_eval(){
    # bash run.sh argus_terrain_combined_dof_32_const_vel_eval -p
    argus_terrain_combined_dof_32_const_vel
    argus_terrain_eval_template
}

argus_terrain_combined_dof_20_const_vel_eval(){
    # bash run.sh argus_terrain_combined_dof_20_const_vel_eval -p
    argus_terrain_combined_dof_20_const_vel
    argus_terrain_eval_template
}

argus_terrain_combined_dof_12_const_vel(){
    # bash run.sh argus_terrain_combined_dof_12_const_vel -p
    argus_terrain_dof_12_const_vel
    argus_terrain_combined_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_combined_dof_12_const_vel_20250623_140622/argus_terrain_combined_dof_12_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_combined_dof_12_const_vel.json
        --wandb_run_name=argus_terrain_combined_dof_12_const_vel
    )
}

argus_terrain_combined_dof_32_const_vel(){
    # bash run.sh argus_terrain_combined_dof_32_const_vel -p
    argus_terrain_dof_32_const_vel
    argus_terrain_combined_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_combined_dof_32_const_vel_20250620_134022/argus_terrain_combined_dof_32_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_combined_dof_32_const_vel.json
        --wandb_run_name=argus_terrain_combined_dof_32_const_vel
    )
}

argus_terrain_combined_dof_20_const_vel(){
    # bash run.sh argus_terrain_combined_dof_20_const_vel -p
    argus_terrain_dof_20_const_vel
    argus_terrain_combined_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_combined_dof_20_const_vel_20250620_134018/argus_terrain_combined_dof_20_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_combined_dof_20_const_vel.json
        --wandb_run_name=argus_terrain_combined_dof_20_const_vel
    )
}

argus_terrain_combined_template(){
    PLAY_ARGS+=(
        task.env.terrain.numTerrains=10
    )
    BASE_ARGS+=(
        task.env.terrain.terrainProportions=[1,1,1,1,1,1,1,3,0]
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.slopeTreshold=0.8
    )
}

#----- rough slope
argus_terrain_rough_slope_dof_12_const_vel_eval(){
    # bash run.sh argus_terrain_rough_slope_dof_12_const_vel_eval -p
    argus_terrain_rough_slope_dof_12_const_vel
    argus_terrain_eval_template
}

argus_terrain_rough_slope_dof_32_const_vel_eval(){
    # bash run.sh argus_terrain_rough_slope_dof_32_const_vel_eval -p
    argus_terrain_rough_slope_dof_32_const_vel
    argus_terrain_eval_template
}

argus_terrain_rough_slope_dof_20_const_vel_eval(){
    # bash run.sh argus_terrain_rough_slope_dof_20_const_vel_eval -p
    argus_terrain_rough_slope_dof_20_const_vel
    argus_terrain_eval_template
}

argus_terrain_rough_slope_dof_12_const_vel(){
    # bash run.sh argus_terrain_rough_slope_dof_12_const_vel -p
    argus_terrain_dof_12_const_vel
    argus_terrain_rough_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_rough_slope_dof_12_const_vel_20250620_133132/argus_terrain_rough_slope_dof_12_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_rough_slope_dof_12_const_vel.json
        --wandb_run_name=argus_terrain_rough_slope_dof_12_const_vel
    )
}

argus_terrain_rough_slope_dof_32_const_vel(){
    # bash run.sh argus_terrain_rough_slope_dof_32_const_vel -p
    argus_terrain_dof_32_const_vel
    argus_terrain_rough_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_rough_slope_dof_32_const_vel_20250620_133128/argus_terrain_rough_slope_dof_32_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_rough_slope_dof_32_const_vel.json
        --wandb_run_name=argus_terrain_rough_slope_dof_32_const_vel
    )
}

argus_terrain_rough_slope_dof_20_const_vel(){
    # bash run.sh argus_terrain_rough_slope_dof_20_const_vel -p
    argus_terrain_dof_20_const_vel
    argus_terrain_rough_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_rough_slope_dof_20_const_vel_20250620_133124/argus_terrain_rough_slope_dof_20_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_rough_slope_dof_20_const_vel.json
        --wandb_run_name=argus_terrain_rough_slope_dof_20_const_vel
    )
}

argus_terrain_rough_slope_template(){
    PLAY_ARGS+=(
        task.env.terrain.numTerrains=2
        task.env.terrain.terrainProportions=[1,0,0,0,0,0,0,0,0]
    )
    BASE_ARGS+=(
        task.env.terrain.terrainProportions=[1,1,0,0,0,0,0,0,0]
        task.env.terrain.terrainType=heightfield
    )
}

#----- mooth slope

argus_terrain_smooth_slope_dof_12_const_vel_eval(){
    # bash run.sh argus_terrain_smooth_slope_dof_12_const_vel_eval -p
    argus_terrain_smooth_slope_dof_12_const_vel
    argus_terrain_eval_template
}

argus_terrain_smooth_slope_dof_32_const_vel_eval(){
    # bash run.sh argus_terrain_smooth_slope_dof_32_const_vel_eval -p
    argus_terrain_smooth_slope_dof_32_const_vel
    argus_terrain_eval_template
}

argus_terrain_smooth_slope_dof_20_const_vel_eval(){
    # bash run.sh argus_terrain_smooth_slope_dof_20_const_vel_eval -p
    argus_terrain_smooth_slope_dof_20_const_vel
    argus_terrain_eval_template
}

argus_terrain_smooth_slope_dof_12_const_vel(){
    # bash run.sh argus_terrain_smooth_slope_dof_12_const_vel -p
    argus_terrain_dof_12_const_vel
    argus_terrain_smooth_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_mooth_slope_dof_12_const_vel_20250620_133120/argus_terrain_mooth_slope_dof_12_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_smooth_slope_dof_12_const_vel.json
        --wandb_run_name=argus_terrain_smooth_slope_dof_12_const_vel
    )
}

argus_terrain_smooth_slope_dof_32_const_vel(){
    # bash run.sh argus_terrain_smooth_slope_dof_32_const_vel -p
    argus_terrain_dof_32_const_vel
    argus_terrain_smooth_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_mooth_slope_dof_32_const_vel_20250620_133117/argus_terrain_mooth_slope_dof_32_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_smooth_slope_dof_32_const_vel.json
        --wandb_run_name=argus_terrain_smooth_slope_dof_32_const_vel
    )
}

argus_terrain_smooth_slope_dof_20_const_vel(){
    # bash run.sh argus_terrain_smooth_slope_dof_20_const_vel -p
    argus_terrain_dof_20_const_vel
    argus_terrain_smooth_slope_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_mooth_slope_dof_20_const_vel_20250620_133113/argus_terrain_mooth_slope_dof_20_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_smooth_slope_dof_20_const_vel.json
        --wandb_run_name=argus_terrain_smooth_slope_dof_20_const_vel
    )
}

argus_terrain_smooth_slope_template(){
    PLAY_ARGS+=(
        task.env.terrain.numTerrains=2
        task.env.terrain.terrainProportions=[0,0,0,0,0,1,0,0,0]
    )
    BASE_ARGS+=(
        task.env.terrain.terrainProportions=[0,0,0,0,0,1,1,0,0]
        task.env.terrain.terrainType=heightfield
    )
}

# ----- stairs
argus_terrain_stairs_dof_12_const_vel_eval(){
    # bash run.sh argus_terrain_stairs_dof_12_const_vel_eval -p
    argus_terrain_stairs_dof_12_const_vel
    argus_terrain_eval_template
}

argus_terrain_stairs_dof_32_const_vel_eval(){
    # bash run.sh argus_terrain_stairs_dof_32_const_vel_eval -p
    argus_terrain_stairs_dof_32_const_vel
    argus_terrain_eval_template
}

argus_terrain_stairs_dof_20_const_vel_eval(){
    # bash run.sh argus_terrain_stairs_dof_20_const_vel_eval -p
    argus_terrain_stairs_dof_20_const_vel
    argus_terrain_eval_template
}

argus_terrain_stairs_dof_12_const_vel(){
    # bash run.sh argus_terrain_stairs_dof_12_const_vel -p
    argus_terrain_dof_12_const_vel
    argus_terrain_stair_template
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_stairs_dof_12_const_vel_20250620_133110/argus_terrain_stairs_dof_12_const_vel_newest.pt
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_stairs_dof_12_const_vel_20250623_142058/argus_terrain_stairs_dof_12_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_stairs_dof_12_const_vel.json
        --wandb_run_name=argus_terrain_stairs_dof_12_const_vel
    )
} 

argus_terrain_stairs_dof_32_const_vel(){
    # bash run.sh argus_terrain_stairs_dof_32_const_vel -p
    argus_terrain_dof_32_const_vel
    argus_terrain_stair_template
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_stairs_dof_32_const_vel_20250620_133105/argus_terrain_stairs_dof_32_const_vel_newest.pt
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_stairs_dof_32_const_vel_20250623_142102/argus_terrain_stairs_dof_32_const_vel_newest.pt
    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_stairs_dof_32_const_vel.json
        --wandb_run_name=argus_terrain_stairs_dof_32_const_vel
    )
} 

argus_terrain_stairs_dof_20_const_vel(){
    # bash run.sh argus_terrain_stairs_dof_20_const_vel -p
    argus_terrain_dof_20_const_vel
    argus_terrain_stair_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_stairs_dof_20_const_vel_20250623_142055/argus_terrain_stairs_dof_20_const_vel_newest.pt

    )
    BASE_ARGS+=(
        ++task.env.evaluate.filename=eval/argus_terrain_stairs_dof_20_const_vel.json
        --wandb_run_name=argus_terrain_stairs_dof_20_const_vel
    ) 
}

argus_terrain_stair_template(){
    PLAY_ARGS+=(
        task.env.terrain.numTerrains=2
        task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]
    )
    BASE_ARGS+=(
        task.env.terrain.terrainProportions=[0,0,0,1,1,0,0,0,0]
        # task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]

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

argus_terrain_dof_20_const_vel_shape_eval(){
    # bash run.sh argus_terrain_dof_20_const_vel_shape_eval -p
    argus_terrain_dof_20_const_vel_eval
    PLAY_ARGS+=(
        ++task.env.evaluate.shape_evaluate=True
        ++task.env.evaluate.filename=eval/argus_discrete_terrain_shape_eval.json
        --num_envs=512
        task.env.learn.episodeLength_s=5
    )

}

argus_terrain_dof_12_const_vel(){
    # bash run.sh argus_terrain_dof_12_const_vel -p
    argus_terrain
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_12_const_vel_20250614_193143/argus_terrain_dof_12_const_vel_newest.pt
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

argus_terrain_dof_32_const_vel(){
    # bash run.sh argus_terrain_dof_32_const_vel -p
    argus_terrain
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_32_const_vel_20250614_192007/argus_terrain_dof_32_const_vel_newest.pt
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



argus_terrain_dof_20_const_vel(){
    # bash run.sh argus_terrain_dof_20_const_vel -p
    argus_terrain
    PLAY_ARGS+=(

        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_20_const_vel_20250614_193145/argus_terrain_dof_20_const_vel.pt
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_20_const_vel_20250618_142117/argus_terrain_dof_20_const_vel.pt
        --checkpoint=../envs/runs/argus_debug/train/argus_terrain_dof_20_const_vel_20250624_222607/argus_terrain_dof_20_const_vel_newest.pt

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


argus_terrain_eval(){
    # bash run.sh argus_terrain_eval -p
    argus_terrain
    argus_terrain_eval_template
}

argus_terrain_eval_template(){
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        # ++task.env.evaluate.filename=eval/argus_discrete_terrain.json
        task.env.terrain.minInitMapLevel=0
        task.env.terrain.maxInitMapLevel=10
        task.env.terrain.numLevels=10
        task.env.terrain.numTerrains=80
        task.env.terrain.discrete.height=0.1
        # task.env.baseHeightOffset=0.1
        --num_envs=1024
        --headless=True
        task.env.learn.episodeLength_s=10
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        --total_timesteps=2048000
    )
}


argus_terrain(){
    # bash run.sh argus_terrain -pk
    # https://wandb.ai/grl_argus/argus_debug/runs/kvz2s81m

    argus_base
    PLAY_ARGS+=(
        task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
        # task.env.terrain.minInitMapLevel=0
        # task.env.terrain.maxInitMapLevel=10
        # task.env.terrain.numLevels=10
        task.env.terrain.discrete.height=0.1
        task.env.terrain.numTerrains=1
        task.env.terrain.numLevels=5

        task.env.dataPublisher.enable=False
        task.env.learn.episodeLength_s=15
        --num_envs=8
        --checkpoint=../envs/runs/argus_debug/train/discrete_terrain_dof20_20250614_174701/discrete_terrain_dof20_newest.pt

        # task.env.urdfAsset.file=urdf/argus/argus_dof20.urdf
        # task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        # task.env.randomize.erfi.enable=true

        # task.env.terrain.horizontalScale=0.05
    )
    BASE_ARGS+=(
        --wandb_run_name=discrete_terrain_dof20
        # task.env.terrain.terrainType=trimesh
        task.env.terrain.terrainType=heightfield
        # task.env.urdfAsset.file=urdf/argus/argus_dof20.urdf
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        --num_envs=4096
        --num_steps=16
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


argus_disable_leg_dof_12_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_12_const_vel_eval -p
    argus_disable_leg_dof_12_const_vel
    argus_disable_leg_eval_template
}

argus_disable_leg_dof_12_const_vel(){
    # bash run.sh argus_disable_leg_dof_12_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_disable_leg_dof_12_const_vel_20250614_192003/argus_disable_leg_dof_12_const_vel_newest.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_12_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_12_const_vel
    )
}

argus_disable_leg_dof_32_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_32_const_vel_eval -p
    argus_disable_leg_dof_32_const_vel
    argus_disable_leg_eval_template
}

argus_disable_leg_dof_32_const_vel(){
    # bash run.sh argus_disable_leg_dof_32_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_disable_leg_dof_32_const_vel_20250614_192002/argus_disable_leg_dof_32_const_vel_newest.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_32_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_32_const_vel
    )
} 


argus_disable_leg_dof_20_const_vel_eval(){
    # bash run.sh argus_disable_leg_dof_20_const_vel_eval -p
    argus_disable_leg_dof_20_const_vel
    argus_disable_leg_eval_template
}


argus_disable_leg_dof_20_const_vel(){
    # bash run.sh argus_disable_leg_dof_20_const_vel -p
    argus_disable_leg
    PLAY_ARGS+=(
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_disable_leg_dof_20_const_vel_debug_20250601_165132/argus_disable_leg_dof_20_const_vel_debug_newest.pt
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/server/argus_debug/train/argus_disable_leg_dof_20_const_vel_20250520_234255/argus_disable_leg_dof_20_const_vel_newest.pt
        --checkpoint=../envs/runs/server/argus_debug/train/argus_disable_leg_dof_20_const_vel_20250614_192001/argus_disable_leg_dof_20_const_vel_newest.pt
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_disable_leg_dof_20_const_vel.json
        --wandb_run_name=argus_disable_leg_dof_20_const_vel
    ) 
}

argus_disable_leg_eval(){
    # bash run.sh argus_base_eval -p
    argus_disable_leg
    argus_disable_leg_eval_template
}


argus_disable_leg(){
    # https://wandb.ai/grl_argus/argus_debug/runs/b68r4a6l  
    argus_base
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/baseline_train__20250413_014947/baseline.pt
        # https://wandb.ai/grl_argus/argus_debug/runs/nvxsdta0
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/train/disable_leg_20250418_202437/disable_leg.pt
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

argus_throw_eval(){
    # bash run.sh argus_throw_eval -p
    argus_push
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.filename=eval_throw.json
        --num_envs=8192
        --headless=True

        task.env.learn.episodeLength_s=5
        task.env.baseHeightOffset=0.5
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0,0]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomize.push.velMin=[-5,-5,0,0,0,0]
        task.env.randomize.push.velMax=[5,5,0,0,0,0]

        # --num_envs=2
        # # task.env.baseHeightOffset=1.0
        # # task.env.randomize.push.velMin=[1,1,1,-5,-5,-5]
        # # task.env.randomize.push.velMax=[5,5,1,5,5,5]
        # task.env.randomize.push.velMin=[-2,-2,-2,-2,-2,-2]
        # task.env.randomize.push.velMax=[2,2,2,2,2,2]
        # task.env.urdfAsset.file=urdf/argus/argus_dof20.urdf
        # task.env.envSpacing=0
        # task.env.learn.episodeLength_s=4
        # task.env.randomize.base_init_pos.enable=False
        # task.env.assetDofProperties.damping=100
        # task.env.viewer.follower_offset=[-2,-2,2]
        # task.env.control.stiffness=4000
        # task.env.control.damping=60
        # task.env.control.actionScale=0.105
        # --checkpoint=../assets/checkpoint/push_newest.pt


    )
}

argus_push_eval_shape(){
    # bash run.sh argus_push_eval_shape -p
    argus_push_eval
    PLAY_ARGS+=(
        ++task.env.evaluate.shape_evaluate=True
        ++task.env.evaluate.filename=eval/argus_stabilize_shape_eval.json

    )
}

argus_push_dof_20_const_vel_eval(){
    # bash run.sh argus_push_dof_20_const_vel_eval -p
    argus_push_dof_20
    argus_push_eval_template
}

argus_push_dof_32_const_vel_eval(){
    # bash run.sh argus_push_dof_32_const_vel_eval -p
    argus_push_dof_32
    argus_push_eval_template
}

argus_push_dof_12_const_vel_eval(){
    # bash run.sh argus_push_dof_12_const_vel_eval -p
    argus_push_dof_12
    argus_push_eval_template
}

argus_push_eval_template(){
    PLAY_ARGS+=(
        --headless=True
        ++task.env.evaluate.enable=True
        --num_envs=4096
        task.env.learn.episodeLength_s=5
        --total_timesteps=1024000
        task.env.baseHeightOffset=0
        task.env.dataPublisher.enable=false
        task.env.randomCommandVelocityRanges.linear_x=[0,0]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomize.push.velMin=[-5,-5,0,0,0,0]
        task.env.randomize.push.velMax=[5,5,0,0,0,0]
        task.env.randomize.base_init_orientation.enable=False
    )
}

argus_push_dof_12(){
    # bash run.sh argus_push_dof_12 -p
    argus_push
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_push_dof_12_20250731_153930/argus_push_dof_12.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_push_dof_12.json
        --wandb_run_name=argus_push_dof_12
    )
}

argus_push_dof_32(){
    # bash run.sh argus_push_dof_32 -p
    argus_push
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_push_dof_32_20250731_153932/argus_push_dof_32.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_push_dof_32.json
        --wandb_run_name=argus_push_dof_32
    )
}

argus_push_dof_20(){
    # bash run.sh argus_push_dof_20 -p
    argus_push
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_push_dof_20_20250731_153934/argus_push_dof_20.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        ++task.env.evaluate.filename=eval/argus_push_dof_20.json
        --wandb_run_name=argus_push_dof_20
    )
}



argus_push_debug_3(){
    # bash run.sh argus_push_debug_3 -p
    argus_push
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/train/argus_push_debug_3_20250607_214250/argus_push_debug_3.pt
        --checkpoint=../assets/checkpoint/argus_push_debug_3_20250607_214250/argus_push_debug_3.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_push_debug_3
        task.env.control.decimation=8
        task.env.control.integral=200
        "task.env.observationNames=[angularVelocity,commands_xy,dofPosition,dofVelocity,actions,base_rotation_matrix]"
    )
}

argus_push_eval(){
    # bash run.sh argus_push_eval -p
    argus_push
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.filename=eval_push.json
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

argus_push(){
    # bash run.sh argus_push -p
    # https://wandb.ai/grl_argus/argus_debug/runs/bkoy9hgq
    argus_base
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/push_20250520_001612/push_newest.pt
        # --checkpoint=None
        # task.env.randomize.push.velMin=[-5,-5,-5,-5,-5,-5]
        # task.env.randomize.push.velMax=[5,5,5,5,5,5]

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


argus_debug_step(){
    argus_base
    PLAY_ARGS+=(
        --checkpoint=
    )
    TRAIN_ARGS+=(
        --num_envs=512
        --num_steps=32
    )
    BASE_ARGS+=(
        --wandb_run_name=debug_num_envs512_num_steps32
    )
}


argus_climbup_smooth_wall_vision_1x1_robstride03(){
    # bash run.sh argus_climbup_smooth_wall_vision_1x1_robstride03 -p
    argus_climbup_smooth_wall
    argus_climbup_vision_template
    argus_robstride03_template
    PLAY_ARGS+=(
        # --checkpoint=None
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_smooth_wall_vision_1x1_robstride03_20250428_142857/climbup_smooth_wall_vision_1x1_robstride03_newest.pt
        # task.env.dataPublisher.enable=False
        --num_envs=2

    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_smooth_wall_vision_1x1_robstride03
        task.env.tof.resolution=[1,1]
    )
}

argus_climbup_smooth_no_vision_robstride03(){
    # bash run.sh argus_climbup_smooth_no_vision_robstride03 -p
    argus_climbup_robstride03
    PLAY_ARGS+=(
        # --checkpoint=None
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_robstride03_smooth_wall_no_vision_20250427_195858/climbup_robstride03_smooth_wall_no_vision_iter_0091500_return_13.1.pt
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_robstride03_smooth_wall_no_vision_20250427_195858/climbup_robstride03_smooth_wall_no_vision_iter_0044900_return_12.6.pt
        --num_envs=1
    )
    TRAIN_ARGS+=(

    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_robstride03_smooth_wall_no_vision
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls.npz'
    )
}

 

argus_climbup_vision_1x1_robstride03(){ # not yet working # TODO
    # bash run.sh argus_climbup_vision_1x1_robstride03 -p
    argus_climbup
    argus_climbup_vision_template
    argus_robstride03_template
    PLAY_ARGS+=(
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_vision_1x1_robstride03_20250428_103542/climbup_vision_1x1_robstride03_iter_0001900_return_9.7.pt
    )
    TRAIN_ARGS+=(

    )
    BASE_ARGS+=(
      --wandb_run_name=climbup_vision_1x1_robstride03
        task.env.tof.resolution=[1,1]
    )
}

argus_climbup_vision_robstride03(){ # not yet working
    # bash run.sh argus_climbup_vision_robstride03 -p
    argus_climbup
    argus_climbup_vision_template
    argus_robstride03_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_robstride03_vision_20250427_195846/climbup_robstride03_vision_iter_0024000_return_10.5.pt
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_robstride03_vision_20250427_195846/climbup_robstride03_vision_iter_0025700_return_10.9.pt

    )
    TRAIN_ARGS+=(

    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_robstride03_vision
    )
}

argus_climbup_robstride03(){
    # bash run.sh argus_climbup_robstride03 -p
    argus_climbup
    argus_robstride03_template
    PLAY_ARGS+=(
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_robstride03_20250428_144024/climbup_robstride03_newest.pt

    )
    TRAIN_ARGS+=(

    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_robstride03
    )
}

argus_robstride03_template(){
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum_robstride03.urdf
        task.env.control.limit=1375
        task.env.control.stiffness=2500
        task.env.control.damping=250
        task.env.control.actionScale=0.5
        task.env.assetDofProperties.damping=3
        task.env.assetDofProperties.effort=1375
        task.env.motor_type=robstride_03
        task.env.gravity_support.end_acc=6.1 # # mars gravity=3.71 m/s^2, external acc needed = 9.81 - 3.71 = 6.1 m/s^2
    )
}


argus_climbup_vision_debug_obs_1x1_hist_5(){
# bash run.sh argus_climbup_vision_debug_obs_1x1_hist_5 -p
    argus_climbup
    argus_climbup_vision_template
    PLAY_ARGS+=(
        --checkpoint=None
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_vision_debug_obs
        task.env.tof.history_length=5
        task.env.tof.resolution=[1,1]
        task.env.observationNames=[angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,base_rotation_matrix,ray_point_cloud]
    )
}

argus_climbup_vision_debug_obs_1x1_hist_3(){
# bash run.sh argus_climbup_vision_debug_obs_1x1_hist_3 -p
    argus_climbup
    argus_climbup_vision_template
    PLAY_ARGS+=(
        --checkpoint=None
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_vision_debug_obs
        task.env.tof.history_length=3
        task.env.tof.resolution=[1,1]
        task.env.observationNames=[angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,base_rotation_matrix,ray_point_cloud]
    )
}


argus_climbup_vision_debug_obs_5x5(){
# bash run.sh argus_climbup_vision_debug_obs_5x5 -p
    argus_climbup
    argus_climbup_vision_template
    PLAY_ARGS+=(
        --checkpoint=None
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_vision_debug_obs
        task.env.tof.history_length=1
        task.env.tof.resolution=[5,5]
        task.env.observationNames=[angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,base_rotation_matrix,ray_point_cloud]
    )
}

argus_climbup_vision_debug_obs_1x1(){ # ✅
# bash run.sh argus_climbup_vision_debug_obs_1x1 -p 
    argus_climbup
    argus_climbup_vision_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/climbup_vision_debug_obs_1x1_20250428_140119/climbup_vision_debug_obs_1x1_iter_0004000_return_9.4.pt
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/server/argus_climbup_diymap/train/climbup_vision_debug_obs_1x1_20250520_000858/climbup_vision_debug_obs_1x1_newest.pt
    
    # task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
    # task.env.gravity_support.end_acc=9
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=climbup_vision_debug_obs_1x1
        task.env.tof.resolution=[1,1]
    )
}

argus_climbup_vision(){ # TODO
# bash run.sh argus_climbup_vision -p
    argus_climbup
    argus_climbup_vision_template
    PLAY_ARGS+=(
        # --checkpoint=None
        # ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls.npz'
        task.env.dataPublisher.enable=True
        --num_envs=8
        # --checkpoint=None
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/rough_wall_vision_20250422_223120/rough_wall_vision_iter_0048200_return_12.4.pt
    )
    TRAIN_ARGS+=(
        --num_envs=32
        # --num_steps=8
        # --headless=False
        task.env.visualize_ray_point_cloud=True

    )
    BASE_ARGS+=(
        --wandb_run_name=rough_wall_vision
    )
}


argus_climbup_smooth_wall_vision_debug(){
# bash run.sh argus_climbup_smooth_wall_vision_debug -p
    argus_climbup_smooth_wall_vision
    PLAY_ARGS+=(
        --num_envs=1
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/checkpoint/climbup/smooth_wall_vision_warmstart_150N_iter_0002100_return_8.3.pt
        task.env.dataPublisher.enable=True

    )
    TRAIN_ARGS+=(
        --checkpoint=envs/checkpoint/climbup/smooth_wall_vision_iter_0035300_return_13.4.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=smooth_wall_vision_warmstart_150N
    )
}

argus_climbup_smooth_wall_vision(){
# bash run.sh argus_climbup_smooth_wall_vision -p
    argus_climbup_smooth_wall
    argus_climbup_vision_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/smooth_wall_vision_20250422_162308/smooth_wall_vision.pt
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/smooth_wall_vision_20250422_160903/smooth_wall_vision_iter_0008800_return_11.2.pt
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/smooth_wall_vision_20250422_160903/smooth_wall_vision_iter_0087100_return_14.4.pt
        --num_envs=1
    )
    TRAIN_ARGS+=(
        # --num_envs=32
        # --num_steps=8
        # --headless=False
    )
    BASE_ARGS+=(
        --wandb_run_name=smooth_wall_vision
    )
}

argus_climbup_vision_template(){
        BASE_ARGS+=(
        task.env.tof.history_length=1
        task.env.observationNames=[angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,base_rotation_matrix,ray_point_cloud]
    )
}

argus_climbup_rough_wall_real(){
    # bash run.sh argus_climbup_rough_wall_real -p
    argus_climbup
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_climbup_diymap/train/baseline_no_vision_smooth_wall_20250708_003411/baseline_no_vision_smooth_wall_newest.pt
        --num_envs=1
    )
    TRAIN_ARGS+=(
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_rough_wall_real
    )
}

argus_climbup_smooth_wall_real_00_05(){
    # bash run.sh argus_climbup_smooth_wall_real_00_05 -p
    argus_climbup_smooth_wall_real_00
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_05
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.6

    )
}

argus_climbup_smooth_wall_real_00_04_04(){ #totest itsok
    # bash run.sh argus_climbup_smooth_wall_real_00_04_04 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
    )
    TRAIN_ARGS+=(
        --num_envs=5120
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_04
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.35

    )
}

argus_climbup_smooth_wall_real_00_04_03(){ #totest itsok
    # bash run.sh argus_climbup_smooth_wall_real_00_04_03 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
    )
    TRAIN_ARGS+=(
        --num_envs=5120
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_03
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.1

    )
}




argus_climbup_smooth_wall_real_00_04_01(){
    # bash run.sh argus_climbup_smooth_wall_real_00_04_01 -p
    argus_climbup_smooth_wall_real_00
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_01
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.6
        task.env.learn.reward.dof_limit.scale=-8
    )
}

argus_climbup_smooth_wall_real_00_04_00_00_01(){ 
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00_00_01 -p
    argus_climbup_smooth_wall_real_00_04_00_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_00_00_01
        # task.env.desired_vel=0.15

        # task.env.learn.reward.lin_vel.scale=4
        # task.env.learn.reward.lin_vel.exp_scale=-32
        task.env.learn.reward.lin_vel.normalize_by=[1,1,1]
        task.env.learn.reward.base_height.exp_scale=-1
        task.env.base_height_target=1.5
    )
}

argus_climbup_smooth_wall_real_00_04_00_00_00(){ 
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00_00_00 -p
    argus_climbup_smooth_wall_real_00_04_00_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_00_00_00
        task.env.desired_vel=0.15

        task.env.learn.reward.lin_vel.scale=4
        task.env.learn.reward.lin_vel.exp_scale=-32
        task.env.learn.reward.lin_vel.normalize_by=[1,1,1]
        task.env.learn.reward.base_height.exp_scale=-1
        task.env.base_height_target=1.5
    )
}

argus_climbup_smooth_wall_real_00_04_00_01(){ 
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00_01 -p
    argus_climbup_smooth_wall_real_00_04_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_00_01
        "task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position]"
        task.env.desired_vel=0.15
        task.env.learn.reward.lin_vel.scale=4
        task.env.learn.reward.lin_vel.exp_scale=-32
        task.env.learn.reward.base_height.exp_scale=-1
        task.env.base_height_target=1.5
    )
}

argus_climbup_smooth_wall_real_00_04_00_00(){ 
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00_00 -p
    argus_climbup_smooth_wall_real_00_04_00
    PLAY_ARGS+=(
        --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_climbup_smooth_wall_real_00_04_00_00.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_00_00
        "task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,dofPosition,dofVelocity,actions,contact,base_rotation_matrix,robot_root_position]"

    )
}


argus_climbup_smooth_wall_real_00_04_02(){ #totest itsok
    # bash run.sh argus_climbup_smooth_wall_real_00_04_02 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_climbup_smooth_wall_real_00_04_02.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_02
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.1

    )
}
argus_climbup_smooth_wall_real_00_04_00_eval(){
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00_eval -p
    argus_climbup_smooth_wall_real_00_04_00
    PLAY_ARGS+=(
        --num_envs=4096
        --headless=True
        ++task.env.evaluate.enable=True
        task.env.learn.episodeLength_s=5
        task.env.randomize.base_init_pos.range=[[-5000,5000],[0,0],[0,0]]
        task.env.randomize.base_init_pos.enable=true
        task.env.dataPublisher.enable=False
        task.env.gravity_support.end_acc=7.19
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls_1.0m.npz'
    )
}

argus_climbup_smooth_wall_real_00_04_00(){ #GOOD! Evaluated in real.
    # bash run.sh argus_climbup_smooth_wall_real_00_04_00 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_04_00_20250722_220029/argus_climbup_smooth_wall_real_00_04_00.pt
        # --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_climbup_smooth_wall_real_00_04_00.pt

    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04_00
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
        task.env.gravity_support.start_acc=7.19
    )
}

argus_climbup_smooth_wall_real_00_04(){
    # bash run.sh argus_climbup_smooth_wall_real_00_04 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_climbup_smooth_wall_real_00_04_newest.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_04
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.control.actionScale=0.105
    )
}

argus_climbup_smooth_wall_real_00_03_02(){
    # bash run.sh argus_climbup_smooth_wall_real_00_03_02 -p
    argus_climbup_smooth_wall_real_00_03

    PLAY_ARGS+=(
        --checkpoint=/home/generalroboticslab/repo/Argus_Boxi/vrobot_env/envs/runs/argus_climbup_smooth_wall_real_00_03_02.pt

    )

    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_03_02
        task.env.learn.reward.dof_limit.scale=-5
        task.env.gravity_support.start_acc=7.1


    )
}

argus_climbup_smooth_wall_real_00_03_01(){ #totest - slower but swing in the air
    # bash run.sh argus_climbup_smooth_wall_real_00_03_01 -p
    argus_climbup_smooth_wall_real_00_03
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_03_01_20250722_215309/argus_climbup_smooth_wall_real_00_03_01.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_03_01
        task.env.learn.reward.dof_limit.scale=-8

    )
}

argus_climbup_smooth_wall_real_00_03_00(){ #totest - weaker
    # bash run.sh argus_climbup_smooth_wall_real_00_03_00 -p
    argus_climbup_smooth_wall_real_00_03
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_03_00_20250722_213632/argus_climbup_smooth_wall_real_00_03_00.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_03_00
        task.env.learn.reward.dof_limit.scale=-5

    )
}


argus_climbup_smooth_wall_real_00_03(){ #totest
    # bash run.sh argus_climbup_smooth_wall_real_00_03 -p
    argus_climbup_smooth_wall_real_00
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/argus_climbup_smooth_wall_real_00_03_20250722_205626/argus_climbup_smooth_wall_real_00_03.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_03
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.control.dynamic_action_bound.enable=false
        task.env.gravity_support.start_acc=7.6
        task.env.control.actionScale=0.15
    )
}

argus_climbup_smooth_wall_real_00_02(){
    # bash run.sh argus_climbup_smooth_wall_real_00_02 -p
    argus_climbup_smooth_wall_real_00
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_02
        task.env.learn.reward.dof_force_target.scale=0.1

    )
}

argus_climbup_smooth_wall_real_00_01(){
    # bash run.sh argus_climbup_smooth_wall_real_00_01 -p
    argus_climbup_smooth_wall_real_00
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_01
        task.env.gravity_support.start_acc=7.6
    )
}

argus_climbup_smooth_wall_real_00_00(){
    # bash run.sh argus_climbup_smooth_wall_real_00_00 -p
    argus_climbup_smooth_wall_real_00
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00_00
        task.env.control.stiffness=1800
    )
}


argus_climbup_smooth_wall_real_00(){
    # bash run.sh argus_climbup_smooth_wall_real_00 -p
    argus_climbup_smooth_wall
    PLAY_ARGS+=(
        --checkpoint=
        --num_envs=1
    )
    TRAIN_ARGS+=(
        --num_envs=4096

    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real_00
        task.env.num_stacked_obs_frame=1
        task.env.randomize.base_init_orientation.enable=False # added 
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls_1.0m.npz'
        task.env.control.stiffness=2000
        task.env.gravity_support.start_acc=8.6
        task.env.randomize.friction.enable=True
        task.env.randomize.friction.range=[0.8,1.2]
        task.env.control.decimation=10
        task.env.baseInitState.rot=[0.2500,0.2500,-0.0670,0.9330]
        task.env.control.dynamic_action_bound.enable=true
        task.env.control.dynamic_action_bound.alpha=0.95
        task.env.control.dynamic_action_bound.bound=0.15
        task.env.learn.addNoise=True # noisy observation
        task.env.max_observation_delay_steps=1 # 1 step delay
        task.env.randomize.erfi.enable=true
        task.env.randomize.default_dof_pos.enable=true
    )
}

argus_climbup_smooth_wall_real(){
    # bash run.sh argus_climbup_smooth_wall_real -p
    argus_climbup_smooth_wall
    PLAY_ARGS+=(

         # smaller random range
        --checkpoint=
        
        --num_envs=1

        # task.env.randomize.friction.enable=true
        # task.env.randomize.control_pd.enable=true
        # task.env.randomize.baseMass.enable=true
        # task.env.randomize.baseInertiaOrigin.enable=true
        # task.env.randomize.push.enable=false
        # task.env.randomize.initDofPos.enable=true
        # task.env.randomize.initDofVel.enable=true
        # # task.env.randomize.erfi.enable=false
        # task.env.randomize.dof_strength.enable=true
        # task.env.randomize.default_dof_pos.enable=false
        # task.env.randomize.link_mass.enable=true
        # task.env.randomize.link_inertia.enable=true
        # task.env.randomize.body_force.enable=false
        # task.env.randomize.action_delay.enable=true
        # # task.env.learn.addNoise=false
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_climbup_smooth_wall_real
        task.env.control.dynamic_action_bound.enable=true
        task.env.control.dynamic_action_bound.alpha=0.95
        task.env.control.dynamic_action_bound.bound=0.1
    )
}


argus_climbup_smooth_wall(){
    # bash run.sh argus_climbup_smooth_wall -p
    argus_climbup
    # argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/baseline_no_vision_smooth_wall_20250422_123129/baseline_no_vision_smooth_wall_iter_0031600_return_11.6.pt
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls.npz'
        # ++task.env.terrain.file_path='../assets/urdf/climbup/tmp_debug.obj'
        task.env.dataPublisher.enable=True
        # --checkpoint=../envs/runs/argus_climbup_diymap/train/baseline_no_vision_smooth_wall_20250707_184754/baseline_no_vision_smooth_wall_newest.pt
        --checkpoint=../envs/runs/argus_climbup_diymap/train/baseline_no_vision_smooth_wall_20250708_003337/baseline_no_vision_smooth_wall_newest.pt

        --num_envs=2

    )
    TRAIN_ARGS+=(
        # --num_envs=4096
        # --num_steps=8

        # --headless=False
        # --track=False
        # --num_envs=16
        # # task.env.terrain.terrainType=plane
        # task.env.randomize.base_init_pos.range=[[-5,5],[0,0],[0,3]]

    )
    BASE_ARGS+=(
        --wandb_run_name=baseline_no_vision_smooth_wall
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_smooth_walls.npz'
    )
}

argus_climbup(){
# bash run.sh argus_climbup -p
    ENTRY_POINT=ppo_isaacgym.py
    PLAY_ARGS=(
        # https://wandb.ai/grl_argus/argus_climbup_diymap/runs/ix94x5ra
        # --checkpoint=None
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_climbup_diymap/train/baseline_no_vision_20250514_234346/baseline_no_vision_newest.pt
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/server/argus_climbup_diymap/train/baseline_no_vision_20250521_143920/baseline_no_vision_newest.pt
        
        --train_mode=play
        --num_envs=8
        --headless=False
        --track=False

        # ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_rough_walls.obj'
        # ++task.env.terrain.file_path='../assets/urdf/climbup/tmp_debug.obj'
        task.env.randomize.base_init_pos.range=[[-3,3],[0,0],[0,1]]

        # task.env.learn.episodeLength_s=5

        task.env.gravity_support.curriculum=False # use end_acc

        # ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_rough_walls_200m.obj'
        # ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_rough_walls_200m.npz'

        task.env.urdfAsset.file=urdf/argus/argus_dof20.urdf
        ++task.env.viewer.follower_offset=[-2,0,0]
        --num_envs=1
    )
    
    TRAIN_ARGS=(
        --train_mode=train

        # --headless=False
        # --track=False

        --headless=True
        --track=True
        --num_envs=4096
        --num_steps=8
    )
    BASE_ARGS=(
        --task_name=argus_climbup_diymap
        --wandb_run_name=baseline_no_vision
        --wandb_entity=grl_argus
        --agent_name=baseline

        task.env.randomize.base_init_pos.range=[[-5000,5000],[0,0],[0,3]]
        task.env.randomize.base_init_pos.enable=True
        ++task.env.terrain.file_path='../assets/urdf/climbup/parallel_rough_walls.obj'
    )
}


argus_carry_object_debug_2(){ # GOOD IN REAL AS WELL. *VERIFIED* 👍
    # bash run.sh argus_carry_object_debug_2 -pk
    argus_carry_object
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/train/argus_carry_object_debug_2_20250603_115533/argus_carry_object_debug_2.pt
        # --checkpoint=../assets/checkpoint/argus_carry_object_debug_2_20250603_115533/argus_carry_object_debug_2.pt
        # with integral=500
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_carry_object_debug_2_20250604_003444/argus_carry_object_debug_2.pt
        --checkpoint=../assets/checkpoint/argus_carry_object_debug_2.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_carry_object_debug_2
        task.env.control.actionScale=0.1
        task.env.control.stiffness=2000
        task.env.control.damping=100
        task.env.randomize.baseInertiaOrigin.enable=True
        task.env.randomize.baseInertiaOrigin.range=[[-0.05,0.05],[-0.05,0.05],[-0.05,0.05]]
        task.env.randomize.baseMass.range=[0,5]
    )
}

argus_carry_object_dof_12_const_vel_eval(){
    argus_carry_object_eval_template
    argus_carry_object_dof_12_const_vel
}

argus_carry_object_dof_20_const_vel_eval(){
    argus_carry_object_dof_20_const_vel
    argus_carry_object_eval_template
}

argus_carry_object_dof_32_const_vel_eval(){
    argus_carry_object_eval_template
    argus_carry_object_dof_32_const_vel
}


argus_carry_object_dof_12_const_vel(){
    # bash run.sh argus_carry_object_dof_12_const_vel -p
    argus_carry_object
    argus_carry_object_eval_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_carry_object_dof_12_const_vel_20250615_014236/argus_carry_object_dof_12_const_vel.pt
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_12_const_vel_20250614_193143/argus_terrain_dof_12_const_vel_newest.pt

        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf

    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_12_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof12_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_12_const_vel
    )
}

argus_carry_object_dof_20_const_vel(){
    # bash run.sh argus_carry_object_dof_20_const_vel -p
    argus_carry_object
    argus_carry_object_eval_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_carry_object_dof_20_const_vel_20250615_014223/argus_carry_object_dof_20_const_vel.pt
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_20_const_vel.json
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_20_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_20_const_vel
    )
}

argus_carry_object_dof_32_const_vel(){
    # bash run.sh argus_carry_object_dof_32_const_vel -p
    argus_carry_object
    argus_carry_object_eval_template
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/server/argus_debug/train/argus_carry_object_dof_32_const_vel_20250615_014216/argus_carry_object_dof_32_const_vel.pt
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        # --checkpoint=../envs/runs/server/argus_debug/train/argus_terrain_dof_32_const_vel_20250614_192007/argus_terrain_dof_32_const_vel_newest.pt

    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        ++task.env.evaluate.filename=eval/argus_carry_object_dof_32_const_vel.json
        task.env.urdfAsset.file=urdf/argus/argus_dof32_minimum.urdf
        --wandb_run_name=argus_carry_object_dof_32_const_vel
    )
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
        task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf
    )
}

argus_carry_object_eval(){
    # bash run.sh argus_carry_object_eval -p
    argus_carry_object
    argus_carry_object_eval_template
}


argus_carry_object(){
    # bash run.sh argus_carry_object -p
    argus_base
    PLAY_ARGS+=(
        --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/argus_debug/train/carry_object_20250419_222447/carry_object.pt
        --num_envs=1
        
        # task.env.urdfAsset.file=urdf/argus/argus_dof20_load.urdf
        # task.env.randomize.baseMass.enable=False
        # task.env.randomize.baseInertiaOrigin.enable=False
        # task.env.urdfAsset.AssetOptions.collapse_fixed_joints=False
        # ++task.env.urdfAsset.footName=link_
        
        # # visualization only
        # task.env.terrain.discrete.height=0.1
        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.numTerrains=5
        # ++task.env.renderFPS=60
        # task.env.learn.episodeLength_s=9999

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


argus_base_dynamic_matching(){
    # bash run.sh argus_base_dynamic_matching -pk
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --num_envs=1
        --checkpoint=None
        # task.env.urdfAsset.file=urdf/argus/argus_dof20.urdf
        # task.env.urdfAsset.AssetOptions.fix_base_link=True
        task.env.baseHeightOffset=0
        task.env.learn.episodeLength_s=9999
        task.env.randomize.base_init_orientation.enable=False
        task.env.randomize.base_init_pos.enable=False
        ++task.env.renderFPS=50

        task.env.dataReceiver.enable=True # for receiving udp data
        task.env.dataPublisher.enable=True # for sending udp data

        # task.sim.dt=0.001
        # task.env.control.decimation=20

        # task.env.randomize.baseMass.enable=True
        # task.env.randomize.baseMass.range=[-10,-10]

    )
    BASE_ARGS+=(
        task.env.control.actionScale=0.1
        # task.env.control.stiffness=2000
        task.env.control.stiffness=2000
        task.env.control.damping=100
    )
}

argus_sim2real_dynamics_setup_template(){
    PLAY_ARGS+=(
        task.env.randomize.erfi.enable=false
        task.env.randomize.default_dof_pos.enable=false
        task.env.randomize.action_delay.enable=true
    )
    BASE_ARGS+=(
        task.env.assetDofProperties.friction=0.0 # can be as high as 0.2
        task.env.learn.addNoise=True # noisy observation
        task.env.max_observation_delay_steps=1 # 1 step delay
        task.env.randomize.erfi.enable=true
        task.env.randomize.default_dof_pos.enable=true

        task.env.control.actionScale=0.1
        task.env.control.integral=200
        task.env.control.stiffness=2000
        task.env.control.damping=100
        task.env.control.decimation=8

    )
}



argus_base_sim2real_replay(){
    # bash run.sh argus_base_sim2real_replay -pk
    argus_base
    PLAY_ARGS+=(

        # task.env.control.limit=1
        # ++task.env.assetDofProperties.effort=1

        --num_envs=1
        task.env.baseHeightOffset=0
        task.env.learn.episodeLength_s=9999
        task.env.randomize.base_init_orientation.enable=False
        task.env.randomize.base_init_pos.enable=False
        ++task.env.renderFPS=50
        task.env.sim2realDataPublisher.enable=true
    )
}


argus_disable_leg_debug_2(){ # OK IN SIM, *👍 **verified**
    # bash run.sh argus_disable_leg_debug_2 -p
    argus_disable_leg
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        --checkpoint=../assets/checkpoint/disable_leg_debug_2.pt
    )
    BASE_ARGS+=(
        --wandb_run_name=disable_leg_debug_2
        task.env.control.decimation=8
        # task.env.control.integral=500
        task.env.control.actionScale=0.1
        task.env.control.stiffness=2000
        task.env.control.damping=100
    )
}



argus_base_debug_23(){ # GOOD INREAL as well 👍 **verified**
    # bash run.sh argus_base_debug_23 -pk
    argus_base
    argus_sim2real_dynamics_setup_template
    PLAY_ARGS+=(
        # --checkpoint=/home/grl/repo/vrobot_env_exp/envs/runs/server/argus_debug/train/argus_base_debug_23_20250604_003437/argus_base_debug_23.pt
        --checkpoint=../assets/checkpoint/argus_base_debug_23_20250604_003437/argus_base_debug_23.pt
        task.env.dataReceiver.enable=True
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_base_debug_23
        task.env.control.decimation=8
        task.env.control.actionScale=0.1
        task.env.control.stiffness=2000
        task.env.control.damping=100
        task.env.control.integral=200
        "task.env.observationNames=[angularVelocity,commands_xy,dofPosition,dofVelocity,actions,base_rotation_matrix]"
    )
}

argus_sim2real_replay_template(){
    PLAY_ARGS+=(
        --num_envs=1
        task.env.baseHeightOffset=0
        task.env.learn.episodeLength_s=9999
        task.env.randomize.base_init_orientation.enable=False
        task.env.randomize.base_init_pos.enable=False
        ++task.env.renderFPS=50
        task.env.sim2realDataPublisher.enable=true
    )
}


argus_base_eval_debug(){
    argus_base
    PLAY_ARGS+=(
        ++task.env.evaluate.enable=True
        ++task.env.evaluate.filename=eval/argus_base_debug.json
        --num_envs=16
        --headless=True
        task.env.learn.episodeLength_s=3
        task.env.dataPublisher.enable=false
        "task.env.urdfAsset.file=urdf/argus/argus_dof20_minimum.urdf"

    )
}


argus_carry_object_COMtrajectory_eval(){
    # bash run.sh argus_carry_object_COMtrajectory_eval -p
    argus_carry_object_dof_20_const_vel
    PLAY_ARGS+=(
        --num_envs=512
        # --headless=False
        task.env.learn.episodeLength_s=20
        task.env.dataPublisher.enable=false
        ++task.env.evaluate.COM_evaluation=True

        ++task.env.evaluate.filename=eval/argus_carry_object_COMtrajectory.json
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomize.base_init_orientation.enable=False # added
        task.env.randomize.base_init_pos.enable=False # added
        task.env.randomize.baseMass.range=[4.5,4.5]
        task.env.randomize.baseInertiaOrigin.range=[[0.05,0.05],[-0.0,0.0],[-0.0,0.0]]

        task.env.randomize.friction.enable=False
        # task.env.randomize.baseMass.enable=False
        # task.env.randomize.baseInertiaOrigin.enable=False
        task.env.randomize.push.enable=False
        task.env.randomize.initDofPos.enable=False
        task.env.randomize.initDofVel.enable=False
        task.env.randomize.dof_strength.enable=False
        task.env.randomize.control_pd.enable=False
        task.env.randomize.link_mass.enable=False
        task.env.randomize.link_inertia.enable=False
        task.env.randomize.body_force.enable=False
        task.env.randomize.action_delay.enable=False
        task.env.randomize.orientation_delay.enable=False


    )
}

argus_base_COMtrajectory_eval(){
    # bash run.sh argus_base_COMtrajectory_eval -p
    argus_base_eval
    PLAY_ARGS+=(
        --num_envs=512
        # --headless=False
        task.env.learn.episodeLength_s=20
        task.env.dataPublisher.enable=false
        ++task.env.evaluate.COM_evaluation=True

        ++task.env.evaluate.filename=eval/argus_base_COMtrajectory.json
        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomize.base_init_orientation.enable=False # added
        task.env.randomize.base_init_pos.enable=False # added

        task.env.randomize.friction.enable=False
        task.env.randomize.baseMass.enable=False
        task.env.randomize.baseInertiaOrigin.enable=False
        task.env.randomize.push.enable=False
        task.env.randomize.initDofPos.enable=False
        task.env.randomize.initDofVel.enable=False
        task.env.randomize.dof_strength.enable=False
        task.env.randomize.control_pd.enable=False
        task.env.randomize.link_mass.enable=False
        task.env.randomize.link_inertia.enable=False
        task.env.randomize.body_force.enable=False
        task.env.randomize.action_delay.enable=False
        task.env.randomize.orientation_delay.enable=False


    )
}

argus_base_shape_eval(){
    # bash run.sh argus_base_shape_eval -p
    argus_base_eval
    PLAY_ARGS+=(
        ++task.env.evaluate.shape_evaluate=True
        ++task.env.evaluate.filename=eval/argus_base_shape.json
        task.env.randomCommandVelocityRanges.linear_x=[0.6,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
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
        --task_name=argus_debug
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



argus_sim_rand_joint(){
    # bash run.sh argus_sim_rand_joint -p
    argus_base
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512/sim_rand_joint_0000_argus_dof20_minimum.urdf/sim_rand_joint_0000_argus_dof20_minimum.urdf_newest.pt
        # task.env.learn.episodeLength_s=20
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_sim_rand_joint

        task.env.randomCommandVelocityRanges.linear_x=[0.8,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0000_argus_dof20_minimum.urdf
    )
}

argus_sim_rand_joint_0511(){
    # bash run.sh argus_sim_rand_joint_0511 -p
    argus_sim_rand_joint
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512/sim_rand_joint_0511_argus_dof20_minimum.urdf/sim_rand_joint_0511_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0511_argus_dof20_minimum.urdf
    )
}

argus_sim_rand_joint_0061(){
    # bash run.sh argus_sim_rand_joint_0061 -p
    argus_sim_rand_joint
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512/sim_rand_joint_0061_argus_dof20_minimum.urdf/sim_rand_joint_0061_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0061_argus_dof20_minimum.urdf
    )
}

argus_sim_rand_joint_0303(){
    # bash run.sh argus_sim_rand_joint_0303 -p
    argus_sim_rand_joint
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512/sim_rand_joint_0303_argus_dof20_minimum.urdf/sim_rand_joint_0303_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0303_argus_dof20_minimum.urdf
    )
}


argus_sim_rand_joint_terrain_discrete(){
    # bash run.sh argus_sim_rand_joint_terrain_discrete -p
    argus_base
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512_terrain_discrete/sim_rand_joint_0000_argus_dof20_minimum.urdf/sim_rand_joint_0000_argus_dof20_minimum.urdf_newest.pt
    
        task.env.terrain.numLevels=10
        task.env.terrain.numTerrains=1
        task.env.terrain.minInitMapLevel=0
        task.env.terrain.maxInitMapLevel=10
        task.env.terrain.numLevels=10
        task.env.terrain.numTerrains=5
        task.env.terrain.discrete.height=0.1
    )
    BASE_ARGS+=(
        --wandb_run_name=argus_sim_rand_joints_terrain_discrete
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0000_argus_dof20_minimum.urdf
    
        task.env.terrain.terrainType=heightfield
        task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
    )
}


argus_sim_rand_joint_terrain_discrete_0511(){
    # bash run.sh argus_sim_rand_joint_terrain_discrete_0511 -p
    argus_sim_rand_joint_terrain_discrete
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512/sim_rand_joint_0511_argus_dof20_minimum.urdf/sim_rand_joint_0511_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0511_argus_dof20_minimum.urdf
    )
}

argus_sim_rand_joint_terrain_discrete_0061(){
    # bash run.sh argus_sim_rand_joint_terrain_discrete_0061 -p
    argus_sim_rand_joint_terrain_discrete
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512_terrain_discrete/sim_rand_joint_0061_argus_dof20_minimum.urdf/sim_rand_joint_0061_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0061_argus_dof20_minimum.urdf
    )
}


argus_sim_rand_joint_terrain_discrete_0303(){
    # bash run.sh argus_sim_rand_joint_terrain_discrete_0303 -p
    argus_sim_rand_joint_terrain_discrete
    PLAY_ARGS+=(
        --checkpoint=../envs/runs/argus_sim_rand_joint/train/trial_512_terrain_discrete/sim_rand_joint_0303_argus_dof20_minimum.urdf/sim_rand_joint_0303_argus_dof20_minimum.urdf_newest.pt
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/argus_sim_rand_joint/sim_rand_joint_0303_argus_dof20_minimum.urdf
    )
}