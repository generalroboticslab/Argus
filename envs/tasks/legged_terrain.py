import os
import sys
import time
import datetime
import numpy as np
from typing import Dict, Any, Tuple, Union
from operator import itemgetter
from gym import spaces
from collections.abc import Iterable
import orjson
import torch.nn.functional as F
from urdfpy import URDF
from isaacgym.torch_utils import get_axis_params, torch_rand_float, quat_rotate_inverse, quat_apply, normalize, quat_conjugate,quat_mul,scale
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask
from scipy.spatial.transform import Rotation
import torch
from envs.common.utils import bcolors as bc
from envs.common.urdf_utils import get_leaf_nodes, trace_edges, urdf_to_graph
from envs.common.publisher import DataPublisher, DataReceiver
from envs.common.terrain import Terrain
from isaacgym import gymutil
from hydra.utils import to_absolute_path
from common.tof_sensor import depth_to_tof, make_noisy_depth,tof_to_depth
from scipy.spatial.transform import Rotation as R
from envs.common import buffer
from torchvision.io import decode_image
from torchvision import transforms
import cv2
import trimesh

class LeggedTerrain(VecTask):
    """
    issaac gym envs any legged robot locomotion task
    """

    def __init__(
            self,
            cfg: Dict[str, Any],
            rl_device: str,
            sim_device: str,
            graphics_device_id: int,
            headless: bool,
            virtual_screen_capture: bool,
            force_render: bool
            ):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array
                                    via `env.render(mode='rgb_array')`.
            force_render: Set to True to always force rendering in the steps
                          (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.init_done = False

        self.cfg = cfg
        self.rl_device = rl_device
        self.headless = headless  # if training in a headless mode
        self.virtual_screen_capture = virtual_screen_capture
        self.force_render = force_render

        # set device and rl_device
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if self.cfg["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() in {"cuda", "gpu"}:
                self.device = f"cuda:{self.device_id}"
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                self.cfg["sim"]["use_gpu_pipeline"] = False

        # Rendering
        self.graphics_device_id = graphics_device_id
        self.camera_sensor_enable = self.cfg["env"].get("camera_sensor", {}).get("enable", False)
        if self.camera_sensor_enable:
            self.camera_sensor_size:tuple = tuple(self.cfg["env"]["camera_sensor"]["size"])
            self.camera_sensor_visualize = self.cfg["env"]["camera_sensor"]["visualize"]
            self.camera_on_foot = self.cfg["env"]["camera_sensor"]["on_foot"]
            self.obs_image_shape = ((20,) if self.camera_on_foot else (12,)) + self.camera_sensor_size

        self.enable_keyboard_operator: bool = self.cfg["env"]["viewer"]["keyboardOperator"]
        # self.enable_viewer_sync = False  # by default freeze the viewer until "V" is pressed
        self.enable_viewer_sync: bool = self.cfg["env"]["viewer"]["sync"]

        if (not self.camera_sensor_enable) and self.headless:
            self.graphics_device_id = -1
        self.num_environments = self.cfg["env"]["numEnvs"]  # self.num_envs
        self.num_agents = self.cfg["env"].get("numAgents", 1)  # used for multi-agent environments

        # self.sim_params = self._VecTask__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        self.sim_params = self._parse_sim_params()
        self.gym = gymapi.acquire_gym()

        # Creates the physics simulation and terrain.
        self.up_axis_idx = {"x": 0, "y": 1, "z": 2}[self.cfg["sim"]["up_axis"]]  # index of up axis: x=0, y=1, z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # keep still at zero command
        self.keep_still_at_zero_command: bool = self.cfg["env"].get("keep_still_at_zero_command",True)

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # treat commends below this threshold as zero [m/s]
        # square it to compare the square sum instead of the norm (slightly more efficient)
        self.command_zero_threshold: float = np.square(self.cfg["env"]["commandZeroThreshold"])
        self.command_zero_probability: float = self.cfg["env"]["commandZeroProbability"]
        # TODO group them together in config

        # commands: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float, device=self.device
        )
        self.is_zero_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # time related
        self.decimation: int = self.cfg["env"]["control"]["decimation"]
        self.dt: float = self.cfg["sim"]["dt"]
        self.dt_inv = 1.0 / self.dt
        self.rl_dt = self.dt*self.decimation
        self.rl_dt_inv = 1.0 / self.rl_dt
        self.max_episode_length_s: float = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.rl_dt + 0.5)
        self.command_zero_threshold_distance = self.max_episode_length_s * self.command_zero_threshold


        self.use_ray_obs = self.cfg['env']['ray_obs']['enable']
        self.object_tracking_enabled = self.cfg['env']['objectTracking']['enable']
        self.object_pushing_enabled = self.cfg['env']['objectPushing']['enable']

        if self.object_tracking_enabled:
            self.y_tracking_position = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device)
            self.x_tracking_position = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.y_tracking_direction = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.x_tracking_direction = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.random_x_displacement = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.random_y_displacement = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.object_robot_initial_distance =  self.cfg['env']['objectTracking']["objectRobotInitialDistance"]
            self.displacement_range = [self.cfg["env"]['objectTracking']["velocity_range"][0]*self.rl_dt, self.cfg["env"]['objectTracking']["velocity_range"][1]*self.rl_dt]
            self.object_vel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        if self.object_pushing_enabled:
            self.object_goal_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        # load assets
        self.load_asset()


        def get_dof_param(param: Union[float, int, dict, Iterable]) -> torch.Tensor:
            """
            Helper function to get a tensor of parameters for each actuated degree of freedom.
            Args:
                param: parameters, can be a single float/int, a dict, or an iterable
            Returns:
                a tensor of parameters with shape (1, num_actuated_dof), and a dict of named parameters
            """
            if isinstance(param, (float, int)):  # single value
                dof_param = np.full(shape=self.num_actuated_dof, fill_value=param)
            elif isinstance(param, dict):  # dict of parameters
                dof_param = np.zeros(self.num_actuated_dof)
                if "default" in param:  # default parameter
                    dof_param[:] = param["default"]
                for dof_name, value in param.items():
                    if dof_name == "default":  # skip the default parameter
                        continue
                    actual_names = get_matching_str(source=dof_name, destination=self.dof_names, case_sensitive=False, comment="")
                    for name in actual_names:  # set the parameter for each matching joint
                        dof_param[self.dof_dict[name]] = value
            elif isinstance(param, Iterable):  # iterable of parameters
                dof_param = np.array(param)
            named_dof_param = dict(zip(self.dof_names, dof_param))
            dof_param = torch.tensor(dof_param, device=self.device, dtype=torch.float).view(1, -1).repeat(self.num_envs, 1)
            return dof_param,named_dof_param

        self.named_default_dof_pos = self.cfg["env"].get("defaultJointPositions", {n: 0 for n in self.dof_names})
        self.default_dof_pos,self.named_default_dof_pos = get_dof_param(self.named_default_dof_pos)

        self.named_desired_dof_pos = self.cfg["env"].get("desiredJointPositions", self.named_default_dof_pos)
        self.desired_dof_pos,self.named_desired_dof_pos = get_dof_param(self.named_desired_dof_pos)

        self.named_init_dof_pos = self.cfg["env"].get("initialJointPositions", self.named_desired_dof_pos)
        self.init_dof_pos,self.named_init_dof_pos = get_dof_param(self.named_init_dof_pos)

        # control
        self.kp_default, _ = get_dof_param(self.cfg["env"]["control"]["stiffness"]) # Stiffness [N/m]
        self.kd_default,_ = get_dof_param(self.cfg["env"]["control"]["damping"]) # Damping [Ns/m]
        self.ki_default, _ = get_dof_param(self.cfg["env"]["control"].get("integral", 0.0)) # Integral [N.s/m]

        self.kp = self.kp_default.clone()
        self.kd = self.kd_default.clone()
        self.ki = self.ki_default.clone()

        self.dof_force_target_limit,_ = get_dof_param(self.cfg["env"]["control"]["limit"]) # Torque limit [N.m]

        self.dof_force_target_bound_ratio: float = self.cfg["env"]["learn"]["reward"]["dof_force_target"]["bound_ratio"]
        self.dof_force_target_soft_bound_max = self.dof_force_target_bound_ratio * self.dof_force_target_limit
        self.dof_force_target_soft_bound_min = -self.dof_force_target_bound_ratio * self.dof_force_target_limit

        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.voltage_motor = 48.0 # [V] motor voltage,note that motor torque is not realted to voltage
        self.radius_rotor_wheel = 0.04 # [m] radius of the rotor wheel

        motor_type = self.cfg["env"]["motor_type"]
        if motor_type == "robstride_02":
            self.dof_max_torque = 15 # [N.m] maximum torque
            def get_dof_max_linear_force_r02(dof_vel): # for robstrid 02 motor
                return torch.clip(self.voltage_motor-6-torch.abs(dof_vel/self.radius_rotor_wheel),min=0,max=self.dof_max_torque)/self.radius_rotor_wheel
            self.get_dof_max_linear_force_to_accelerate = get_dof_max_linear_force_r02
        elif motor_type == "robstride_03":
            self.dof_max_torque = 55 # [N.m] maximum torque
            def get_dof_max_linear_force_r03(dof_vel): # for robstrid 03 motor
                return torch.clip(self.voltage_motor+72-5*torch.abs(dof_vel/self.radius_rotor_wheel),min=0,max=self.dof_max_torque)/self.radius_rotor_wheel
            self.get_dof_max_linear_force_to_accelerate = get_dof_max_linear_force_r03
        self.dof_max_linear_force = self.dof_max_torque/self.radius_rotor_wheel # [N] maximum linear force

        # TODO: MAYBE SET A BETTER SCALE FOR DOF FORCES
        self.dof_force_scale = self.cfg["env"]["learn"].get("dofForceScale",1/self.dof_force_target_limit)
        self.dof_force_target_scale = self.cfg["env"]["learn"].get("dofForceTargetScale",1/self.dof_force_target_limit)
        self.heightmap_scale = self.cfg["env"]["learn"]["heightMapScale"]

        # update pos to default dof pos
        self.asset_urdf.update_cfg(self.named_init_dof_pos)
        lower_bound_z = self.asset_urdf.collision_scene.bounding_box.bounds[0,2]

        # target base height [m]
        self.base_height_target = self.cfg["env"].get("baseHeightTarget",None)

        if self.base_height_target is None:
            base_height_offset= self.cfg["env"].get("baseHeightOffset",0.1)
            base_height_tareget_offset = self.cfg["env"].get("baseHeightTargetOffset",0)
            self.base_height_target = -lower_bound_z + base_height_tareget_offset
            self.cfg["env"]["baseInitState"]["pos"][2] = float(self.base_height_target+base_height_offset)
            print(f"{bc.WARNING}[infer from URDF] target_base_height = {self.base_height_target:.4f} {bc.ENDC}")
            print(f"{bc.WARNING}[infer from URDF] self.cfg['env']['baseInitState']['pos'][2] = {self.cfg['env']['baseInitState']['pos'][2]:.4f} {bc.ENDC}")

        # needed for foot height reward
        base_pos = self.asset_urdf.get_transform(self.base_name,collision_geometry=True)[:3,3]
        base_pos = torch.tensor(base_pos, dtype=torch.float, device=self.device)
        foot_pos_rel = np.stack([self.asset_urdf.get_transform(foot_name,collision_geometry=True)[:3,3] for foot_name in self.foot_names])
        self.default_foot_pos_rel = torch.tensor(foot_pos_rel, dtype=torch.float, device=self.device).unsqueeze(0) #(1,num_foot,3)
        foot_z_pos = np.mean(foot_pos_rel[:,2]) # z position of foot
        self.foot_height_offset: float = lower_bound_z - foot_z_pos
        print(f"{bc.WARNING}[infer from URDF] foot_height_offset = {self.foot_height_offset:.4f} {bc.ENDC}")

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        np.testing.assert_almost_equal(np.square(rot).sum(), 1, decimal=6, err_msg="env.baseInitState.rot should be normalized to 1")
        self.base_init_state_default = torch.tensor(pos + rot + v_lin + v_ang, dtype=torch.float, device=self.device)
        self.base_init_state = self.base_init_state_default.repeat(self.num_envs, 1)  # (num_envs, 13)

        # other
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        self.enable_data_publisher: bool = self.cfg["env"]["dataPublisher"]["enable"]
        if self.enable_data_publisher:  # plotJuggler related
            self.data_publisher = DataPublisher(**self.cfg["env"]["dataPublisher"])
            self.items_to_publish = self.cfg["env"]["dataPublisher"].get("keys", None)
            self.data_root_label = self.cfg["env"]["data_root_label"]
        self.enable_sim2real_data_publisher: bool = self.cfg["env"]["sim2realDataPublisher"]["enable"]
        if self.enable_sim2real_data_publisher:
            self.sim2real_publisher = DataPublisher(**self.cfg["env"]["sim2realDataPublisher"])

        self.enable_data_receiver: bool = self.cfg["env"]["dataReceiver"]["enable"]
        if self.enable_data_receiver:
            self.data_receiver = DataReceiver(**self.cfg["env"]["dataReceiver"])
            self.data_receiver_data_id = self.data_receiver.data_id # for check if data is new
            self.data_receiver.receive_continuously()

        gravity_norm = np.linalg.norm(self.cfg["sim"]["gravity"])

        # reward scales
        cfg_rew: Dict[str, Any] = self.cfg["env"]["learn"]["reward"]
        self.rew_scales = {key: rew_item["scale"] for key,rew_item in cfg_rew.items()}
        for key in self.rew_scales:
            self.rew_scales[key] = float(self.rew_scales[key]) * self.rl_dt
        # do not scale termination reward
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]

        # self.rew_lin_vel_xy_exp_scale: float = cfg_rew["lin_vel_xy"]["exp_scale"]
        # self.rew_lin_vel_z_exp_scale: float = cfg_rew["lin_vel_z"]["exp_scale"]
        # self.rew_ang_vel_z_exp_scale: float = cfg_rew["ang_vel_z"]["exp_scale"]

        # self.rew_orient_exp_scale: float = cfg_rew["orientation"]["exp_scale"]
        self.rew_foot_orient_exp_scale: float = cfg_rew["foot_orientation"]["exp_scale"]

        self.rew_action_exp_scale: float = cfg_rew["action"]["exp_scale"]

        self.rew_action_rate_exp_scale: float = cfg_rew["action_rate"]["exp_scale"]

        self.rew_dof_acc_exp_scale: float = cfg_rew["dof_acc"]["exp_scale"]

        self.rew_dof_vel_exp_scale: float = cfg_rew["dof_vel"]["exp_scale"]

        self.rew_dof_pos_exp_scale: float = cfg_rew["dof_pos"]["exp_scale"]

        # self.rew_dof_jerk_exp_scale: float = cfg_rew["dof_jerk"]["exp_scale"]

        # self.rew_impact_exp_scale: float = cfg_rew["impact"]["exp_scale"]

        self.rew_lin_vel_exp_scale: torch.Tensor =cfg_rew["lin_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["lin_vel"]["normalize_by"],dtype=torch.float, device=self.device)
        self.rew_ang_vel_exp_scale: torch.Tensor =cfg_rew["ang_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["ang_vel"]["normalize_by"],dtype=torch.float, device=self.device)

        self.rew_foot_forward_exp_scale: float = cfg_rew["foot_forward"]["exp_scale"]

        self.rew_foot_pos_exp_scale: torch.Tensor = cfg_rew["foot_pos"]["exp_scale"] * \
            torch.tensor(cfg_rew["foot_pos"]["normalize_by"],dtype=torch.float, device=self.device)

        self.foot_contact_threshold: float = self.cfg["env"]["learn"]["foot_contact_threshold"]
        self.max_foot_contact_force = 100  # [N] # todo refactor
        # foot_height reward
        self.foot_height_clamp_max = cfg_rew["foot_height"]["clamp_max"]

        self.rew_dof_force_target_exp_scale: float = cfg_rew["dof_force_target"]["exp_scale"]
        dof_force_limit = torch.tensor(self.dof_props["effort"],dtype=torch.float, device=self.device)[self.actuated_dof_mask]
        assert torch.all(dof_force_limit < 10000), f"dof_force_limit={dof_force_limit} double check this value is correct"

        self.actuated_dof_force_max = dof_force_limit
        self.actuated_dof_force_min = -dof_force_limit
        # self.rew_dof_force_target_exp_scale= (self.rew_dof_force_target_exp_scale/ (dof_force_limit*self.num_actuated_dof))


        # # base height reward: reverse bell shaped curve
        # # https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
        # a, b = self.cfg["env"]["learn"].get("baseHeightRewardParams", [0.04, 3])
        # self.base_height_rew_a, self.base_height_rew_b = float(a), float(b)

        self.rew_base_height_exp_scale = cfg_rew["base_height"]["exp_scale"]
        self.rew_robot_object_distance_exp_scale = cfg_rew["robot_object_distance"]["exp_scale"]

        # min air time and stance time in seconds
        self.air_time_offset = float(cfg_rew["air_time"]["offset"])
        self.stance_time_offset = float(cfg_rew["stance_time"]["offset"])

        # ramdomize:push robot
        randomize = self.cfg["env"]["randomize"]
        self.should_push_robots = randomize["push"]["enable"]
        self.push_enable_at_reset = randomize["push"]["enable_at_reset"]
        self.push_interval = int(randomize["push"]["interval_s"] / self.rl_dt + 0.5)
        self.push_vel_min = torch.tensor(randomize["push"]["velMin"], dtype=torch.float, device=self.device)
        self.push_vel_max = torch.tensor(randomize["push"]["velMax"], dtype=torch.float, device=self.device)
        self.push_vel = torch.empty(self.num_envs, 6, dtype=torch.float, device=self.device) # vx, vy, vz, wx, wy, wz

        # randomize: init_dof_pos
        self.randomize_init_dof_pos = randomize["initDofPos"]["enable"]
        self.randomize_init_dof_pos_range = randomize["initDofPos"]["range"]

        # randomize: init_dof_vel
        self.randomize_init_dof_vel = randomize["initDofVel"]["enable"]
        self.randomize_init_dof_vel_range = randomize["initDofVel"]["range"]

        self.randmoize_base_init_orientation:bool = randomize["base_init_orientation"]["enable"]

        # randomize: erfi
        self.enable_erfi:bool = randomize["erfi"]["enable"]
        self.erfi_rfi_range:Iterable[float] = randomize["erfi"]["rfi_range"] # random force injection
        self.erfi_rao_range:Iterable[float] = randomize["erfi"]["rao_range"] # random actuation offset
        if self.enable_erfi:
            self.erfi_rao = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)
            self.erfi_rfi = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rfi_range)
        else:
            self.erfi_rfi = 0

        # randomize: dof_strength
        self.randomize_dof_strength = randomize["dof_strength"]["enable"]
        if self.randomize_dof_strength:
            self.dof_strength_range:Iterable[float] = randomize["dof_strength"]["range"]
            self.dof_strength = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.dof_strength_range)
        else:
            self.dof_strength = torch.ones(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device)

        # randomize: control pd values
        self.randomize_control_pd = randomize["control_pd"]["enable"]
        if self.randomize_control_pd:
            self.control_kp_range:Iterable[float] = randomize["control_pd"]["stiffness_range"]
            self.control_kd_range:Iterable[float] = randomize["control_pd"]["damping_range"]
            assert self.control_kp_range[0] > 0 and self.control_kp_range[0] > 0
            assert self.control_kp_range[1] > 0 and self.control_kp_range[1] > 0

        # random force purturbation
        self.randomize_body_force = randomize["body_force"]["enable"]
        self.force_scale=0
        if self.randomize_body_force:
            self.force_scale = randomize["body_force"]["scale"]
            self.force_log_prob_range = np.log(randomize["body_force"]["prob_range"])
            force_decay_time_constant:float = randomize["body_force"]["decay_time_constant"]
            self.force_decay = np.exp(-self.rl_dt/force_decay_time_constant)
            self.random_force_prob = torch.empty(self.num_envs, device=self.device).uniform_(*self.force_log_prob_range).exp_()
        # object apply random forces parameters
        if self.object_tracking_enabled or self.object_pushing_enabled:
            self.rb_forces = torch.zeros((self.num_envs, self.num_bodies+1, 3), dtype=torch.float, device=self.device)
        else:
            self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # randomize default dof pos
        # possibly check out https://pytorch.org/docs/stable/generated/torch.randn.html
        self.randomize_default_dof_pos = randomize["default_dof_pos"]["enable"]
        if self.randomize_default_dof_pos:
            self.default_dof_pos_range = randomize["default_dof_pos"]["range"]
            self.default_dof_pos+=torch.empty_like(self.default_dof_pos).uniform_(*self.default_dof_pos_range)

        # randomize action delay
        self.randomize_action_delay = randomize["action_delay"]["enable"]
        if self.randomize_action_delay:
            self.action_delay_log_range = np.log(randomize["action_delay"]["range"])
            self.action_delay = torch.empty((self.num_envs,1), dtype=torch.float, device=self.device).uniform_(*self.action_delay_log_range).exp_()

        self.randomize_projected_gravity_delay = randomize["projected_gravity_delay"]["enable"]
        if self.randomize_projected_gravity_delay:
            self.projected_gravity_delay_log_range = np.log(randomize["projected_gravity_delay"]["range"])
            self.projected_gravity_delay = torch.empty((self.num_envs,1), dtype=torch.float, device=self.device).uniform_(*self.projected_gravity_delay_log_range).exp_()

        self.randmoize_orientation_delay = randomize["orientation_delay"]["enable"]
        if self.randmoize_orientation_delay:
            self.orientation_delay_log_range = np.log(randomize["orientation_delay"]["range"])
            self.orientation_delay = torch.empty((self.num_envs,1), dtype=torch.float, device=self.device).uniform_(*self.orientation_delay_log_range).exp_()


        # heightmap
        self.init_height_points()

        self.is_train = False
        if  self.cfg["test"] in ["train", False]:
            self.is_train = True

        # passive dynamics
        self.enable_passive_dynamics = self.cfg["env"]["learn"].get("enablePassiveDynamics", False)
        self.passive_curriculum = self.cfg["env"]["learn"].get("passiveCurriculum", False)
        self.passive_curriculum = self.passive_curriculum and self.is_train # only train with curriculum

        self.action_is_on_rate = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)

        infer_action = self.cfg["env"]["numActions"] == "infer"
        if infer_action:
            self.cfg["env"]["numActions"] = len(self.actuated_dof_names)
            if self.enable_passive_dynamics:
                self.min_action_is_on:float = self.cfg["env"]["learn"]["action_is_on_min"] # 0.1
                self.action_is_on_sigmoid_k:float = self.cfg["env"]["learn"]["action_is_on_sigmoid_k"]
                self.cfg["env"]["numActions"] = len(self.actuated_dof_names)*2
                self.duration_since_action_switch = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
                self.last_action_is_on = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)


        # observation dimensions of specific items
        self.obs_dim_dict = {
            "linearVelocity": 3,
            "worldSpaceAngularVelocity": 3,
            "angularVelocity": 3,
            "projectedGravity": 3,
            "projected_gravity_xy": 2,
            "projected_gravity_filtered": 3,
            "base_rotation_matrix": 9,
            "base_rotation_matrix_filtered": 9,
            "commands": 3,  # vel_x,vel_y, vel_yaw, (excluding heading)
            "commands_xy": 2,
            "dofPosition": self.num_actuated_dof,
            "dofVelocity": self.num_actuated_dof,
            "dof_force_target": self.num_actuated_dof,
            "dof_strength": self.num_actuated_dof,
            "dofForce": self.num_actuated_dof,
            "heightMap": self.num_height_points,  # excluding the base origin measuring point
            "base_height": 1,
            "actions": self.cfg["env"]["numActions"],
            "last_actions": self.cfg["env"]["numActions"],
            "contact": self.num_foot,  # foot contact indicator
            "phase": self.num_foot*2, # phase of each foot (contact sequece)
            "contactTarget": self.num_foot,  # foot contact indicator
            "robot_root_position": 3,
        }
        # object tracking
        if self.object_tracking_enabled or self.object_pushing_enabled:
            self.obs_dim_dict["object_velocity"] = 3
            self.obs_dim_dict["object_orientation"] = 4
            self.obs_dim_dict["object_root_state"] = 13
            self.obs_dim_dict["object_goal_velocity"] = 3
            self.object_orientation_history_buffer = torch.zeros(
                (self.num_envs, 5*4),
                dtype=torch.float,
                device=self.device,
            )

        self.obs_names = tuple(self.cfg["env"]["observationNames"])
        self.obs2_names = tuple(self.cfg["env"].get("observation2Names", {}))
        self.use_obs2 = True if len(self.obs2_names)>0 else False

        self.num_stacked_obs_frame: int = cfg["env"].get("num_stacked_obs_frame", 1)
        self.num_stacked_state_frame: int = cfg["env"].get("num_stacked_state_frame", 1)

        # delayed observations
        self.max_observation_delay_steps: int = cfg["env"].get("max_observation_delay_steps",0 )

        def make_obs(obs_names, num_stacked_frame, delay_steps=0):
            num_obs_single_frame: int = np.sum(itemgetter(*obs_names)(self.obs_dim_dict))
            num_obs = num_obs_single_frame * num_stacked_frame
            batched_obs_buf = buffer.BatchedRingTensorBuffer(
                buffer_len=delay_steps+num_stacked_frame,
                batch_size=self.num_envs, shape=num_obs_single_frame, dtype=torch.float, device=self.device)
            obs_space = spaces.Box(np.ones(num_obs) * -np.Inf, np.ones(num_obs) * np.Inf)
            return num_obs_single_frame, num_obs, batched_obs_buf, obs_space

        # setup observations
        self.num_obs_single_frame, self.num_observations, self.batched_obs_buf, self.obs_space = \
            make_obs(self.obs_names, self.num_stacked_obs_frame, self.max_observation_delay_steps)
        self.cfg["env"]["numObservations"] = self.num_observations
        print(f"\033[93m[inferring] numObservations={self.num_obs}\033[0m")

        # setup observation2
        if self.use_obs2:
            self.num_obs2_single_frame, self.num_obs2, self.batched_obs2_buf, self.obs2_space = \
                make_obs(self.obs2_names, self.num_stacked_obs_frame, self.max_observation_delay_steps)
            self.cfg["env"]["numObservations2"] = self.num_obs2
            print(f"\033[93m[inferring] numObservations2={self.num_obs2}\033[0m")

        # setup states
        self.asymmetric_obs = self.cfg["env"].get("asymmetric_observations", False)
        if self.asymmetric_obs:
            self.state_names = tuple(self.cfg["env"]["stateNames"])
            self.num_state_single_frame, self.num_states, self.batched_states_buf, self.state_space = \
                make_obs(self.state_names, self.num_stacked_state_frame, delay_steps=0)
            self.cfg["env"]["numStates"] = self.num_states
            print(f"\033[93m[inferring] numStates={self.cfg['env']['numStates']}\033[0m")
        self.num_states = self.cfg["env"].get("numStates", 0)


        self.num_actions = self.cfg["env"]["numActions"]
        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = self.cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0
        self.lin_vel_curriculum = 0.5
        self.render_fps: int = self.cfg["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = os.path.join("recorded_frames", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # randomization_related_parameters
        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False

        # create plane/triangle mesh/heigh field
        self.terrain_type = self.cfg["env"]["terrain"]["terrainType"]
        self.terrain = Terrain(
            self.cfg["env"]["terrain"], num_robots=self.num_envs, device=self.device, gym=self.gym, sim=self.sim
        )

        if self.terrain_type in {'trimesh', 'heightfield'}:
            self.custom_origins = True
        elif self.terrain_type in {'plane', 'diy'}:
            self.custom_origins = False
        else:
            raise NotImplementedError(f'Unsupported terrain type: {self.terrain_type}')

        if self.cfg['env']['save_blender_trajectory']:
            self.rb_position_blender_trajectory = []
            self.ray_point_blender_recording = []
            self.object_state_blender_recording = []
            self.ray_pixel_blender_recording = []

        self._create_envs()

        if self.use_ray_obs:
            self.enable_ray_visualization = self.cfg["env"]["ray_obs"]["visualize_ray_point_cloud"]
            self.point_history_length = self.cfg['env']["ray_obs"]['point_history_length']
            self.num_points_per_foot = self.cfg['env']["ray_obs"]['resolution'][0] * self.cfg['env']["ray_obs"]['resolution'][1]
            self.num_perception_units = self.cfg['env']["ray_obs"]['num_perception_units']
            self.num_points = self.num_points_per_foot*self.num_foot
            self.point_cloud_history_buffer = torch.zeros(
                (self.num_envs, self.point_history_length*self.num_points,3),
                dtype=torch.float,
                device=self.device,
            )
            self.distance_history_buffer = torch.zeros(
                (self.num_envs, self.point_history_length*self.num_points),
                dtype=torch.float,
                device=self.device,
            )
            self.obs_dim_dict["ray_point_cloud"] = self.point_history_length*self.num_points  # num_points * 3 (x,y,z)

        self.foot_contact = torch.zeros(self.num_envs, self.num_foot, dtype=torch.bool, device=self.device)
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.allocate_buffers()
        if self.use_obs2:
            self.obs2_buf = torch.zeros((self.num_envs, self.num_obs2), device=self.device, dtype=torch.float)

        self.obs_dict = {}

        #######
        # get gym GPU state tensors
        self.dof_state_raw = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.net_contact_force_raw = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.rb_state_raw = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.dof_force_tensor_raw = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        if self.object_tracking_enabled or self.object_pushing_enabled:
            self.all_root_state: torch.Tensor = gymtorch.wrap_tensor(self.root_state_raw) #index every 2 elements in the list 2 is the number of actors and we only need the first actor
            self.root_state = self.all_root_state[::2] # this creates a view of the tensor, so it is not a copy
        else:
            self.root_state: torch.Tensor = gymtorch.wrap_tensor(self.root_state_raw)
        self.base_quat: torch.Tensor = self.root_state[:, 3:7] # [x,y,z,w] # scalar last!!!
        self.base_pos: torch.Tensor = self.root_state[:, :3] # [x,y,z]

        self.base_quat_filtered = torch.zeros_like(self.base_quat)

        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(self.dof_state_raw)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.dof_pos_error = torch.zeros_like(self.dof_pos)
        self.dof_pos_error_integral = torch.zeros_like(self.dof_pos)

        self.dof_vel_computed = torch.zeros_like(self.dof_vel)
        self.dof_vel_computed_single_substep = torch.zeros_like(self.dof_vel)
        self.dof_vel_computed_all_substeps = torch.zeros(self.decimation, self.num_envs ,self.num_dof, device=self.device, dtype=torch.float)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_pos_substep = torch.zeros_like(self.dof_pos)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_vel_computed = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_vel)
        self.last_base_quat = torch.zeros(self.num_envs ,4, device=self.device, dtype=torch.float)

        self.dof_pos_last_sub_step = torch.zeros_like(self.dof_pos)

        # contact_force: (num_envs, num_bodies, xyz axis)
        self.contact_force: torch.Tensor = gymtorch.wrap_tensor(self.net_contact_force_raw).view(self.num_envs, -1, 3)

        # rb_state: (num_envs,num_rigid_bodies,13)
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.rb_state: torch.Tensor = gymtorch.wrap_tensor(self.rb_state_raw).view(self.num_envs, -1, 13)
        # dof force tensor
        self.dof_force: torch.Tensor = gymtorch.wrap_tensor(self.dof_force_tensor_raw).view(self.num_envs, self.num_dof)

        # user-specified actuation forces (including passive joint, although we do not use it) [num_envs, num_dof]
        self.dof_actuation_force = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        self.dof_actuation_force_tensor = gymtorch.unwrap_tensor(self.dof_actuation_force)

        if self.use_ray_obs:
            self.init_ray_casting()
            self.joint_directions = self.rb_state[:,self.foot_ids,:3] - self.rb_state[:,self.base_id,:3].unsqueeze(1)
            self.joint_origins = self.rb_state[:,self.foot_ids,:3]
            self.joint_quaternions = self.rb_state[:,self.foot_ids,3:7]
            self.robot_root_position = self.rb_state[:,self.base_id,:3]
            self.robot_root_quaternions = self.rb_state[:,self.base_id,3:7]

        # reward episode sums (unscaled)
        def torch_zeros(): return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_sums = {key: torch_zeros() for key in self.rew_scales.keys()}

        # initialize some data used later on
        self.common_step_counter: int = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.noise_vec = torch.zeros((self.num_envs, self.num_obs_single_frame),dtype=torch.float, device=self.device)

        # for force perturbation
        self.rb_force_mags =  gravity_norm * self.force_scale * self.actor_rigid_body_masses[:,:,None]

        # gravity_vec = [0,0,-1]
        gravity_vec = torch.tensor(get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device)
        self.base_gravity_vec = gravity_vec.repeat((self.num_envs, 1))
        self.projected_gravity = self.base_gravity_vec.clone()
        self.projected_gravity_filtered = self.base_gravity_vec.clone()
        self.foot_gravity_vec = gravity_vec.repeat((self.num_foot*self.num_envs, 1))
        self.base_rotation_matrix = torch.eye(3, device=self.device).repeat((self.num_envs, 1, 1))
        self.base_rotation_matrix_filtered = self.base_rotation_matrix.clone()

        # forward_vec = [1,0,0]
        forward_vec = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
        self.base_forward_vec_local = forward_vec.repeat((self.num_envs, 1))
        self.foot_forward_vec_local = forward_vec.repeat((self.num_foot*self.num_envs, 1))

        # idealized actuated dof force target
        self.actuated_dof_force_target = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
        # after applying randomization and noise
        self.actuated_dof_force_target_actual = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
        self.action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.action_to_use = torch.zeros_like(self.action)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.action_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # foot air time and stance time
        self.air_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.stance_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.last_foot_contact_force = torch.zeros(self.num_envs, self.num_foot, 3,
                                                   dtype=torch.float, device=self.device)
        
        # total mass of the actors [num_envs]
        total_mass = torch.tensor([np.sum([p.mass for p in body_props]) for body_props in self.actor_rigid_body_properties],dtype=torch.float, device=self.device)
        self.total_gravity = total_mass*gravity_norm
        self.total_gravity_inv = 1.0/self.total_gravity  # inverse of total gravity

        # single contact reward parameters
        self.max_single_contact: int = cfg_rew["single_contact"]["max_single_contact"]
        self.foot_multi_contact_grace_period: float = cfg_rew["single_contact"]["grace_period"]
        self.foot_multi_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.last_foot_contact = torch.zeros(self.num_envs, self.num_foot, dtype=torch.bool, device=self.device)

        cfg_guided_contact = self.cfg["env"]["learn"].get("guided_contact", {})
        self.guided_contact = cfg_guided_contact.get("enable", False)
        self.phase=None
        self.phase_sin_cos=None
        self.contact_target=None
        self.last_contact_target=None
        if self.guided_contact:
            self.phase_start_with_swing: bool = self.cfg["env"]["learn"]["guided_contact"]["phase_start_with_swing"]
            # normalized phase [0,1]
            self.episodic_phase_offset = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device)
            self.episodic_phase_offset[::2]= 0.5 # half of the episode is 0.5 phase offset
            self.phase_offset = torch.tensor(cfg_guided_contact["phase_offset"],dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.phase_freq: float = cfg_guided_contact["phase_freq"] # [Hz]
            self.phase_stance_ratio: float = cfg_guided_contact["phase_stance_ratio"] # [1]
            self.phase_swing_ratio = 1 - self.phase_stance_ratio
            self.last_contact_target = torch.ones(self.num_envs,self.num_foot, dtype=torch.bool, device=self.device)
            self.update_phase()

            # assume swing start from 0->self.phase_swing_ratio
            # compute 4th order polynomial coefficients
            sw = self.phase_swing_ratio
            q = 0.5 * sw
            A = np.array([
                [sw**2,sw**3,sw**4],
                [2*sw,3*sw**2,4*sw**3],
                [q**2,q**3,q**4]
            ])
            b = np.array([0,0,self.foot_height_clamp_max])
            self.foot_height_coeff = a =  np.linalg.solve(A, b) # y = a[0]*t**2+a[1]*t**3+a[2]*t**4
            self.foot_height_vel_coeff = np.array([2*a[0],3*a[1],4*a[2]])

            graph = urdf_to_graph(self.asset_urdf)
            # get mapping from a foot to the all the joints traced from that foot shaped [num_foot, num_dof] (including all passive and actuated joints)
            self.leg_to_dof_mask = torch.zeros(self.num_foot,self.num_dof, dtype=torch.bool, device=self.device)
            for k, foot_name in enumerate(self.foot_names):
                joint_traced_from_foot = trace_edges(graph, start_node=foot_name)
                joint_ids_traced_from_foot = [self.dof_dict[edge] for edge in joint_traced_from_foot]
                self.leg_to_dof_mask[k,joint_ids_traced_from_foot]=True
            self.leg_to_dof_mask = self.leg_to_dof_mask.repeat(self.num_envs,1,1) # expanded to [num_envs, num_foot, num_dof]

        if self.num_actuated_dof != self.num_dof: # has passive DOF
            self.pre_physics_step = self.pre_physics_step_with_passive_dof

        self.evaluate_dict = self.cfg["env"].get("evaluate", {})
        self.evaluate = self.evaluate_dict.get("enable", False)
        self.shape_evaluate = self.evaluate_dict.get("shape_evaluate", False)
        self.COM_evaluation = self.evaluate_dict.get("COM_evaluation", False)

        self.sim_step_count = 0

        self.compute_cost_of_transport_metrics = False
        # compute cost of transport metrics
        if self.evaluate or self.enable_data_publisher:
            self.compute_cost_of_transport_metrics = True
            buffer_len=250 # TODO,  5 seconds
            self.buffer_com_pos = buffer.BatchedRingTensorBuffer(buffer_len=buffer_len, batch_size=self.num_envs, shape=3, dtype=torch.float, device=self.device)
            self.buffer_dof_pow = buffer.BatchedRingTensorBuffer(buffer_len=buffer_len, batch_size=self.num_envs, shape=1, dtype=torch.float, device=self.device)

        if self.evaluate:
            self.evaluate_json_name = self.evaluate_dict.get("filename", "evaluate.json")
            self.orjson_option = orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE
            # get full path using the cwd
            self.evaluate_json_name = os.path.abspath(os.path.join(os.getcwd(), self.evaluate_json_name))
            if os.path.exists(self.evaluate_json_name):
                os.remove(self.evaluate_json_name)

            self.has_first_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # position at which the base touches the ground at the first time of the episode
            self.base_pos_at_first_contact = torch.zeros(self.num_envs,3, dtype=torch.float, device=self.device)
            if self.object_pushing_enabled:
                self.object_goal_vel_list_eval = []
                self.object_vel_list_eval = []
                self.robot_pos_list_eval = []
                self.object_pos_list_eval = []
            if self.object_tracking_enabled:
                self.robot_vel_list_eval = []
                self.object_vel_list_eval = []
                self.robot_pos_list_eval = []
                self.object_pos_list_eval = []
            self.dof_pos_list_eval = []
            self.robot_root_position_list_eval = []
            self.robot_orientation_list_eval = []  
            self.robot_linvel_list_eval= []
            self.robot_angularvel_list_eval= []
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True

        # rendering
        self.set_viewer()

        if self.cfg['env']['export_discrete_terrain']:
            import trimesh
            self.terrain_trimesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
            self.terrain_trimesh.export('output_discrete_mesh.obj')

    def init_ray_casting(self):

        #init ray point cloud setting
        self.combined_mesh_list = []
        if self.object_tracking_enabled:
            self.cube_size = self.cfg['env']["objectTracking"]['cube_size']
        elif self.object_pushing_enabled:
            self.cube_size = self.cfg['env']["objectPushing"]['cube_size']

        if self.terrain_type == "plane":
            self.plane = trimesh.primitives.Box(extents=[50, 50, 0.1], transform=trimesh.transformations.translation_matrix([0, 0, -0.1/2]))
            if self.object_tracking_enabled:
                self.cube = trimesh.primitives.Box(extents=[self.cube_size, self.cube_size, self.cube_size], 
                                                   transform=trimesh.transformations.translation_matrix([0.0, self.object_robot_initial_distance, 0.7]))
            if self.object_pushing_enabled:
                self.cube = trimesh.primitives.Box(extents=[self.cube_size, self.cube_size, self.cube_size], 
                                                   transform=trimesh.transformations.translation_matrix([0.0, self.cfg['env']['objectPushing']['cube_y_position'], 0.55]))
            for idx in range(self.num_envs):
                self.combined_mesh_list.append(trimesh.util.concatenate([self.plane,self.cube]))

        self.combined_mesh = self.combined_mesh_list[0]
        self.init_ray_distance = torch.full((self.num_envs, self.num_points), 1, dtype=torch.float, device=self.device)
        self.updated_ray_distance = self.init_ray_distance.clone()
        self.updated_ray_point_clouds = torch.full((self.num_envs, self.num_points,3), 0, dtype=torch.float, device=self.device)
        self.perfect_point_cloud_blender = torch.full((self.num_envs, self.num_points,3), 0, dtype=torch.float, device=self.device)

        self.num_step_to_delay_ray = self.cfg['env']['ray_obs']['num_step_to_delay_ray'] 
        # if self.cfg['evaluation_type'] == 'points':
        #     self.random_perception_mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        #     self.random_perception_mask[torch.randperm(self.num_points, device=self.device)[:int(self.num_points * self.cfg['evaluation_mask'])]] = True
        # elif self.cfg['evaluation_type'] == 'units':
        #     chunk_size = self.num_points_per_foot
        #     num_chunks = self.num_perception_units  # total number of chunks
        #     selected_chunks =  self.cfg['evaluation_mask']

        #     self.random_perception_mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        #     selected_indices = torch.randperm(num_chunks, device=self.device)[:selected_chunks]
        #     for idx in selected_indices:
        #         start = idx * chunk_size
        #         end = start + chunk_size
        #         self.random_perception_mask[start:end] = True

        self.sphere_pose = gymapi.Transform()

    def get_point_cloud_ray_casting_from_foot_batch(self,base_origins, foot_origins,directions,rotations):
        max_distance = self.cfg['env']['ray_obs']['max_distance']
        self.joint_origins = torch.tensor(foot_origins,dtype=torch.float,device=self.device)
        self.joint_directions = torch.tensor(directions,dtype=torch.float,device=self.device)
        self.joint_quaternions = torch.tensor(rotations,dtype=torch.float,device=self.device)
        self.robot_root_position = self.rb_state[:,self.base_id,:3]
        self.robot_root_quaternions = self.rb_state[:,self.base_id,3:7]
        self.updated_ray_distance = self.init_ray_distance.clone()
        self.updated_ray_point_clouds = torch.full((self.num_envs, self.num_points,3), 0, dtype=torch.float, device=self.device)

        for idx in range(self.num_envs):
            x = self.all_root_state[self.object_handles[idx], 0].cpu()
            y = self.all_root_state[self.object_handles[idx], 1].cpu()
            z = self.all_root_state[self.object_handles[idx], 2].cpu()
            q_x = self.all_root_state[self.object_handles[idx], 3].cpu()
            q_y = self.all_root_state[self.object_handles[idx], 4].cpu()
            q_z = self.all_root_state[self.object_handles[idx], 5].cpu()
            q_w = self.all_root_state[self.object_handles[idx], 6].cpu()

            translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
            rotation_matrix = trimesh.transformations.quaternion_matrix([q_w, q_x, q_y, q_z])
            transform_matrix = trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)

            self.plane = trimesh.primitives.Box(extents=[50, 50, 0.1], transform=trimesh.transformations.translation_matrix([0, 0, -0.1/2]))
            self.cube = trimesh.primitives.Box(extents=[self.cube_size, self.cube_size, self.cube_size], transform=transform_matrix)

            combined_mesh= trimesh.util.concatenate([self.plane,self.cube])

            ray_origins, ray_directions = get_ray_casting_pyramid(
                origins=foot_origins[idx], directions=directions[idx], quaternions=rotations[idx],
                resolution=[self.cfg['env']["ray_obs"]['resolution'][0],self.cfg['env']["ray_obs"]['resolution'][1]])

            # Perform ray casting
            ray_locations, index_ray, index_tri = combined_mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
            )

            # Compute distances and filter by range
            relative_distances = ray_locations - ray_origins[index_ray]
            ray_distances = np.linalg.norm(relative_distances, axis=1)
            clipped_distances = np.clip(ray_distances, 0.04, max_distance) #0.137 is an value that matched real sensor reading
            self.updated_ray_distance[idx, index_ray] = torch.tensor(clipped_distances, dtype=torch.float, device=self.device)

            if self.cfg['env']['save_blender_trajectory']:
                self.perfect_point_cloud_blender[idx][index_ray] = torch.tensor(ray_locations, dtype=torch.float, device=self.device)
                mask = ray_distances > 1.5
                self.perfect_point_cloud_blender[idx][index_ray[mask]] = 0
                if self.num_perception_units == 12:
                    self.perfect_point_cloud_blender[...,:-self.num_perception_units*self.num_points_per_foot,:]=0
                elif self.num_perception_units == 20:
                    self.perfect_point_cloud_blender[...,self.num_perception_units*self.num_points_per_foot:,:]=0
                    

            if self.cfg['env']['ray_obs']['ray_distance_to_pixel']:
                self.updated_ray_distance[idx, index_ray] = depth_to_tof(self.updated_ray_distance[idx, index_ray])
                if self.cfg['env']['ray_obs']['apply_tof_noise']:
                    self.updated_ray_distance[idx, index_ray] = make_noisy_depth(self.updated_ray_distance[idx, index_ray])

                self.updated_ray_distance[idx, index_ray] = self.updated_ray_distance[idx, index_ray]/255
            else:
                self.updated_ray_distance[idx, index_ray] = self.updated_ray_distance[idx, index_ray]/2.5

            combined_mesh = None
            locations_within_range = ray_locations[ray_distances <=  max_distance]

            if self.cfg['env']['save_blender_trajectory']:
                reshape_distance = self.updated_ray_distance[idx,:].cpu().detach().numpy().reshape(-1,self.num_points_per_foot)
            else:
                reshape_distance = self.distance_history_buffer[idx,self.num_points*(self.point_history_length-self.num_step_to_delay_ray-1):self.num_points*(self.point_history_length-self.num_step_to_delay_ray)].cpu().detach().numpy().reshape(-1,self.num_points_per_foot)

            point_positions = convert_ray_distance_to_position(
                        origins=foot_origins[idx],
                        directions=directions[idx],
                        quaternions=rotations[idx],
                        robot_root_pos=self.robot_root_position[idx].cpu().detach().numpy(),
                        distance_pixel_normalized=reshape_distance,
                        resolution=[5, 5],
                        dropout_rate=self.cfg['env']['ray_obs']['random_dropout'],  # 0.2 dropout
                        noise_std=self.cfg['env']['ray_obs']['random_noise'],   # 0.005
                        apply_noise=self.cfg['env']['ray_obs']['apply_noise_on_point_cloud']
                    )
            self.updated_ray_point_clouds[idx] = torch.tensor(point_positions, dtype=torch.float, device=self.device)

    def update_phase(self):
        """update normalized contact phase for each foot,
           stance phase: 0<phase<stance_ratio
           swing phase:  stance_ratio<phase<1
        """
        self.phase = (self.phase_freq * self.rl_dt * self.progress_buf.unsqueeze(1) +self.phase_offset + self.episodic_phase_offset) % 1
        self.phase_sin_cos = torch.column_stack((torch.sin(2*torch.pi*self.phase), torch.cos(2*torch.pi*self.phase)))

        if self.phase_start_with_swing:
            # NOTE! BREAKING CHANGE 09/07 : phase start with swing first instead
            self.contact_target = self.phase > self.phase_swing_ratio
        else:
            self.contact_target = self.phase <= self.phase_stance_ratio # 1 = stance, 0 = swing

        if self.keep_still_at_zero_command:
            # todo fix 0.5 bug
            should_stop = self.is_zero_command & (self.last_contact_target.sum(dim=-1)==self.num_foot) # & (self.projected_gravity[:,2]<-0.99) # 10 deg tilt
            self.contact_target[should_stop] = 1 # set contact_target to 1 if zero command
        self.last_contact_target[:] = self.contact_target


    def set_viewer(self):
        """set viewers and camera events"""

        # rendering: virtual display
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            SCREEN_CAPTURE_RESOLUTION = (1027, 768)
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()

        # todo: read from config
        if self.headless:
            self.viewer = None
            return

        # if running with a viewer, set up keyboard shortcuts and camera

        # subscribe to keyboard shortcuts
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        def subscribe_keyboard_event(key, event_str):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, event_str)

        subscribe_keyboard_event(gymapi.KEY_ESCAPE, "QUIT")
        subscribe_keyboard_event(gymapi.KEY_V, "toggle_viewer_sync")
        subscribe_keyboard_event(gymapi.KEY_R, "record_frames")
        subscribe_keyboard_event(gymapi.KEY_9, "reset")

        if self.enable_keyboard_operator:
            subscribe_keyboard_event(gymapi.KEY_I, "vx+")
            subscribe_keyboard_event(gymapi.KEY_K, "vx-")
            subscribe_keyboard_event(gymapi.KEY_J, "vy+")
            subscribe_keyboard_event(gymapi.KEY_L, "vy-")
            subscribe_keyboard_event(gymapi.KEY_U, "heading+")
            subscribe_keyboard_event(gymapi.KEY_O, "heading-")
            subscribe_keyboard_event(gymapi.KEY_0, "v=0")
            subscribe_keyboard_event(gymapi.KEY_Y, "m+")
            subscribe_keyboard_event(gymapi.KEY_H, "m-")
            subscribe_keyboard_event(gymapi.KEY_P, "push")

            self.keyboard_operator_cmd = torch.zeros(3, dtype=torch.float, device=self.device)

        subscribe_keyboard_event(gymapi.KEY_F, "toggle_viewer_follow")
        # switch camera follow target
        subscribe_keyboard_event(gymapi.KEY_LEFT_BRACKET, "ref_env-") # [
        subscribe_keyboard_event(gymapi.KEY_RIGHT_BRACKET, "ref_env+") # ]

        # set the camera position based on up axis
        # self.sim_params = self.gym.get_sim_params(self.sim)
        if self.sim_params.up_axis == gymapi.UP_AXIS_Z:
            self.cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            self.cam_target_pos = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            self.cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            self.cam_target_pos = gymapi.Vec3(10.0, 0.0, 15.0)

        self.cam_pos = torch.tensor(self.cfg["env"]["viewer"]["pos"], dtype=torch.float, device=self.device)
        self.cam_target_pos = torch.tensor(self.cfg["env"]["viewer"]["lookat"], dtype=torch.float, device=self.device)
        self.ref_env:int = int(self.cfg["env"]["viewer"]["refEnv"])%self.num_envs

        self.viewer_follow = self.cfg["env"]["viewer"]["follow"]
        self.viewer_follow_offset = torch.tensor(self.cfg["env"]["viewer"].get("follower_offset", [-1.5, -1.5, 0.4]), dtype=torch.float, device=self.device)
        fs = self.rl_dt_inv
        filter_order=1
        from common.buffer import RingTensorFilterBuffer
        self.cam_pos_filter_buffer = RingTensorFilterBuffer(fs=fs, cut_off_frequency=1,filter_order=filter_order,shape=3,device=self.device)
        self.cam_pos_filter_buffer.fill(self.cam_pos)
        self.cam_target_pos_filter_buffer = RingTensorFilterBuffer(fs=fs, cut_off_frequency=1,filter_order=filter_order,shape=3,device=self.device)
        self.cam_target_pos_filter_buffer.fill(self.cam_target_pos)
        if self.viewer_follow:
            self.cam_target_pos = self.root_state[self.ref_env, :3].clone()
            self.cam_pos = self.viewer_follow_offset + self.cam_target_pos

        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.ref_env], gymapi.Vec3(*self.cam_pos), gymapi.Vec3(*self.cam_target_pos))

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.01, 5, 5, None, color=(1, 1, 0))
        self.sphere_geom_alt_color = gymutil.WireframeSphereGeometry(0.01, 5, 5, None, color=(1, 0, 0))


    def _parse_sim_params(self):
        """Parse the config dictionary for physics stepping settings.
        Returns
            IsaacGym SimParams object with updated settings.
        """
        config_sim = self.cfg["sim"]
        physics_engine = self.cfg["physics_engine"]

        sim_params = gymapi.SimParams()
        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        elif config_sim["up_axis"] == "y":
            sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {config_sim['up_axis']}")

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        elif physics_engine == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {self.cfg['physics_engine']}")

        return sim_params

    def _get_noise_scale_vec(self):
        """Calculates noise scaling factors for observations."""
        cfg_learn = self.cfg["env"]["learn"]
        self.add_noise = cfg_learn["addNoise"]
        noise_level = cfg_learn["noiseLevel"]
        noise_dict = {
            "linearVelocity": cfg_learn["linearVelocityNoise"] * noise_level * self.lin_vel_scale,
            "worldSpaceAngularVelocity": cfg_learn["angularVelocityNoise"] * noise_level * self.ang_vel_scale,
            "angularVelocity": cfg_learn["angularVelocityNoise"] * noise_level * self.ang_vel_scale,
            "projectedGravity": cfg_learn["gravityNoise"] * noise_level,
            "projected_gravity_filtered": cfg_learn["gravityNoise"] * noise_level,
            "projected_gravity_xy": cfg_learn["gravityNoise"] * noise_level,
            "base_rotation_matrix": 0,
            "base_rotation_matrix_filtered": 0,
            "commands": 0,
            "commands_xy": 0,
            "dofPosition": cfg_learn["dofPositionNoise"] * noise_level * self.dof_pos_scale,
            "dofVelocity": cfg_learn["dofVelocityNoise"] * noise_level * self.dof_vel_scale,
            "dof_force_target": 0,
            "dof_strength": 0,
            "dofForce": 0, # TODO, MAYBE ADD NOISE FOR DOF FORCE
            "heightMap": cfg_learn["heightMapNoise"] * noise_level * self.heightmap_scale,
            "base_height": 0,
            "actions": 0,  # previous actions
            "last_actions": 0,
            "contact": 0,  # foot contact
            "contactTarget":0,
            "phase": 0,
            "robot_root_position": 0,
        }
        if self.object_tracking_enabled or self.object_pushing_enabled:
            noise_dict["object_velocity"] = 0
            noise_dict["object_root_state"] = 0
            noise_dict["object_orientation"] = 0

            noise_dict["object_goal_velocity"] = 0

        if self.use_ray_obs:
            noise_dict["ray_point_cloud"] = 0  # num_points * 3 (x,y,z)
        noise_vec_lists = [torch.ones(self.obs_dim_dict[name]) * noise_dict[name] for name in self.obs_names]
        noise_vec = torch.cat(noise_vec_lists, dim=-1).to(self.device)
        return noise_vec

    @property
    def asset_urdf(self):
        try:
            return self._asset_urdf
        except AttributeError:
            import yourdfpy
            self._asset_urdf = yourdfpy.URDF.load(
                self.asset_path,
                build_scene_graph=False,
                load_meshes=False,
                load_collision_meshes=True,
                build_collision_scene_graph=True
            )
            return self._asset_urdf

    def load_asset(self):
        """Loads the robot asset (URDF).
        Requires gym to be initialized
        """

        cfg_asset = self.cfg["env"]["urdfAsset"]
        cube_asset = self.cfg["env"]["urdfAsset"]["cube_asset"]

        if "root" not in cfg_asset:  # root directory
            cfg_asset["root"] = 'assets'  # relative to the legged_env folder
        if not os.path.isabs(cfg_asset["root"]):
            cfg_asset["root"] = os.path.abspath(
                os.path.join(os.path.dirname(to_absolute_path(__file__)), "./../../", cfg_asset["root"]))

        self.asset_path = os.path.join(cfg_asset["root"], cfg_asset["file"])

        # bitwise filter for elements in the same collisionGroup to mask off collision
        self.collision_filter = cfg_asset["collision_filter"]

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        self.override_inertia = cfg_asset["AssetOptions"].get("override_inertia", False)
        asset_options.override_inertia = self.override_inertia
        for attribute in cfg_asset["AssetOptions"]:
            if attribute == "vhacd_params":
                vhacd_params = cfg_asset["AssetOptions"]["vhacd_params"]
                for key in vhacd_params:
                    setattr(asset_options.vhacd_params, key, vhacd_params[key])
            elif hasattr(asset_options, attribute):
                setattr(asset_options, attribute, cfg_asset["AssetOptions"][attribute])
            else:
                print(f"{bc.WARNING}{attribute} not in AssetOptions!{bc.ENDC}")

        self.asset = self.gym.load_asset(self.sim, cfg_asset["root"], cfg_asset["file"], asset_options)
        print(f"\033[92mloaded asset {os.path.abspath(os.path.join(cfg_asset['root'], cfg_asset['file']))}\033[0m")
        if self.object_tracking_enabled or self.object_pushing_enabled:
            self.cube_asset_path = os.path.join(cfg_asset["root"], cube_asset)
            self.cube_asset = self.gym.load_asset(self.sim, cfg_asset["root"], cube_asset, asset_options)


        self.perception_asymmetry_experiment = self.cfg['env']['perception_asymmetry_experiment']['enable']
        self.perception_asymmetry_experiment_collect = self.cfg['env']['perception_asymmetry_experiment']['data_collection']
        if self.perception_asymmetry_experiment:
            asymmetry_design = self.cfg["env"]["urdfAsset"]["asymmetry_design_asset"]
            self.asymmetry_design_path = os.path.join(cfg_asset["root"], asymmetry_design)
            self.asymmetry_design = URDF.load(self.asymmetry_design_path)
            # Get home positions in one line
            fk_home = self.asymmetry_design.link_fk()

            # Create a name-to-link mapping
            link_dict = {link.name: link for link in self.asymmetry_design.links}

            # Extract both positions and orientations
            self.asymmetric_joint_positions = {}
            self.asymmetric_joint_orientations = {}

            for joint in self.asymmetry_design.joints:
                if joint.joint_type != 'fixed' and joint.child in link_dict and link_dict[joint.child] in fk_home:
                    transform_matrix = fk_home[link_dict[joint.child]]  # 4x4 transformation matrix
                    
                    # Extract position (translation)
                    position = transform_matrix[:3, 3]
                    self.asymmetric_joint_positions[joint.name] = position
                    
                    # Extract orientation (rotation matrix -> quaternion)
                    rotation_matrix = transform_matrix[:3, :3]
                    rotation = Rotation.from_matrix(rotation_matrix)
                    quaternion = rotation.as_quat()  # [x, y, z, w] format
                    self.asymmetric_joint_orientations[joint.name] = quaternion

        self.dof_names = self.gym.get_asset_dof_names(self.asset)
        # name to index mapping
        self.dof_dict = self.gym.get_asset_dof_dict(self.asset)
        # number of DOF
        self.num_dof = self.gym.get_asset_dof_count(self.asset)

        try:
            self.passive_dof_names = get_matching_str(source="PASSIVE", destination=self.dof_names, case_sensitive=True, comment="passive DOF")
        except KeyError:
            self.passive_dof_names = []
        self.actuated_dof_names = list(sorted([d for d in self.dof_names if d not in self.passive_dof_names]))
        self.num_actuated_dof = len(self.actuated_dof_names)
        actuated_dof_ids = torch.tensor([self.dof_dict[n] for n in self.actuated_dof_names], device=self.device, dtype=torch.long)
        self.actuated_dof_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        self.actuated_dof_mask[actuated_dof_ids] = True


        # dof properties
        # lower: lower limit of DOF. in [radians] or [meters]
        # upper: upper limit of DOF. in [radians] or [meters]
        # velocity: Maximum velocity of DOF. in [radians/s] or [meters/s]
        # effort: Maximum effort of DOF. in [N] or [Nm].
        # stiffness: DOF stiffness.
        # damping: DOF damping.
        # friction: DOF friction coefficient, a generalized friction force is calculated as DOF force multiplied by friction.
        # armature: DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia.
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        # asset dof properties override
        asset_dof_properties = self.cfg["env"].get("assetDofProperties", {})
        if asset_dof_properties is not None:
            for key, value in asset_dof_properties.items():
                self.dof_props[key][:] = np.asarray(value,dtype=np.float32)  # used in set_actor_dof_properties
                print(f"overwrite asset dof [{key}]: {value}")

        # dof limit
        self.dof_soft_limit_lower = torch.tensor(
            self.cfg["env"].get("dof_soft_limit",{}).get("lower",self.dof_props['lower']), dtype=torch.float, device=self.device)
        self.dof_soft_limit_upper = torch.tensor(
            self.cfg["env"].get("dof_soft_limit",{}).get("upper",self.dof_props['upper']),dtype=torch.float, device=self.device)

        dof_margin: float = self.cfg["env"]["learn"]["reward"]["dof_limit"]["margin"]
        dof_margin = torch.tensor(dof_margin, dtype=torch.float, device=self.device)
        self.dof_limit_lower = torch.tensor(self.dof_props['lower'], dtype=torch.float, device=self.device) + dof_margin
        self.dof_limit_upper = torch.tensor(self.dof_props['upper'], dtype=torch.float, device=self.device) - dof_margin

        # body
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.asset)

        asset_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        asset_rigid_body_id_dict = {value: key for key, value in asset_rigid_body_dict.items()}

        # body: base
        self.base_name = cfg_asset.get("baseName", None)
        if self.base_name is None:  # infer base_name
            self.base_name = self.asset_urdf.base_link
        else:
            self.base_name = get_matching_str(source=self.base_name, destination=self.body_names, comment="base_name")[0]
        self.base_id = asset_rigid_body_dict[self.base_name]

        foot_name = cfg_asset.get("footName", None)
        if foot_name is None:  # infering the feet are leaf links, they do not appear in any joint.parent
            self.foot_names = get_leaf_nodes(urdf=self.asset_urdf,
                collapse_fixed_joints=self.cfg["env"]["urdfAsset"]["AssetOptions"]["collapse_fixed_joints"])
        else:
            self.foot_names = get_matching_str(source=foot_name, destination=self.body_names, comment="foot_name")
        self.num_foot = len(self.foot_names)
        self.foot_ids = torch.tensor([asset_rigid_body_dict[n] for n in self.foot_names],
                                     dtype=torch.long, device=self.device)
        assert(self.foot_ids.numel() > 0)

        # TODO CHANGE KNEE COLLISIONN TO A MORE GENARIC TYPE OF REWARD: collision for anything other than the foot maybe?
        knee_name = cfg_asset.get("kneeName", None)
        if knee_name is None:
            # if knee_name is None: exclude base link and feet, include all other links
            exclude_link_names = set(self.foot_names)
            # exclude_link_names.add(self.base_name)
            knee_names = set(asset_rigid_body_dict.keys()) - exclude_link_names
        else:
            knee_names = get_matching_str(source=knee_name, destination=self.body_names, comment="knee_name")
        self.knee_ids = torch.tensor([asset_rigid_body_dict[n] for n in knee_names],
                                     dtype=torch.long, device=self.device)
        assert (self.knee_ids.numel() > 0)

        print(f"base = {self.base_name}: {self.base_id}")
        print(f"knee = {dict(zip(knee_names,self.knee_ids.tolist()))}")
        print(f"foot = {dict(zip(self.foot_names,self.foot_ids.tolist()))}")
        assert self.base_id != -1

        marker_pair_names = cfg_asset.get("marker_pair_names",[])
        self.num_marker_pairs = len(marker_pair_names)
        if len(marker_pair_names) > 0:
            self.marker_pair_l0 = torch.tensor(cfg_asset["marker_pair_length"], dtype=torch.float, device=self.device)
            self.last_marker_pair_length = self.marker_pair_l0.repeat(self.num_envs,1)

            self.marker_pair_names = [get_matching_str(
                source=marker_pair_name, destination=self.body_names, case_sensitive=True,comment="marker_pairs")
                for marker_pair_name in marker_pair_names]
            print("marker pair names:", self.marker_pair_names)
            self.marker_pair_ids = torch.tensor(
                [[asset_rigid_body_dict[n] for n in marker_pair] for marker_pair in self.marker_pair_names],
                dtype=torch.long, device=self.device)

    def _create_envs(self):
        """Creates multiple environments with randomized properties."""

        # randomize properties
        randomize = self.cfg["env"]["randomize"]
        # friction randomization
        randomize_friction:bool = randomize["friction"]["enable"]
        if randomize_friction:
            rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(self.asset)
            friction_buckets = torch.empty(self.num_envs, device=self.device,dtype=torch.float).uniform_(*randomize["friction"]["range"])
        # baseMass randomization
        self.randomize_base_mass:bool = randomize["baseMass"]["enable"]
        if self.randomize_base_mass:
            self.baseMass_buckets = torch.empty(self.num_envs, device=self.device,dtype=torch.float).uniform_(*randomize["baseMass"]["range"])
            # added_masses = np.random.uniform(*self.cfg["env"]["learn"]["addedMassRange"], self.num_envs)
        randomize_base_inertia_origin:bool = randomize["baseInertiaOrigin"]["enable"]
        if randomize_base_inertia_origin:
            origin_range = torch.tensor(randomize["baseInertiaOrigin"]["range"], device=self.device, dtype=torch.float)
            origin_buckets = torch_rand_tensor(lower=origin_range[:,0], upper=origin_range[:,1], shape=(self.num_envs, 3), device=self.device)
        # radomize link mass
        randomize_link_mass = randomize["link_mass"]["enable"]
        if randomize_link_mass:
            link_mass_range = randomize["link_mass"]["range"]
            link_mass_buckets = torch.empty(self.num_envs,self.num_bodies, device=self.device,dtype=torch.float).uniform_(*link_mass_range)
            link_inv_mass_buckets = 1.0 / link_mass_buckets
        # randomize link inertia
        randomize_link_inertia = randomize["link_inertia"]["enable"]
        if randomize_link_inertia:
            link_inertia_range = randomize["link_inertia"]["range"]
            link_inertia_buckets = torch.empty(self.num_envs,self.num_bodies, device=self.device,dtype=torch.float).uniform_(*link_inertia_range)
            # link_inertia_buckets = link_inertia_buckets.unsqueeze(-1).repeat_interleave(9, dim=-1)
            link_inv_inertia_buckets = 1.0 / link_inertia_buckets
            # self.link_mass+=torch.empty_like(self.link_mass).uniform_(*self.link_mass_range)
        # randomize base_pos
        self.randomize_base_init_pos = randomize["base_init_pos"]["enable"]
        self.randomize_base_init_pos_range = self.cfg["env"]["randomize"]["base_init_pos"]["range"]
        if not self.randomize_base_init_pos:
            self.randomize_base_init_pos_range = [[0,0],[0,0],[0,0]]

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        cfg_terrain = self.cfg["env"]["terrain"]
        cfg_terrain["minInitMapLevel"] = min(cfg_terrain["minInitMapLevel"], cfg_terrain["numLevels"])
        cfg_terrain["maxInitMapLevel"] = max(cfg_terrain["minInitMapLevel"], cfg_terrain["numLevels"])
        if not self.curriculum:
            cfg_terrain["maxInitMapLevel"] = cfg_terrain["numLevels"]
        self.terrain_levels = torch.randint(cfg_terrain["minInitMapLevel"], cfg_terrain["maxInitMapLevel"], (self.num_envs,), device=self.device)
        self.terrain_level_mean = self.terrain_levels.float().mean()
        self.heights_curriculum_started = False
        self.heights_curriculum_ratio = 0.001
        self.terrain_types = torch.randint(0, cfg_terrain["numTerrains"], (self.num_envs,), device=self.device)

        spacing = self.cfg["env"]['envSpacing']
        num_per_row = int(np.sqrt(self.num_envs))

        start_pose = gymapi.Transform()

        if self.custom_origins:
            spacing = 0.0
        # randomize dof_disable
        self.dof_color_disable = gymapi.Vec3(1.0, 0.0, 0.0)  # Red
        self.dof_color_enable = gymapi.Vec3(1.0, 1.0, 1.0)  # white
        self.should_dof_disable:bool = self.cfg["env"]["randomize"]["dof_disable"]["enable"]
        self.disable_dof_freeze = False # disable by freezing dof
        if self.should_dof_disable:
            self.dof_disable_prob_final:float = self.cfg["env"]["randomize"]["dof_disable"]["probability"]
            self.dof_disable_curriculum_enable:bool = self.cfg["env"]["randomize"]["dof_disable"]["curriculum"]["enable"]
            self.dof_disable_curriculum_num_step_inv:float = 1.0/self.cfg["env"]["randomize"]["dof_disable"]["curriculum"]["num_step"]
            self.dof_disable_prob:float = 0.0 if self.dof_disable_curriculum_enable else self.dof_disable_prob_final
            self.disable_dof_boolean = torch.zeros(self.num_envs,self.num_dof, device=self.device) < self.dof_disable_prob

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        self.actor_rigid_body_properties = []
        if self.camera_sensor_enable:
            self.cams = []
            self.cam_tensors = []
            cam_props = gymapi.CameraProperties()
            cam_props.horizontal_fov = self.cfg["env"]["camera_sensor"]['horizontal_fov']

            cam_props.width = self.cfg["env"]["camera_sensor"]["size"][0]
            cam_props.height = self.cfg["env"]["camera_sensor"]["size"][1]
            cam_props.enable_tensors = True

        if self.object_tracking_enabled or self.object_pushing_enabled:
            object_handles = []
            actor_handles = []
            self.cube_orientation = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain.env_origins[self.terrain_levels[i], self.terrain_types[i]]
            actor_handle = self.gym.create_actor(env_handle, self.asset, start_pose, "actor", i, self.collision_filter, 0)

            if self.object_tracking_enabled:
                self.cube_start_pose = gymapi.Transform()
                self.cube_pose = self.env_origins[i].clone() + torch.tensor([0.0, self.object_robot_initial_distance, 0.7], device=self.device)
                self.cube_start_pose.p = gymapi.Vec3(*self.cube_pose)
                cube_handle = self.gym.create_actor(env_handle, self.cube_asset, self.cube_start_pose, "cube", i, self.collision_filter, 0)
                cube_index = self.gym.get_actor_index(env_handle, cube_handle, gymapi.DOMAIN_SIM)
                actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
                actor_handles.append(actor_index)
                object_handles.append(cube_index)
            if self.object_pushing_enabled:
                self.cube_start_pose = gymapi.Transform()
                self.cube_pose = self.env_origins[i].clone() + torch.tensor([0, self.cfg['env']['objectPushing']['cube_y_position'], 0.55], device=self.device)
                self.cube_start_pose.p = gymapi.Vec3(*self.cube_pose)
                cube_handle = self.gym.create_actor(env_handle, self.cube_asset, self.cube_start_pose, "cube", i, 2, 0)
                cube_index = self.gym.get_actor_index(env_handle, cube_handle, gymapi.DOMAIN_SIM)
                actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
                actor_handles.append(actor_index)
                object_handles.append(cube_index)

            actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i==0:
                if randomize_link_mass:
                    mass = torch.tensor([p.mass for p in actor_rigid_body_props],device=self.device, dtype=torch.float)
                    inv_mass = torch.tensor([p.invMass for p in actor_rigid_body_props],device=self.device, dtype=torch.float)
                    link_masses = mass[None,:]*link_mass_buckets
                    inv_link_masses = 1.0/link_masses
                if randomize_link_inertia:
                    inertia = torch.tensor([[[p.inertia.x.x,p.inertia.x.y,p.inertia.x.z],
                                    [p.inertia.y.x,p.inertia.y.y,p.inertia.y.z],
                                    [p.inertia.z.x,p.inertia.z.y,p.inertia.z.z]] 
                        for p in actor_rigid_body_props],device=self.device, dtype=torch.float)
                    # shape = (num_envs, num_links, 3, 3)
                    link_inertias = inertia[None,:,:,:]*link_inertia_buckets[:,:,None,None]
                    
                    inv_inertia = torch.tensor([[[p.invInertia.x.x,p.invInertia.x.y,p.invInertia.x.z],
                                    [p.invInertia.y.x,p.invInertia.y.y,p.invInertia.y.z],
                                    [p.invInertia.z.x,p.invInertia.z.y,p.invInertia.z.z]] 
                        for p in actor_rigid_body_props],device=self.device, dtype=torch.float)
                    # shape = (num_envs, num_links, 3, 3)
                    link_inv_inertias =  inv_inertia[None,:,:,:]/link_inertia_buckets[:,:,None,None]

            # dof_force_tensor
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            actor_rigid_shape_prop = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            if randomize_friction:
                for s in range(len(actor_rigid_shape_prop)):
                    actor_rigid_shape_prop[s].friction = friction_buckets[i]
                self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, actor_rigid_shape_prop)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, self.dof_props)

            if randomize_link_mass:
                for b in range(self.num_bodies):
                    actor_rigid_body_props[b].mass=link_masses[i, b]
                    actor_rigid_body_props[b].invMass=inv_link_masses[i, b]
            if self.randomize_base_mass:
                actor_rigid_body_props[self.base_id].mass += self.baseMass_buckets[i]
                actor_rigid_body_props[self.base_id].invMass = 1.0 / actor_rigid_body_props[self.base_id].mass
            if randomize_base_inertia_origin:
                actor_rigid_body_props[self.base_id].com += gymapi.Vec3(*origin_buckets[i])
            if randomize_link_inertia:
                for b in range(self.num_bodies):
                    actor_rigid_body_props[b].inertia.x.x = link_inertias[i, b, 0, 0]
                    actor_rigid_body_props[b].inertia.x.y = link_inertias[i, b, 0, 1]
                    actor_rigid_body_props[b].inertia.x.z = link_inertias[i, b, 0, 2]
                    actor_rigid_body_props[b].inertia.y.x = link_inertias[i, b, 1, 0]
                    actor_rigid_body_props[b].inertia.y.y = link_inertias[i, b, 1, 1]
                    actor_rigid_body_props[b].inertia.y.z = link_inertias[i, b, 1, 2]
                    actor_rigid_body_props[b].inertia.z.x = link_inertias[i, b, 2, 0]
                    actor_rigid_body_props[b].inertia.z.y = link_inertias[i, b, 2, 1]
                    actor_rigid_body_props[b].inertia.z.z = link_inertias[i, b, 2, 2]
                    actor_rigid_body_props[b].invInertia.x.x = link_inv_inertias[i, b, 0, 0]
                    actor_rigid_body_props[b].invInertia.x.y = link_inv_inertias[i, b, 0, 1]
                    actor_rigid_body_props[b].invInertia.x.z = link_inv_inertias[i, b, 0, 2]
                    actor_rigid_body_props[b].invInertia.y.x = link_inv_inertias[i, b, 1, 0]
                    actor_rigid_body_props[b].invInertia.y.y = link_inv_inertias[i, b, 1, 1]
                    actor_rigid_body_props[b].invInertia.y.z = link_inv_inertias[i, b, 1, 2]
                    actor_rigid_body_props[b].invInertia.z.x = link_inv_inertias[i, b, 2, 0]
                    actor_rigid_body_props[b].invInertia.z.y = link_inv_inertias[i, b, 2, 1]
                    actor_rigid_body_props[b].invInertia.z.z = link_inv_inertias[i, b, 2, 2]
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, actor_rigid_body_props, recomputeInertia=self.override_inertia)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.actor_rigid_body_properties.append(actor_rigid_body_props)

            # camera settings
            if self.camera_sensor_enable:
                self.cams.append([])
                self.cam_tensors.append([])
                # joint_transformations = self.gym.get_actor_joint_transforms(env_handle, actor_handle)
                # rb_transformations = self.gym.get_vec_rigid_transform(env_handle, actor_handle)
                if self.camera_on_foot:
                    for body_handle in self.foot_ids:
                        cam_handle = self.gym.create_camera_sensor(env_handle, cam_props)
                        # local_transform = gymapi.Transform(p=gymapi.Vec3(0,0,0), r=joint_transformations[body_handle-1][1])
                        local_transform = gymapi.Transform(p=gymapi.Vec3(0,0,0), r=gymapi.Quat(0, -0.7071068, 0, 0.7071068)) # camera uses z axis
                        # local_transform = gymapi.Transform(p=gymapi.Vec3(0,0,0), r=gymapi.Quat(0, 0, 0, 1)) # camera uses x axis

                        self.gym.attach_camera_to_body(cam_handle, env_handle, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                        self.cams[i].append(cam_handle)
                        camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_COLOR)
                        torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
                        # torch_camera_tensor = torch.swapaxes(torch_camera_tensor, 2,0)[:3,:,:]
                        self.cam_tensors[i].append(torch_camera_tensor)
                else: # camera on base
                    camera_p = np.load('../assets/urdf/camera_p.npy', allow_pickle=True)
                    camera_r = np.load('../assets/urdf/camera_r.npy', allow_pickle=True)
                    for idx in range(len(camera_p)):
                        cam_handle = self.gym.create_camera_sensor(env_handle, cam_props)
                        local_transform = gymapi.Transform(p=gymapi.Vec3(camera_p[idx][0],camera_p[idx][1],camera_p[idx][2]),
                                                           r=gymapi.Quat(camera_r[idx][0], camera_r[idx][1], camera_r[idx][2], camera_r[idx][3]))
                        self.gym.attach_camera_to_body(cam_handle, env_handle, self.base_id, local_transform, gymapi.FOLLOW_TRANSFORM)
                        self.cams[i].append(cam_handle)
                        camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_COLOR)
                        torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
                        self.cam_tensors[i].append(torch_camera_tensor)
                # self.gym.end_access_image_tensors(self.sim)
        if self.camera_sensor_enable:
            self.cam_tensor_stack = torch.stack([torch.stack(inner_list) for inner_list in self.cam_tensors])[...,:3] # rgba->rgb
            self.grayscale_obs_image()

        self.actor_rigid_body_masses = torch.tensor(
            [[prop.mass for prop in props] for props in self.actor_rigid_body_properties], dtype=torch.float, device=self.device)

        if self.object_tracking_enabled or self.object_pushing_enabled:
            self.object_handles = torch.tensor(object_handles, device=self.device, dtype=torch.int32)
            self.actor_handles = torch.tensor(actor_handles, device=self.device, dtype=torch.int32)

    def check_termination(self):
        """Checks if the episode should terminate."""

        self.sim_step_count+=1
        if self.perception_asymmetry_experiment_collect:
            print('current data sim step:', self.sim_step_count)
            if self.sim_step_count > 4000:
                print("finished_data_collection ... exiting ...")
                sys.exit(0)

        if self.evaluate:
            if self.object_pushing_enabled:
                self.object_goal_vel_list_eval.append(self.object_goal_vel[:,:3].cpu().numpy())
                self.object_vel_list_eval.append(self.all_root_state[self.object_handles,7:10].cpu().numpy())
                self.robot_pos_list_eval.append(self.root_state[:, :3].cpu().numpy())
                self.object_pos_list_eval.append(self.all_root_state[self.object_handles,:3].cpu().numpy())
            if self.object_tracking_enabled:
                self.object_vel_list_eval.append(self.object_vel[:,:3].cpu().numpy())
                self.robot_vel_list_eval.append(self.base_lin_vel.cpu().numpy())
                self.robot_pos_list_eval.append(self.root_state[:, :3].cpu().numpy())   
                self.object_pos_list_eval.append(self.all_root_state[self.object_handles,:3].cpu().numpy())

            if self.shape_evaluate:
                self.dof_pos_list_eval.append(self.dof_pos[:,self.actuated_dof_mask].cpu().numpy())
                self.robot_orientation_list_eval.append(self.root_state[:,3:7].cpu().numpy())
            if self.COM_evaluation:
                self.robot_root_position_list_eval.append(self.root_state[:,:3].cpu().numpy())
                self.robot_orientation_list_eval.append(self.root_state[:,3:7].cpu().numpy())
                self.robot_linvel_list_eval.append(self.root_state[:, 7:10].cpu().numpy())
                self.robot_angularvel_list_eval.append(self.root_state[:, 10:13].cpu().numpy())

        self.reset_buf = square_sum(self.contact_force[:, self.base_id, :]) > 1.0

        if self.object_tracking_enabled and self.use_ray_obs and not self.evaluate:
            self.distance_reset = torch.norm(self.root_state[:, :3] - self.all_root_state[self.object_handles,:3], dim=-1)> self.cfg['env']['objectTracking']['distance_reset_threshold']
            self.distance_reset |= torch.norm(self.root_state[:, :3] - self.all_root_state[self.object_handles,:3], dim=-1) < 0.8
            self.reset_buf|=self.distance_reset
        if self.object_pushing_enabled and not self.evaluate:
            self.distance_reset = self.root_state[:, 1] - self.all_root_state[self.object_handles,1]> 0.0
            self.reset_buf|=self.distance_reset

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 0
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.timeout_buf = time_out & (self.reset_buf == 0)
        self.reset_buf[time_out] = 1
        # self.reset_buf = self.progress_buf >= self.max_episode_length - 1
    def compute_observations(self):
        """Computes observations for the current state."""

        if self.use_ray_obs:
            if self.perception_asymmetry_experiment:
                self.get_rays_asymmetric()
            else:
                self.get_rays()

        self.get_heights()
        heights = torch.clip(self.heights_relative - self.base_height_target, -1.0, 1.0) * self.heightmap_scale
        obs_dict = {
            "linearVelocity": self.base_lin_vel * self.lin_vel_scale,
            "worldSpaceAngularVelocity": self.world_space_base_ang_vel * self.ang_vel_scale,
            "angularVelocity": self.base_ang_vel * self.ang_vel_scale,
            "projectedGravity": self.projected_gravity,
            "projected_gravity_xy": self.projected_gravity[:,:2],
            "projected_gravity_filtered":self.projected_gravity_filtered,
            "base_rotation_matrix": self.base_rotation_matrix.view(self.num_envs, 9),
            "base_rotation_matrix_filtered": self.base_rotation_matrix_filtered.view(self.num_envs, 9),
            "commands": self.commands[:, :3] * self.commands_scale,
            "commands_xy": self.commands[:, :2] * self.commands_scale[:2],
            "dofPosition": self.dof_pos[:,self.actuated_dof_mask] * self.dof_pos_scale,
            # "dofVelocity": self.dof_vel[:,self.actuated_dof_mask] * self.dof_vel_scale, #TODO: FIXME. have to use the computed velocity instead
            "dofVelocity": self.dof_vel_computed[:,self.actuated_dof_mask] * self.dof_vel_scale, #TODO: FIXME. have to use the computed velocity instead
            "dof_force_target": self.actuated_dof_force_target * self.dof_force_target_scale,
            "actuated_dof_force_target_actual": self.actuated_dof_force_target_actual * self.dof_force_target_scale,
            # "dofForce": self.dof_force[:,self.actuated_dof_mask] * self.dof_force_scale,
            "dof_strength": self.dof_strength,
            "base_height": heights[:, self.num_height_points+self.base_id],
            "heightMap": heights[:, :self.num_height_points],
            "last_actions": self.last_action,
            "actions": self.action,
            "contact": self.foot_contact,
            "contactTarget": self.contact_target,
            "phase": self.phase_sin_cos,
            "dof_acc_computed": self.dof_acc_computed[:,self.actuated_dof_mask],
            "base_relative_angular_velocity_computed": self.base_relative_angular_velocity_computed,
            "robot_root_position": self.root_state[:, :3],
        }
        if self.object_tracking_enabled:
            obs_dict['object_velocity'] = self.object_vel[:,:3]
        if self.object_pushing_enabled:
            obs_dict['object_velocity'] = self.all_root_state[self.object_handles,7:10]
            self.object_orientation_history_buffer[:,:-4] = self.object_orientation_history_buffer[:,4:].clone()
            self.object_orientation_history_buffer[:,-4:] = self.all_root_state[self.object_handles,3:7].clone()
            obs_dict['object_orientation'] = self.object_orientation_history_buffer[:,:4] #the oldest object orientation to delay
            obs_dict['object_goal_velocity'] = self.object_goal_vel[:,:3]
            obs_dict['object_root_state'] = self.all_root_state[self.object_handles,:13]

        # update observation buffer
        obs_buf_single_frame = torch.cat(itemgetter(*self.obs_names)(obs_dict), dim=-1)
        if self.add_noise:
            self.noise_vec.uniform_(-1.0, 1.0).mul_(self.noise_scale_vec)  # scaled noise vector
            obs_buf_single_frame += self.noise_vec

        # delayed observations
        # self.batched_obs_buf.add_and_fill_batch(obs_buf_single_frame)
        self.batched_obs_buf.add(obs_buf_single_frame)
        # self.obs_buf = self.batched_obs_buf[self.max_observation_delay_steps].clone()
        self.obs_buf = torch.transpose(self.batched_obs_buf.get_latest_n(self.num_stacked_obs_frame,offset=self.max_observation_delay_steps).clone(),0,1).reshape(self.num_envs, self.num_obs)
        # self.obs_buf = self.batched_obs_buf[:self.num_stacked_obs_frame].clone().

        if self.use_obs2:
            obs2_buf_single_frame = torch.cat(itemgetter(*self.obs2_names)(obs_dict), dim=-1)
            # TODO add noise vector
            self.batched_obs2_buf.add(obs2_buf_single_frame)
            self.obs2_buf = torch.transpose(self.batched_obs2_buf.get_latest_n(self.num_stacked_obs_frame,offset=self.max_observation_delay_steps).clone(),0,1).reshape(self.num_envs, self.num_obs2)

        if self.asymmetric_obs: # update state buffer
            states_buf_single_frame = torch.cat(itemgetter(*self.state_names)(obs_dict), dim=-1)
            # self.batched_states_buf.add_and_fill_batch(states_buf)
            self.batched_states_buf.add(states_buf_single_frame)
            self.states_buf = torch.transpose(self.batched_states_buf.get_latest_n(self.num_stacked_state_frame).clone(),0,1).reshape(self.num_envs, self.num_states)

    def compute_reward(self):
        """Computes the reward for the current state and action."""
        rew = {}

        # velocity tracking reward
        desired_vel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        desired_vel[:, [0,1,5]] = self.commands[:, [0,1,2]]

        if self.object_tracking_enabled:
            desired_vel[:, :3] = self.object_vel[:,:3] # use the desired velocity from the object tracking
        # linear velocity error xyz
        self.lin_vel_error: torch.Tensor = desired_vel[:, :3] - self.base_lin_vel
        ang_vel_error: torch.Tensor = desired_vel[:, 3:] - self.base_ang_vel
        
        if self.object_pushing_enabled:
            self.lin_vel_error: torch.Tensor = self.object_goal_vel[:,:3] - self.all_root_state[self.object_handles,7:10]
            robot_object_distance_error = torch.norm(self.root_state[:, :3] - self.all_root_state[self.object_handles,:3],dim=1)
            rew["robot_object_distance"] =torch.exp(torch.square(robot_object_distance_error)*self.rew_robot_object_distance_exp_scale)
            rew["orientation_along_command_direction"] = compute_y_axis_velocity_alignment_reward(self.all_root_state[self.object_handles,3:7],self.object_goal_vel[:,:3])

        if self.cfg['env']['learn']['adaptive_linvel']:
            self.rew_lin_vel_exp_scale = 20*abs(desired_vel[:, :3]) -16
            self.rew_lin_vel_exp_scale = torch.clip(self.rew_lin_vel_exp_scale,min=-16,max=-4)

        rew["lin_vel"] = exp_weighted_square_sum(self.lin_vel_error, self.rew_lin_vel_exp_scale)
        rew["lin_vel_mse"] = 0.5 - torch.mean(torch.square(self.lin_vel_error),dim=-1)
        rew["ang_vel"] = exp_weighted_square_sum(ang_vel_error, self.rew_ang_vel_exp_scale)
        # orientation penalty
        rew["orientation"] = square_sum_clamp_max(self.projected_gravity[:, :2], max=0.1)
        base_height_error = self.base_height - self.base_height_target
        rew["base_height"] =torch.exp(torch.square(base_height_error)*self.rew_base_height_exp_scale)
        normalized_dof_force_target_out_of_bound = (self.actuated_dof_force_target - torch.clamp(self.actuated_dof_force_target, self.dof_force_target_soft_bound_min, self.dof_force_target_soft_bound_max))/self.dof_force_target_limit
        rew["dof_force_target"] = exp_square_sum(normalized_dof_force_target_out_of_bound, self.rew_dof_force_target_exp_scale)
        rew["dof_acc"] = exp_square_sum(self.dof_acc, self.rew_dof_acc_exp_scale)
        rew["dof_vel_computed"] = torch.square(self.dof_vel_computed).sum(dim=-1)
        rew["dof_acc_computed"] = torch.square(self.dof_acc_computed).sum(dim=-1)
        # joint vel penalty
        rew["dof_vel"] = exp_square_sum(self.dof_vel_computed, self.rew_dof_vel_exp_scale) # TODO FIXME
        actuated_dof_pos_delta = (self.dof_pos - self.desired_dof_pos)[:,self.actuated_dof_mask].abs()
        # joint position penalty
        rew["dof_pos"] = exp_square_sum(actuated_dof_pos_delta, self.rew_dof_pos_exp_scale)
        rew["dof_absolute_position"] = torch.mean(self.dof_pos[:,self.actuated_dof_mask]+0.105,dim=-1)
        rew["dof_pow"] = (self.dof_vel_computed[:,self.actuated_dof_mask] * self.actuated_dof_force_target).abs().mean(dim=1)*self.total_gravity_inv # TODO FIXME
        # penalty for position exceeding dof limit
        rew["dof_limit"] = out_of_bound_square_sum(self.dof_pos, self.dof_limit_lower, self.dof_limit_upper)
        # collision penalty
        knee_collision = square_sum(self.contact_force[:, self.knee_ids, :], dim=2) > 1.0
        rew["collision"] = torch.sum(knee_collision, dim=1, dtype=torch.float)  # sum vs any ?
        # foot impact penalty (num_envs,num_foot,3)
        rew["impact"] = torch.clamp(self.foot_contact_force[:, :, 2]*self.total_gravity_inv.view(self.num_envs,1)-1,min=0,max=2).square().sum(dim=1)
        # foot slip penalty
        rew["slip"] = (self.foot_lin_vel.square().sum(dim=2) * self.foot_contact_filt).sum(dim=1)
        rew["action"] = out_of_bound_exp_square_sum(self.action,lower=-0.5, upper=0.5,exp_scale=self.rew_action_exp_scale, dim=-1)
        # action rate penalty
        self.action_rate = (self.action - self.last_action) * self.rl_dt_inv
        rew["action_rate"] = exp_square_sum(self.action_rate, self.rew_action_rate_exp_scale)
        # penalize high contact forces
        contact_force_norm = torch.norm(self.contact_force[:, self.foot_ids, :], dim=-1)
        rew["contact_force"] = torch.sum((contact_force_norm - self.max_foot_contact_force).clip(min=0.0), dim=1)

        # log episode reward sums
        for key in rew.keys():
            self.episode_sums[key] += rew[key]  # unscaled
            rew[key] *= self.rew_scales[key]

        # total reward
        stacked_rewards = torch.stack(list(rew.values()), dim=0)
        self.rew_buf = torch.sum(stacked_rewards, dim=0)
        if self.cfg['env']['learn']["no_negative_reward"]:
            self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None) # NOTE THAT WE SCALE IT BY rl_dt

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        if self.compute_cost_of_transport_metrics:
            # dof_pow = (self.dof_vel[:,self.actuated_dof_mask] * self.actuated_dof_force_target) # TODO FIXME
            dof_pow = (self.dof_vel_computed[:,self.actuated_dof_mask] * self.actuated_dof_force_target) # TODO FIXME
            dof_pow_sum = dof_pow.clamp_min(0).sum(dim=1)
            self.buffer_com_pos.add_and_fill_batch(self.root_state[:, :3])
            self.buffer_dof_pow.add_and_fill_batch(dof_pow_sum.unsqueeze(1))
            self.total_energy = torch.sum(self.buffer_dof_pow.storage,dim=0).view(-1) * self.rl_dt
            self.distance_traveled = (self.buffer_com_pos[0] - self.buffer_com_pos[-1])[:,:2].norm(dim=-1)
            self.cost_of_transport = self.total_energy/(torch.clamp_min(self.distance_traveled, 0.01)*self.total_gravity)


        if self.enable_data_publisher:  # send UDP info to plotjuggler
            data = {
                "t": self.control_steps*self.rl_dt,
                # "step_height_target": step_height_target,
                "cot": self.cost_of_transport,
                "action": self.action,
                "action_to_use": self.action_to_use,
                "action_is_on": self.action_is_on,
                "action_is_on_rate": self.action_is_on_rate,
                "action_rate": self.action_rate,
                # "dof_jerk": self.dof_jerk[:,self.actuated_dof_mask],
                "dof_acc": self.dof_acc[:,self.actuated_dof_mask],
                "dof_vel": self.dof_vel[:,self.actuated_dof_mask],
                "dof_vel_computed": self.dof_vel_computed[:,self.actuated_dof_mask],
                "dof_vel_computed_all_substeps": self.dof_vel_computed_all_substeps,
                "dof_acc_computed":self.dof_acc_computed[:,self.actuated_dof_mask],
                "dof_pos": self.dof_pos[:,self.actuated_dof_mask],
                "dof_pos_target": self.dof_pos_target,
                "dof_force_target": self.actuated_dof_force_target,
                "dof_force": self.dof_force[:,self.actuated_dof_mask],
                "dof_max_linear_force": self.dof_max_linear_force_to_accelerate,
                "dof_pow": dof_pow,
                "base_lin_vel": self.base_lin_vel,
                "base_ang_vel": self.base_ang_vel,
                "base_height": self.base_height,
                # "foot_height": step_height,
                "projected_gravity": self.projected_gravity,
                "time_air": self.air_time,
                "time_stance": self.stance_time,
                "foot_pos": self.foot_pos_rel_yaw,
                "contact": self.foot_contact_filt,
                "phase": self.phase,
                "contact_target":self.contact_target,
                "rew_buf": self.rew_buf * self.rl_dt_inv,
                "commands": self.commands,
                "rew": {key: rew[key] * self.rl_dt_inv for key in rew},
                "rew_rel":{key: rew[key]/self.rew_buf for key in rew},
                "foot_rb_state": self.rb_state[:,self.foot_ids],
                # "foot_quat": self.foot_quat,
                "base_quat": self.base_quat,
                "base_quat_filtered": self.base_quat_filtered,
                "base_rotation_matrix": self.base_rotation_matrix,
                "base_rotation_matrix_filtered": self.base_rotation_matrix_filtered,
                "root_state": self.root_state,
                "base_forward": self.base_forward,
                "foot_forward": self.foot_forward,
                "projected_gravity_filtered": self.projected_gravity_filtered,
                # "obs":self.obs_buf,
                # "cam_pos": self.cam_pos,
                "base_relative_angular_velocity_computed": self.base_relative_angular_velocity_computed,

            }

            if self.items_to_publish is not None:
                data = {key: data[key] for key in self.items_to_publish}
            self.data_publisher.publish({self.data_root_label:data})

            if self.enable_sim2real_data_publisher:
                self.sim2real_publisher.publish({
                    # "base_ang_vel": self.base_ang_vel[0],
                    # "projected_gravity": self.projected_gravity[0],
                    "dof_pos": self.dof_pos[0,self.actuated_dof_mask],
                    # "dof_vel": self.dof_vel[0,self.actuated_dof_mask],
                    # "action": self.action[0],
                    # "contact": self.foot_contact_filt[0],
                    # "contact_target":self.contact_target[0],
                    # "dof_pos_target": self.dof_pos_target[0],
                    # "cmd": self.commands[0][:3],
                    # "obs_buf":self.obs_buf[0],
                })


    def grayscale_obs_image(self):
        # num_envs, num_cameras, height, width, 3
        r, g, b = self.cam_tensor_stack.type(torch.float32).unbind(dim=-1)
        self.obs_image_grayscale = (0.2989 * r + 0.587 * g + 0.114 * b)/255

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.update_obs_dict()
        return self.obs_dict

    def update_obs_dict(self):
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.use_obs2:
            self.obs_dict["obs2"] = torch.clamp(self.obs2_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        if self.camera_sensor_enable:
           self.obs_dict["obs_image"] = self.obs_image_grayscale.to(self.rl_device)

        if self.num_states > 0: # asymmetric actor-critic
            self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        if self.use_ray_obs:
            self.obs_dict["ray_point_cloud"] = torch.clamp(self.point_cloud_history_buffer[:,:self.num_points*self.point_history_length], -self.clip_obs, self.clip_obs).to(self.rl_device) #point cloud is the oldest 3 step
            self.obs_dict["ray_distance"] = torch.clamp(self.distance_history_buffer, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["joint_origins"] = torch.clamp(self.joint_origins, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["joint_directions"] = torch.clamp(self.joint_directions, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["joint_quaternions"] = torch.clamp(self.joint_quaternions, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["robot_root_position"] = torch.clamp(self.robot_root_position, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["robot_root_quaternions"] = torch.clamp(self.robot_root_quaternions, -self.clip_obs, self.clip_obs).to(self.rl_device)


        if self.object_tracking_enabled:
            self.obs_dict["object_velocity"] = torch.clamp(self.object_vel[:,:3], -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.object_pushing_enabled:
            self.obs_dict["object_velocity"] = torch.clamp(self.all_root_state[self.object_handles,7:10], -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict['object_orientation'] = torch.clamp(self.all_root_state[self.object_handles,3:7], -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict['contact'] = torch.clamp(self.foot_contact, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def reset_idx(self, env_ids: torch.Tensor):
        """Resets the specified environments."""
        len_ids = env_ids.numel()
        if len_ids == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_raw = gymtorch.unwrap_tensor(env_ids_int32)
        # before reset collect some statistics
        if self.evaluate:
            if self.init_done:
                data = {
                    "command_xy": self.commands[env_ids, :2].cpu().numpy(),
                    "lin_vel_error": self.lin_vel_error[env_ids].norm(dim=1, keepdim=True).cpu().numpy(),
                    "cot_cost_of_tansport": self.cost_of_transport[env_ids].cpu().numpy(),
                    "cot_total_energy": self.total_energy[env_ids].cpu().numpy(),
                    "cot_distance_traveled": self.distance_traveled[env_ids].cpu().numpy(),
                    "dof_pos": np.array(self.dof_pos_list_eval),
                    "robot_orientation": np.array(self.robot_orientation_list_eval),
                }
                if self.should_dof_disable:
                    data["dof_disable"] = self.disable_dof_boolean[env_ids].sum(dim=1).cpu().numpy()
                if self.randomize_base_mass:
                    data["base_mass_bucket"] = self.baseMass_buckets[env_ids].cpu().numpy()
                if self.object_pushing_enabled:
                    data['object_goal_velocity'] = np.array(self.object_goal_vel_list_eval)
                    data['object_velocity'] = np.array(self.object_vel_list_eval)
                    data['robot_pos'] = np.array(self.robot_pos_list_eval)
                    data['object_pos'] = np.array(self.object_pos_list_eval)
                if self.object_tracking_enabled:
                    data['robot_vel'] = np.array(self.robot_vel_list_eval)
                    data['object_velocity'] = np.array(self.object_vel_list_eval)
                    data['robot_pos'] = np.array(self.robot_pos_list_eval)
                    data['object_pos'] = np.array(self.object_pos_list_eval)
                if self.shape_evaluate:
                    data['dof_pos'] = np.array(self.dof_pos_list_eval)
                    data['robot_orientation'] = np.array(self.robot_orientation_list_eval)
                if self.COM_evaluation:
                    data['robot_root_position'] = np.array(self.robot_root_position_list_eval)
                    data['robot_orientation'] = np.array(self.robot_orientation_list_eval)
                    data['robot_linear_velocity'] = np.array(self.robot_linvel_list_eval)
                    data['robot_angular_velocity'] = np.array(self.robot_angularvel_list_eval)
                

                distance_xyz = self.root_state[env_ids, :2] - self.env_origins[env_ids, :2]
                command_distance_xyz = self.commands[env_ids, :2] * self.max_episode_length_s
                distance_error_xyz = distance_xyz - command_distance_xyz
                distance_error = torch.norm(distance_error_xyz, dim=1)
                distance = torch.norm(distance_xyz, dim=1)
                command_distance = torch.norm(command_distance_xyz,dim=-1)
                data["distance"] = distance.cpu().numpy()
                data["command_distance"] = command_distance.cpu().numpy()
                data["distance_error"] = distance_error.cpu().numpy()
                if self.custom_origins:
                    data["terrain_level"] = self.terrain_levels[env_ids].cpu().numpy()
                if self.push_enable_at_reset:
                    data["push_vel"] = self.push_vel[env_ids].cpu().numpy()

                data["base_touch_down_pos"] = self.base_pos_at_first_contact[env_ids].cpu().numpy()
                data["base_pos"] = self.base_pos[env_ids].cpu().numpy()
                # reset the base touch down pos
                self.base_pos_at_first_contact[env_ids] = 0
                self.has_first_contact[env_ids] = 0

                info = orjson.dumps(data,option=self.orjson_option)
                with open(self.evaluate_json_name, "ab") as json_file:  # 'w' for write mode
                    json_file.write(info)
                    print(f"evaluate json file updated: {self.evaluate_json_name}\n")
                self.object_goal_vel_list_eval = []
                self.object_vel_list_eval = []
                self.robot_vel_list_eval = []
                self.robot_pos_list_eval = []
                self.object_pos_list_eval = []
                self.dof_pos_list_eval = []
                self.robot_orientation_list_eval = []  
                self.robot_root_position_list_eval = []
                self.robot_linvel_list_eval= []
                self.robot_angularvel_list_eval= []
                if self.perception_asymmetry_experiment and self.sim_step_count > 800:
                    print('finished asymmetry eval... exiting ...')
                    sys.exit(0)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.base_init_state[env_ids] = self.base_init_state_default
            self.base_init_state[env_ids, :3] += self.env_origins[env_ids]
        else:
            self.base_init_state[env_ids] = self.base_init_state_default

        if self.randomize_base_init_pos:
            for k in range(3):
                self.base_init_state[env_ids, k] += torch.empty(len_ids, device=self.device, dtype=torch.float).uniform_(*self.randomize_base_init_pos_range[k])

        # random orientation
        if self.randmoize_base_init_orientation:
            self.base_init_state[env_ids, 3:7] = random_quaternion(len_ids, device=self.device)
            # # HACK TODO CHANGE BACK
            # self.base_init_state[env_ids, 3:7] = random_quaternion(1, device=self.device)

        if self.push_enable_at_reset:
            push_vel = torch_rand_tensor(
                self.push_vel_min, self.push_vel_max, (len_ids, 6), device=self.device
            )
            # # HACK TODO CHANGE BACK
            # push_vel = torch_rand_tensor(self.push_vel_min, self.push_vel_max, (1, 6), device=self.device)
            self.push_vel[env_ids] = push_vel
            self.base_init_state[env_ids, 7:13]+= push_vel

        # update root state
        self.root_state[env_ids] = self.base_init_state[env_ids]

        if self.randomize_init_dof_pos:
            # dof_pos_offset = torch_rand_float(*self.randomize_init_dof_pos_range, (len_ids, self.num_dof), self.device)
            dof_pos_offset = torch.empty(len_ids, self.num_dof,dtype=torch.float, device=self.device).uniform_(*self.randomize_init_dof_pos_range)
            self.dof_pos[env_ids] = self.init_dof_pos[env_ids] + dof_pos_offset
        else:
            self.dof_pos[env_ids] = self.init_dof_pos[env_ids]

        if self.randomize_init_dof_vel:
            # self.dof_vel[env_ids] = torch_rand_float(*self.randomize_init_dof_vel_range, (len_ids, self.num_dof), self.device)
            self.dof_vel[env_ids] = torch.empty(len_ids, self.num_dof,dtype=torch.float, device=self.device).uniform_(*self.randomize_init_dof_vel_range)
        else:
            self.dof_vel[env_ids,:] = 0

        if self.object_tracking_enabled:
            object_indices = self.object_handles[env_ids]
            actor_indices = self.actor_handles[env_ids]
            reset_indices =torch.cat((actor_indices,object_indices))
            self.y_tracking_position[env_ids] =torch.zeros(len_ids,
                                                         dtype=torch.float, device=self.device)
            self.x_tracking_position[env_ids] = torch.zeros(len_ids,
                                                            dtype=torch.float, device=self.device)

            self.all_root_state[object_indices,:3] = self.cube_pose.repeat(len_ids, 1).to(self.device)
            self.all_root_state[object_indices,3:7] = torch.tensor([0,0,0,1],dtype=torch.float).repeat(len_ids, 1).to(self.device)
            self.all_root_state[object_indices,7:10] = torch.tensor([0,0,0],dtype=torch.float).repeat(len_ids, 1).to(self.device)
            self.all_root_state[object_indices,10:13] = torch.tensor([0,0,0],dtype=torch.float).repeat(len_ids, 1).to(self.device)

            #random_direction
            self.random_x_displacement[env_ids], self.random_y_displacement[env_ids] = sample_points_with_norm_constraint(self.displacement_range[0], self.displacement_range[1], len_ids,self.device)

            yaw_angle = torch.atan2(self.random_y_displacement[env_ids], self.random_x_displacement[env_ids])
            if self.cfg['env']['ray_obs']['random_vel']:
                yaw_angle = torch.rand(len_ids, device=self.device) * 2 * torch.pi - torch.pi
            # # Compute quaternion for yaw rotation
            cos_half_yaw = torch.cos(yaw_angle / 2)
            sin_half_yaw = torch.sin(yaw_angle / 2)
            self.new_orientation = torch.stack([torch.zeros(len_ids).to(self.device),  # x
                                        torch.zeros(len_ids).to(self.device),  # y
                                        sin_half_yaw,                         # z (yaw rotation)
                                        cos_half_yaw], dim=1)                 # w
            self.cube_orientation[env_ids] = self.new_orientation
            # # Compute the norm (magnitude) of the current random displacement
            displacement_norm = torch.sqrt(self.random_x_displacement[env_ids]**2 + self.random_y_displacement[env_ids]**2)

            # # Compute the scaling factor to make the displacements match the amplitude of the base position
            self.scaling_factor = self.object_robot_initial_distance / displacement_norm


            # Assign new position and orientation
            self.all_root_state[object_indices, 0] = self.random_x_displacement[env_ids]*self.scaling_factor
            self.all_root_state[object_indices, 1] = self.random_y_displacement[env_ids]*self.scaling_factor
            self.all_root_state[object_indices, 3:7] = self.new_orientation.to(self.device)

            if self.cfg['env']["ray_obs"]['static_debug']:
                self.random_x_displacement[env_ids], self.random_y_displacement[env_ids] = 1.5,0.5
                self.all_root_state[object_indices, 0] = self.random_x_displacement[env_ids]
                self.all_root_state[object_indices, 1] = self.random_y_displacement[env_ids]
                self.all_root_state[object_indices, 3:7] = torch.tensor([0,0,0,1],dtype=torch.float).repeat(len_ids, 1).to(self.device)


            if self.use_ray_obs:
                self.point_cloud_history_buffer[env_ids, :] = torch.full_like(self.point_cloud_history_buffer[env_ids, :,:], 0)
                self.distance_history_buffer[env_ids, :] = torch.full_like(self.distance_history_buffer[env_ids, :], 1)
            self.object_vel[:, 0] = self.random_x_displacement/self.rl_dt
            self.object_vel[:, 1] = self.random_y_displacement/self.rl_dt

            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.all_root_state),gymtorch.unwrap_tensor(reset_indices)  , len(reset_indices))
            self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_raw, gymtorch.unwrap_tensor(actor_indices), len_ids)
        elif self.object_pushing_enabled:
            object_indices = self.object_handles[env_ids]
            actor_indices = self.actor_handles[env_ids]
            reset_indices = torch.cat((actor_indices, object_indices))
            random_angle = torch.rand(len(env_ids), device='cuda') * torch.pi/2.25 - torch.pi/4.5
            self.object_goal_vel[env_ids, 0] = self.cfg['env']['objectPushing']['goal_vel'] * torch.sin(random_angle)
            self.object_goal_vel[env_ids, 1] = self.cfg['env']['objectPushing']['goal_vel'] * torch.cos(random_angle)
            self.all_root_state[object_indices, :3] = self.cube_pose.repeat(len_ids, 1).to(self.device)
            self.all_root_state[object_indices,3:7] = torch.tensor([0,0,0,1],dtype=torch.float).repeat(len_ids, 1).to(self.device)
            self.all_root_state[object_indices,7:13] = 0
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.all_root_state),gymtorch.unwrap_tensor(reset_indices)  , len(reset_indices))
            self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_raw, gymtorch.unwrap_tensor(actor_indices), len_ids)
            if self.use_ray_obs:
                self.point_cloud_history_buffer[env_ids, :] = torch.full_like(self.point_cloud_history_buffer[env_ids, :,:], 0)
                self.distance_history_buffer[env_ids, :] = torch.full_like(self.distance_history_buffer[env_ids, :], 1)
        else:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_state_raw, env_ids_raw, len_ids)
            self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_raw, env_ids_raw, len_ids)

        temp_vec = torch.empty(len_ids, device=self.device, dtype=torch.float)
        if self.cfg['env']['learn']['lin_vel_curriculum']:
            if self.control_steps > 5000:
                self.lin_vel_curriculum += 0.01
                if self.lin_vel_curriculum > 1:
                    self.lin_vel_curriculum = 1
                self.control_steps = 0
            scaled_range = self.lin_vel_curriculum * np.array(self.command_x_range)
            self.commands[env_ids, 0] = temp_vec.uniform_(*scaled_range)

            scaled_range = self.lin_vel_curriculum * np.array(self.command_y_range)
            self.commands[env_ids, 1]= temp_vec.uniform_(*scaled_range)
        else:
            # vx
            self.commands[env_ids, 0]= temp_vec.uniform_(*self.command_x_range)
            # vy
            self.commands[env_ids, 1]= temp_vec.uniform_(*self.command_y_range)

        # SET COMMANDS TO ZERO FOR A PAERCENTAGE OF ENVIRONMENTS
        if self.command_zero_probability:
            self.commands[env_ids[temp_vec.uniform_()<self.command_zero_probability], :] = 0 # the first 10% of the envs command will be zero

        # set small commands to zero
        self.is_zero_command[env_ids] = square_sum(self.commands[env_ids, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        if self.enable_erfi:
            self.erfi_rao[env_ids] = torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)

        if self.randomize_dof_strength:
            self.dof_strength[env_ids] = torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.dof_strength_range)

        if self.enable_passive_dynamics:
            self.duration_since_action_switch[env_ids]=0
            self.last_action_is_on[env_ids] = 1 # fully active at first

        if self.randomize_control_pd:
            self.kp[env_ids] = self.kp_default[env_ids] *(torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.control_kp_range))
            self.kd[env_ids] = self.kd_default[env_ids] *(torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.control_kd_range))


        # # TODO: reset some observations
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.base_gravity_vec[env_ids])
        self.projected_gravity_filtered[env_ids]=self.projected_gravity[env_ids]

        self.base_quat_filtered[env_ids] = self.base_quat[env_ids]
        self.base_rotation_matrix[env_ids] = torch.eye(3, device=self.device) # TODO maybe remove repeat

        if self.randomize_body_force:
            # reset rigid body forces
            self.rb_forces[env_ids, :, :] = 0.0
            self.random_force_prob[env_ids] = torch.empty(len_ids,dtype=torch.float, device=self.device).uniform_(*self.force_log_prob_range).exp_()

        self.last_foot_contact[env_ids] = 0
        self.foot_multi_contact_time[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.action_filt[env_ids] = 0.0
        # self.dof_pos_filt[env_ids] = self.dof_pos[env_ids]
        self.last_foot_contact_force[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_dof_acc[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.stance_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.dof_pos_error[env_ids] = 0.0
        self.dof_pos_error_integral[env_ids] = 0.0

        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_dof_pos_substep[env_ids] = self.dof_pos[env_ids]
        self.last_dof_vel_computed[env_ids] = 0
        self.last_base_quat[env_ids] = self.root_state[env_ids,3:7]

        # reset delayed observations
        self.batched_obs_buf.reset_batch(env_ids)
        if self.asymmetric_obs:
            self.batched_states_buf.reset_batch(env_ids)

        # #reset action buffer
        # self.batched_action_buf.reset_batch(env_ids)

        # if self.enable_udp:
        #     # strictly this should be reset, it's ok to ommit it for steady state values
        #     self.buffer_com_pos.reset_batch(env_ids)
        #     self.buffer_dof_pow.reset_batch(env_ids)

        if self.should_dof_disable:
            if self.dof_disable_curriculum_enable:
                self.dof_disable_prob = min(self.common_step_counter*self.dof_disable_curriculum_num_step_inv,1.0)*self.dof_disable_prob_final
                print(f"dof_disable_prob: {self.dof_disable_prob}")
            self.disable_dof_boolean[env_ids] = torch.rand(len_ids,self.num_dof, device=self.device) < self.dof_disable_prob
            # # # HACK TODO CHANGE BACK
            # # self.disable_dof_boolean[:]= False
            # # self.disable_dof_boolean[env_ids,3]=True
            # # self.disable_dof_boolean[env_ids,6]=True
            # # self.disable_dof_boolean[env_ids,13]=True

            # self.disable_dof_boolean[:]= False
            # self.disable_dof_boolean[env_ids,0]=True
            # self.disable_dof_boolean[env_ids,8]=True
            # self.disable_dof_boolean[env_ids,11]=True

            # self.disable_dof_boolean[:]= False
            # self.disable_dof_boolean[env_ids,1]=True
            # self.disable_dof_boolean[env_ids,4]=True
            # # self.disable_dof_boolean[env_ids,16]=True

            if (not self.headless) and self.enable_viewer_sync: # should reset graphics as well
                disable_dof_boolean_reset = self.disable_dof_boolean[env_ids]
                for env_idx, dof_idx in torch.nonzero(~disable_dof_boolean_reset):
                    self.gym.set_rigid_body_color(self.envs[env_idx], self.actor_handles[env_idx], dof_idx+1, gymapi.MESH_VISUAL, self.dof_color_enable)
                for env_idx, dof_idx in torch.nonzero(disable_dof_boolean_reset):
                    # HACK for Argus bot only, should use the child of the dof indtead
                    self.gym.set_rigid_body_color(self.envs[env_idx], self.actor_handles[env_idx], dof_idx+1, gymapi.MESH_VISUAL, self.dof_color_disable)

        # for env_idx in range(self.num_envs//2): # HACK CHANGE BACK
        #     for dof_idx in range(self.num_dof+1):
        #         self.gym.set_rigid_body_color(self.envs[env_idx], self.actor_handles[env_idx], dof_idx, gymapi.MESH_VISUAL, gymapi.Vec3(0.255, 0.569, 0.929))

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            raw_sum = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # rewards per second
            self.extras["episode"][f'rew_{key}_raw'] = raw_sum * self.rl_dt  # scaled by policy dt
            self.extras["episode"][f'rew_{key}'] = raw_sum * self.rew_scales[key]
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = self.terrain_level_mean
        self.extras["episode"]["heights_curriculum_ratio"] = self.heights_curriculum_ratio

    def update_terrain_level(self, env_ids):
        """Updates the terrain level for curriculum learning."""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        # distance = torch.norm(self.root_state[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        displacement = self.root_state[env_ids, :2] - self.base_init_state[env_ids, :2] # measured in xy plane
        displacement_norm = torch.norm(displacement, dim=1)
        command_displacement = self.commands[env_ids, :2] * self.max_episode_length_s # measured in xy plane
        command_displacement_norm = torch.norm(command_displacement, dim=1)

        similarity = torch.einsum('bm,bm->b', displacement, command_displacement)/(displacement_norm*command_displacement_norm+1e-6)
        nonzero_command = command_displacement_norm >= self.command_zero_threshold_distance

        is_following_command = nonzero_command & (similarity>0.707) & (displacement_norm>command_displacement_norm*0.7)
        is_not_following_command = nonzero_command & ((similarity<0.5) | (displacement_norm>command_displacement_norm*0.3))

        self.terrain_levels[env_ids] -= (is_not_following_command & (~self.timeout_buf[env_ids])).int()
        self.terrain_levels[env_ids] += (is_following_command | (displacement_norm > self.terrain.env_length / 2)).int()

        # distance = torch.norm(displacement, dim=1)
        # command_distance = torch.norm(self.commands[env_ids, :2],dim=-1) * self.max_episode_length_s
        # # # not timed out
        # # self.terrain_levels[env_ids] -= 1 * ((distance < command_distance * 0.25) & (~self.timeout_buf[env_ids]))
        # self.terrain_levels[env_ids] -= 1 * ((distance < (command_distance * 0.5)) & (~self.timeout_buf[env_ids]))
        # # TODO check level up/down condition
        # self.terrain_levels[env_ids] += (distance > self.terrain.env_length / 2).int()

        # self.terrain_levels[env_ids] += 1 * torch.logical_or(
        #     distance > self.terrain.env_length / 2, distance > (command_distance * 0.9))

        # # if reached max level, go back to level 0
        # self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows

        # if reached max level, go to a random level
        self.terrain_levels[env_ids] = self.terrain_levels[env_ids].clip(0)
        reached_max_levels = env_ids[self.terrain_levels[env_ids]>=self.terrain.env_rows]
        self.terrain_levels[reached_max_levels]=torch.randint_like(self.terrain_levels[reached_max_levels],low=0,high=self.terrain.env_rows)

        self.env_origins[env_ids] = self.terrain.env_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.terrain_level_mean = self.terrain_levels.float().mean()

    def push_robot_base(self):
        """Applies random pushes to the robots."""
        self.root_state[:, 7:13]+= torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (self.num_envs, 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_raw)

    def push_robot_base_indexed(self, env_ids: torch.Tensor):
        """Applies random pushes to the robots."""
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.root_state[env_ids, 7:13]+=torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (len(env_ids), 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_state_raw, gymtorch.unwrap_tensor(env_ids.type(torch.int32)), env_ids.numel())


    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if self.object_pushing_enabled:
                start_pos = self.all_root_state[self.object_handles[0], :3].cpu().numpy().astype(np.float32)
                
                # Scale the velocity vector for better visualization
                velocity_scale = 8.0  # Adjust this to make arrows longer/shorter
                end_pos = start_pos + (self.object_goal_vel[0].cpu().numpy().astype(np.float32) * velocity_scale)
                
                # Create line vertices array (shape: [2, 3])
                line_vertices = np.array([start_pos, end_pos], dtype=np.float32)
                
                # Create color array (yellow line)
                line_colors = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
                
                # Draw the line
                self.gym.add_lines(self.viewer, None, 1, line_vertices, line_colors)
                # colors_vel_dir = gymapi.Vec3(1.0, 1.0, 0.0)  # Yellow color

                # # Convert directly in the function call
                # start_tensor = self.all_root_state[self.object_handles[0], :3].cpu()
                # end_tensor = (self.all_root_state[self.object_handles[0], :3] + self.object_goal_vel[0]).cpu()

                # gymutil.draw_line(
                #     gymapi.Vec3(start_tensor[0].item(), start_tensor[1].item(), start_tensor[2].item()),
                #     gymapi.Vec3(end_tensor[0].item(), end_tensor[1].item(), end_tensor[2].item()),
                #     colors_vel_dir,
                #     self.gym,
                #     self.viewer,
                #     0
                # )
            # check for keyboard events
            events = self.gym.query_viewer_action_events(self.viewer)
            for evt in events:
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames
                elif evt.action == "toggle_viewer_follow" and evt.value > 0:
                    self.viewer_follow = not self.viewer_follow
                elif evt.action == "reset" and evt.value>0:
                    # self.data_publisher.enable= not self.data_publisher.enable
                    # reset
                    self.progress_buf[:]= self.max_episode_length
                elif evt.action == "ref_env-" and evt.value > 0:
                    self.ref_env = (self.ref_env-1)%self.num_envs
                elif evt.action == "ref_env+" and evt.value > 0:
                    self.ref_env = (self.ref_env+1)%self.num_envs
            if self.enable_keyboard_operator:
                for evt in events:
                    if evt.action == "vx+" and evt.value > 0:
                        self.keyboard_operator_cmd[0] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vx-" and evt.value > 0:
                        self.keyboard_operator_cmd[0] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy+" and evt.value > 0:
                        self.keyboard_operator_cmd[1] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy-" and evt.value > 0:
                        self.keyboard_operator_cmd[1] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading+" and evt.value > 0:
                        self.keyboard_operator_cmd[2] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading-" and evt.value > 0:
                        self.keyboard_operator_cmd[2] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "v=0" and evt.value > 0:
                        self.keyboard_operator_cmd[:] = 0
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "push" and evt.value > 0: # TODO fix CPU crashing in push_robots here
                        self.push_robot_base_indexed(torch.arange(self.num_envs, device=self.device))
                    # elif evt.action == "m+" and evt.value > 0:
                        # for i in range(self.num_envs):
                        #     env_handle = self.envs[i]
                        #     actor_handle = self.actor_handles[i]
                            # body_props = self.actor_rigid_body_properties[i]
                            # body_props[self.base_id].mass += 0.05
                            # self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=False)
                        # force = torch.zeros(self.num_envs, 3, device=self.device)
                        # self.gym.apply_rigid_body_force_tensors(self.sim,, gymapi.ENV_SPACE)
                        # print("mass increased by 0.05 kg")
                # self.commands[:, [0, 1, 3]] = self.keyboard_operator_cmd
                self.commands[:, :3] = self.keyboard_operator_cmd

            # fetch results
            if self.should_fetch_result:
                self.gym.fetch_results(self.sim, True)
                self.should_fetch_result = False

            # step graphics
            if self.enable_viewer_sync:
                self.draw_debug_lines()
                if self.should_step_graphics:
                    self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(
                    self.viewer, os.path.join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

            # do modify camera position if viewer_follow
            if self.viewer_follow:
                self.cam_target_pos = self.root_state[self.ref_env, :3].clone()
                self.cam_pos = self.viewer_follow_offset+self.cam_target_pos
                # self.gym.viewer_camera_look_at(
                #     self.viewer, self.envs[self.ref_env], gymapi.Vec3(*self.cam_pos.cpu()), gymapi.Vec3(*self.cam_target_pos.cpu()))
                cam_target_pos_filtered = self.cam_target_pos_filter_buffer.add(self.cam_target_pos)
                cam_pos_filtered = self.cam_pos_filter_buffer.add(self.cam_pos)
                self.gym.viewer_camera_look_at(
                    self.viewer, self.envs[self.ref_env], gymapi.Vec3(*cam_pos_filtered.cpu()), gymapi.Vec3(*cam_target_pos_filtered.cpu()))

        return

    def draw_debug_lines(self):

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        if self.debug_viz:
            # draw height lines
            # self.gym.refresh_rigid_body_state_tensor(self.sim)
            # visualizing command
            viz_cmd_start_point = torch.clone(self.root_state[:, :3])  # base pos
            # viz_cmd_start_point[:,2]+=0.5

            viz_cmd_xy_endpoint = torch.zeros(size=(self.num_envs, 3), dtype=torch.float, device=self.device)
            viz_cmd_xy_endpoint[:, :2] = self.commands[:, :2]
            viz_cmd_xy_endpoint = (viz_cmd_start_point+quat_apply_yaw(
                self.base_quat, viz_cmd_xy_endpoint)) # scaled

            viz_cmd_yaw_endpoint = torch.clone(viz_cmd_start_point)
            viz_cmd_yaw_endpoint[:, 2] += self.commands[:, 2]

            verts = torch.column_stack([viz_cmd_start_point, viz_cmd_xy_endpoint,
                                        viz_cmd_start_point, viz_cmd_yaw_endpoint]).view((-1, 12)).cpu().numpy().view(dtype=gymapi.Vec3.dtype)
            colors_vel_dir = np.array([(1, 1, 0), (1, 1, 0)],dtype=gymapi.Vec3.dtype)

            for i, env in enumerate(self.envs):
                self.gym.add_lines(self.viewer, env, colors_vel_dir.shape[0], verts[i], colors_vel_dir)
                sphere_pose = gymapi.Transform(verts[i,1], r=None)
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, env, sphere_pose)
                sphere_pose = gymapi.Transform(verts[i,3], r=None)
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, env, sphere_pose)

            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_state[:, :3]
            ).unsqueeze(1)
            self._all_height_points_xy[:, :self.num_height_points,:] = points[:, :, :2]
            self._all_height_points_xy[:, self.num_height_points:,:] = self.rb_state[:,:, :2]

            xy = self._all_height_points_xy.cpu().numpy()
            z = self.heights_absolute.cpu().numpy()
            for i in range(self.num_envs):  # draw height points
                 for j in range(xy.shape[1]):
                    sphere_pose = gymapi.Transform(gymapi.Vec3(xy[i,j,0], xy[i,j,1], z[i,j]), r=None)
                    if j<=self.num_height_points:
                        gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                    else:
                        gymutil.draw_lines(self.sphere_geom_alt_color, self.gym, self.viewer, self.envs[i], sphere_pose)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        self.should_fetch_result = True
        self.should_step_graphics = True

        # # randomize actions # TODO: rurrently need to completely bypass this
        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        self.action = torch.clamp(actions.clone().to(self.device), -self.clip_actions, self.clip_actions)

        if self.enable_data_receiver: # keyboard operator overrides using data from self.data_receiver
            if self.data_receiver_data_id != self.data_receiver.data_id:
                self.data_receiver_data_id = self.data_receiver.data_id
                if "cmd" in self.data_receiver.data:
                    self.keyboard_operator_cmd[:] = torch.tensor(self.data_receiver.data["cmd"],device=self.device)
                    print(f"keybaord cmd:{self.keyboard_operator_cmd}")
                if "reset" in self.data_receiver.data and self.data_receiver.data["reset"] == True:
                    self.progress_buf[:]= self.max_episode_length
                if "push" in self.data_receiver.data and self.data_receiver.data["push"] == True:
                    self.push_robot_base_indexed(torch.arange(self.num_envs, device=self.device))
                # if "dof_pos_target" in self.data_receiver.data:
                #     # self.dof_pos_target[:,self.data_receiver.data["leg_id"]] = torch.tensor(self.data_receiver.data["dof_pos_target"],device=self.device)
                #     self.dof_pos_target = self.desired_dof_pos[:]
                #     # self.dof_pos_target[:] = torch.tensor(self.data_receiver.data["dof_pos_target"],device=self.device)
                #     self.dof_pos_target[:,self.data_receiver.data["leg_id"]] = torch.tensor(self.data_receiver.data["dof_pos_target"],device=self.device)
                if "action" in self.data_receiver.data:
                    self.action[:,self.data_receiver.data["leg_id"]] = torch.tensor(self.data_receiver.data["action"],device=self.device)
                    # self.dof_pos_target[:] = self.action_scale * self.action_to_use + self.default_dof_pos[:,self.actuated_dof_mask]


        if self.randomize_action_delay:
            self.action_delay.uniform_(*self.action_delay_log_range).exp_()
            self.action_to_use = self.action * (1 - self.action_delay) + self.action_delay * self.action_to_use
        else:
            self.action_to_use = self.action

        if self.enable_passive_dynamics:
            self.dof_pos_target = self.action_scale * self.action_to_use[:,:self.num_actuated_dof] + self.default_dof_pos[:,self.actuated_dof_mask]
            self.action_is_on = sigmoid_k(self.action_to_use[:,self.num_actuated_dof:], self.action_is_on_sigmoid_k)
            if self.passive_curriculum:  # TODO change the hardcoded values to a variable
                # self.passive_action_min_value = 1.0 - float(min(self.common_step_counter/5e4,1.0))
                # self.passive_action_min_value = 0.8 - float(min(self.common_step_counter/5e4,0.8))
                # self.passive_action_min_value = 0.5 - float(min(self.common_step_counter/5e4,0.5))

                self.passive_action_min_value = max(0.5-self.common_step_counter/5e4,self.min_action_is_on)

                # self.passive_action_min_value = 0.5 - float(min(self.common_step_counter/5e4,0.5)) # always 10% on
                # self.action_is_on = torch.clamp_min(self.action_is_on,min=self.passive_action_min_value)
                self.action_is_on = self.passive_action_min_value+(1-self.passive_action_min_value)*self.action_is_on # alternative

                if self.common_step_counter % 1000 == 0:
                    print(f"self.passive_action_min_value={self.passive_action_min_value}")
            else:
                self.action_is_on = self.min_action_is_on + (1 - self.min_action_is_on) * self.action_is_on
            # self.action_is_on = self.actions_to_use[:,self.num_actuated_dof:]  > -0.5
            # self.actions[:, self.num_actuated_dof:] = torch.where(self.action_is_on, 1, -1)
        else:
            self.dof_pos_target = self.action_scale * self.action_to_use + self.default_dof_pos[:,self.actuated_dof_mask]
            self.action_is_on = 1

        if self.should_dof_disable:
            if self.disable_dof_freeze:
                self.dof_pos_target[self.disable_dof_boolean] = self.default_dof_pos[self.disable_dof_boolean] # HACK TODO CHANGE BACK
            else:
                self.dof_pos_target[self.disable_dof_boolean] = self.dof_pos[self.disable_dof_boolean]

        # # soft limit clamping
        # below_min_limit = (self.dof_pos < self.dof_soft_limit_lower) & (self.dof_pos_target < self.dof_soft_limit_lower)
        # above_max_limit = (self.dof_pos > self.dof_soft_limit_upper) & (self.dof_pos_target > self.dof_soft_limit_upper)
        # torch.where(condition=below_min_limit,input=self.dof_soft_limit_lower,other=self.dof_pos_target,out=self.dof_pos_target)
        # torch.where(condition=above_max_limit,input=self.dof_soft_limit_upper,other=self.dof_pos_target,out=self.dof_pos_target)


        self.pre_physics_step()
        # to fix!
        if (self.device == 'cpu' or self.camera_sensor_enable) and self.should_fetch_result:
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # #TODO currently need to completely bypass this
        # # step physics and render each frame
        # for i in range(self.control_freq_inv):
        #     if self.force_render:
        #         self.render()
        #     self.gym.simulate(self.sim)
        if self.force_render:
            self.render()
            self.gym.clear_lines(self.viewer)

        self.control_steps += 1

        # # randomize observations #TODO currently need to completely bypass dr_randomizations
        # if self.dr_randomizations.get('observations', None):
        #     self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.update_obs_dict()
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


    def pre_physics_step(self):
        """PD position control"""
        if self.randomize_body_force:
            self.rb_forces *= self.force_decay
            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero().ravel()
            if force_indices.numel() > 0:
                self.rb_forces[force_indices, :self.num_bodies, :] = self.rb_forces[force_indices, :self.num_bodies, :].uniform_(-1.0,1.0)* self.rb_force_mags[force_indices]
            # random force perturbation
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)


        if self.enable_erfi: # extended random force injection
            self.erfi_rfi.uniform_(*self.erfi_rfi_range).add_(self.erfi_rao)

        # b = 0.1 # soft dof limit # HACK THIS SHOULD BE CONFIGURED
        # a = torch.tensor(self.action_scale, dtype=torch.float, device=self.device).expand_as(self.dof_pos)
        # k = (b-a)/b
        # dof_pos_target_lower_bound = torch.max(-a, -a + k*self.dof_pos)
        # dof_pos_target_upper_bound = torch.min(a,   a + k*self.dof_pos)
        # self.dof_pos_target.clamp_(dof_pos_target_lower_bound, dof_pos_target_upper_bound)

        self.last_dof_pos[:] = self.dof_pos
        self.last_dof_vel_computed[:] = self.dof_vel_computed

        for i in range(self.decimation):
            self.dof_pos_error[:] = self.dof_pos_target - self.dof_pos

            if self.cfg['env']["ray_obs"]['static_debug']:
                self.dof_pos_error[:] = 0

            self.dof_pos_error_integral.add_(self.dof_pos_error * self.dt) # integral error
            # force_computed = self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel # TODO FIXME
            force_computed = self.ki * self.dof_pos_error_integral + self.kp * self.dof_pos_error - self.kd * self.dof_vel # TODO FIXME
            self.last_dof_pos_substep[:] = self.dof_pos
            # force_computed = self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel_computed_single_substep # TODO FIXME
            if i==0:
                self.actuated_dof_force_target[:] = force_computed # this is the idealized computed torque without motor strength/randomization

            # robstride02 motor linear fit
            # self.dof_max_linear_force_to_accelerate = torch.clip(self.voltage_motor-6-torch.abs(self.dof_vel/self.radius_rotor_wheel),min=0,max=self.dof_max_torque)/self.radius_rotor_wheel
            # dof_vel_is_positive = self.dof_vel>=0 # TODO FIXME
            # dof_vel_is_negative = self.dof_vel<0 # TODO FIXME
            self.dof_max_linear_force_to_accelerate = self.get_dof_max_linear_force_to_accelerate(self.dof_vel_computed_single_substep)
            dof_vel_is_positive = self.dof_vel_computed_single_substep>=0 # TODO FIXME
            dof_vel_is_negative = self.dof_vel_computed_single_substep<0 # TODO FIXME

            force_computed[dof_vel_is_positive].clamp_(None,self.dof_max_linear_force_to_accelerate[dof_vel_is_positive])
            force_computed[dof_vel_is_negative].clamp_(-self.dof_max_linear_force_to_accelerate[dof_vel_is_negative],None)

            force_computed.clamp_(-self.dof_max_linear_force,self.dof_max_linear_force)
            # force_computed.clamp_(-0,0) #HACK TODO CHANGE BACK
            if self.enable_passive_dynamics:
                force_computed*=self.action_is_on
            force_actual = force_computed*self.dof_strength+self.erfi_rfi  # randomized dof strength # TODO: make it better
            if i==0: # use the fist iteration FOR SIM2RAL maching
                self.actuated_dof_force_target_actual[:] = force_actual
            self.dof_actuation_force[:] = force_actual # need to [:] to use the in-place version
            # # HACK FOR THROW ONLY CHANGE BACK
            # self.dof_actuation_force[:self.num_envs//2] = -100

            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(toque_actual))
            self.gym.set_dof_actuation_force_tensor(self.sim, self.dof_actuation_force_tensor)
            self.gym.simulate(self.sim)
            if self.device == 'cpu': # must fetch after simulate for any tensor.
                self.gym.fetch_results(self.sim, True)
                self.should_fetch_result = False
            self.gym.refresh_dof_state_tensor(self.sim)

            self.dof_vel_computed_single_substep[:] = (self.dof_pos - self.last_dof_pos_substep) / self.dt

            self.dof_vel_computed_all_substeps[i] = self.dof_vel_computed_single_substep

            # self.dof_vel[:] = (self.dof_pos - self.last_dof_pos) / self.dt
        # self.dof_vel_computed[:] = (self.dof_pos - self.last_dof_pos) / self.rl_dt
        self.dof_vel_computed[:] = 0.1*((self.dof_pos - self.last_dof_pos) / self.rl_dt) + 0.9*self.last_dof_vel_computed
        self.gym.refresh_rigid_body_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def pre_physics_step_with_passive_dof(self):
    # def pre_physics_step(self):
        """PD position control"""
        if self.enable_erfi: # extended random force injection
            self.erfi_rfi.uniform_(*self.erfi_rfi_range).add_(self.erfi_rao)
        for i in range(self.decimation):
            torque = self.kp * (self.dof_pos_target - self.dof_pos[:,self.actuated_dof_mask]) - self.kd * self.dof_vel[:,self.actuated_dof_mask]
            if self.enable_passive_dynamics:
                torque*=self.action_is_on
            torque.add_(self.erfi_rfi).clamp_(self.actuated_dof_force_min, self.actuated_dof_force_max)
            # TODO maybe check if action exceeds limit and make it a reward
            self.dof_actuation_force[:,self.actuated_dof_mask] = torque
            self.gym.set_dof_actuation_force_tensor(self.sim, self.dof_actuation_force_tensor)

            if self.num_marker_pairs > 0:
                marker_pos_error = self.rb_state[:,self.marker_pair_ids[:,1],:3] -  self.rb_state[:,self.marker_pair_ids[:,0],:3] # [num_envs, num_marker_pairs, 3]
                marker_pos_error_norm = torch.linalg.vector_norm(marker_pos_error, ord=2, dim=-1, keepdim=True) # [num_envs, num_marker_pairs, 1]

                marker_force = 100000 * (marker_pos_error_norm-self.marker_pair_l0.view(1,self.num_marker_pairs,1))*marker_pos_error/marker_pos_error_norm
                forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                forces[:,self.marker_pair_ids[:,0]] = marker_force
                forces[:,self.marker_pair_ids[:,1]] = -marker_force
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)

            self.gym.simulate(self.sim)
            if self.device == 'cpu': # must fetch after simulate for any tensor.
                self.gym.fetch_results(self.sim, True)
                self.should_fetch_result = False
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.actuated_dof_force_target[:] = torque
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)


    def post_physics_step(self):
        if self.camera_sensor_enable:
            if self.should_fetch_result:
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.cam_tensor_stack = torch.stack([torch.stack(inner_list) for inner_list in self.cam_tensors])[...,:3] # rgba->rgb
            # print(self.cam_tensor_stack.shape)

            self.gym.end_access_image_tensors(self.sim)

            self.grayscale_obs_image()

            if self.camera_sensor_visualize:
                # shape: (num_cameras, height, width)
                camera_sample =self.obs_image_grayscale[self.ref_env]
                num_cameras, height, width = camera_sample.shape
                img = camera_sample.transpose(1, 2).reshape(width * num_cameras, height).transpose(0, 1).cpu().numpy()
                cv2.imshow("img", img)
                cv2.waitKey(1)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        # if self.common_step_counter % self.push_interval == 0 and self.should_push_robots:
        #     self.push_robots()

        if self.should_push_robots:
            env_ids = torch.nonzero(self.progress_buf%self.push_interval == 0).squeeze_(1)
            if env_ids.numel()>0:
                self.push_robot_base_indexed(env_ids)

        # prepare quantities
        # self.base_quat = self.root_state[:, 3:7]
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 7:10])
        self.base_lin_vel = self.root_state[:, 7:10] # in world space!!!!!!!!
        self.world_space_base_ang_vel = self.root_state[:, 10:13]
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.base_gravity_vec)
        self.base_rotation_matrix[:] = quaternion_to_rotation_matrix(self.base_quat)

        if self.randomize_projected_gravity_delay:
            self.projected_gravity_delay.uniform_(*self.projected_gravity_delay_log_range).exp_()
            self.projected_gravity_filtered[:] = normalize(self.projected_gravity_delay*self.projected_gravity_filtered + (1 - self.projected_gravity_delay)*self.projected_gravity)
        else:
            self.projected_gravity_filtered[:] = self.projected_gravity

        if self.randmoize_orientation_delay:
            self.orientation_delay.uniform_(*self.orientation_delay_log_range).exp_()
            # self.base_quat_filtered[:] = normalize(self.orientation_delay*self.base_quat_filtered[:]  + (1 - self.orientation_delay)*self.base_quat)
            self.base_quat_filtered[:] = slerp(self.orientation_delay, self.base_quat, self.base_quat_filtered)
            self.base_rotation_matrix_filtered[:] = quaternion_to_rotation_matrix(self.base_quat_filtered)
        else:
            self.base_quat_filtered[:] = self.base_quat
            self.base_rotation_matrix_filtered[:] = self.base_rotation_matrix

        self.base_forward = quat_apply(self.base_quat, self.base_forward_vec_local)
        heading = torch.atan2(self.base_forward[:, 1], self.base_forward[:, 0])

        self.base_quat_yaw = get_quat_yaw(self.base_quat)

        self.dof_acc = (self.dof_vel - self.last_dof_vel) * self.rl_dt_inv  # TODO check if [:] is needed # TODO
        self.dof_acc_computed = (self.dof_vel_computed - self.last_dof_vel_computed) * self.rl_dt_inv  # TODO check if [:] is needed # TODO
        # self.dof_jerk = (self.dof_acc - self.last_dof_acc) * self.rl_dt_inv
        self.base_relative_angular_velocity_computed = calculate_angular_velocity_tensor(self.last_base_quat,self.base_quat,self.rl_dt)
        self.foot_quat = self.rb_state[:, self.foot_ids, 3:7] # foot quaternion [num_envs, num_foot, 4] # this is a copy!!
        self.foot_projected_gravity=quat_rotate_inverse(self.foot_quat.view(-1, 4), self.foot_gravity_vec).view(self.num_envs, self.num_foot, 3)

        self.foot_forward = quat_apply(self.foot_quat.view(-1, 4), self.foot_forward_vec_local).view(self.num_envs, self.num_foot, 3)

        # # foot_quat relative to the base yaw frame: foot_quat * base_quat_yaw_inv
        # base_quat_yaw_inv = quat_conjugate(self.base_quat_yaw) # 1/quat_yaw
        # self.foot_quat_rel = quat_mul(self.foot_quat.view(-1, 4), base_quat_yaw_inv.repeat_interleave(self.num_foot, dim=0)).view(self.num_envs, self.num_foot, 4) # foot quat relative to the base yaw frame
        # self.foot_forward_rel = quat_apply(self.foot_quat_rel.view(-1, 4), self.foot_forward_vec_local).view(self.num_envs, self.num_foot, 3)

        self.foot_pos = self.rb_state[:, self.foot_ids, 0:3]

        if self.cfg['env']['save_blender_trajectory']:
            self.rb_position_blender_trajectory.append(self.rb_state[0, :, :7].clone())
            np.save(f'blender_rendering/rb_position_blender_trajectory.npy', np.array([tensor.cpu().numpy() for tensor in self.rb_position_blender_trajectory]))
            if self.object_tracking_enabled or self.object_pushing_enabled:
                self.object_state_blender_recording.append(self.all_root_state[self.object_handles,:7])
                self.ray_point_blender_recording.append(self.perfect_point_cloud_blender.clone())
                np.save(f'blender_rendering/ray_point_blender_recording.npy', np.array([tensor.cpu().numpy() for tensor in self.ray_point_blender_recording]))
                np.save(f'blender_rendering/object_state_blender_recording.npy', np.array([tensor.cpu().numpy() for tensor in self.object_state_blender_recording]))
            print(self.sim_step_count)
            if self.sim_step_count > 100:
                print("finished_data_collection ... exiting ...")
                sys.exit(0)

        # relative foot position in yaw frame
        foot_pos_rel =  self.foot_pos - self.root_state[:, :3].view(self.num_envs, 1, 3)

        self.foot_pos_rel_yaw =  quat_rotate_inverse(self.base_quat_yaw.repeat_interleave(self.num_foot, dim=0),
                                   foot_pos_rel.view(-1, 3)).view(self.num_envs, self.num_foot, 3)

        self.foot_lin_vel = self.rb_state[:, self.foot_ids, 7:10]

        # foot contact
        self.foot_contact_force = self.contact_force[:, self.foot_ids, :]
        self.foot_contact = self.foot_contact_force[:, :, 2] > self.foot_contact_threshold  # todo check with norm
        # # HACK: no contact
        # self.foot_contact[:] = 0

        self.foot_contact_filt = torch.logical_or(self.last_foot_contact, self.foot_contact) # filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.last_foot_contact = self.foot_contact
        # self.foot_contact = self.sensor_forces[:, :, 2] > 1.0

        if self.viewer and self.enable_keyboard_operator:
            self.commands[:, 3] = wrap_to_pi(2 * self.commands[:, 2] + heading)
        else:
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # set zero command if the magnitude is too small
        self.is_zero_command[:] = square_sum(self.commands[:, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        if self.guided_contact: # update phase for the contact sequence
            self.update_phase()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_observations()

        self.compute_reward()

        if self.evaluate:
            # evaluate touch down
            touch_down = self.foot_contact.any(dim=1)
            first_contact =(~self.has_first_contact) & touch_down
            if first_contact.any():
                self.base_pos_at_first_contact[first_contact, :] = self.base_pos[first_contact, :]
                self.has_first_contact |= first_contact
                # print(first_contact)


        # update last_...

        if self.enable_passive_dynamics:
            self.last_action_is_on = self.action_is_on

        # self.dof_pos_filt[:] = self.dof_pos_filt * 0.97 + self.dof_pos * 0.03
        self.last_action[:] = self.action
        self.last_dof_vel[:] = self.dof_vel
        self.last_dof_acc[:] = self.dof_acc
        self.last_foot_contact_force[:] = self.foot_contact_force
        self.last_base_quat[:] = self.base_quat

        # resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if self.object_tracking_enabled:

            self.x_tracking_position += self.random_x_displacement
            self.y_tracking_position += self.random_y_displacement
            if self.cfg['env']["ray_obs"]['random_vel']:
                mask = (self.progress_buf+1) % 80 == 0
                temp_x_displacement = self.random_x_displacement[mask].clone()

                # Random choice between (-y, x) and (y, -x)
                random_sign = torch.randint(0, 2, (mask.sum(),), device=self.device) * 2 - 1  # generates -1 or 1

                self.random_x_displacement[mask] = -self.random_y_displacement[mask] * random_sign
                self.random_y_displacement[mask] = temp_x_displacement * random_sign

                self.object_vel[:, 0] = self.random_x_displacement/self.rl_dt
                self.object_vel[:, 1] = self.random_y_displacement/self.rl_dt
                # mask = self.progress_buf % 80 == 0
                # len_ids = mask.sum()  # Count of True values
                # self.random_x_displacement[mask], self.random_y_displacement[mask] = sample_points_with_norm_constraint(
                #     self.displacement_range[0], 
                #     self.displacement_range[1], 
                #     len_ids, 
                #     self.device
                # )
            # Update root state while ensuring correct indexing
            if not self.cfg['env']["ray_obs"]['static_debug']:
                self.all_root_state[self.object_handles, 0] += self.random_x_displacement
                self.all_root_state[self.object_handles, 1] += self.random_y_displacement
                self.all_root_state[self.object_handles, 3:7] = self.cube_orientation
            
            self.all_root_state[self.object_handles, 2] = self.cube_pose[2]
            

            self.all_root_state[self.object_handles,7:10] = torch.tensor([0,0,0],dtype=torch.float).repeat(self.num_envs, 1).to(self.device)
            self.all_root_state[self.object_handles,10:13] = torch.tensor([0,0,0],dtype=torch.float).repeat(self.num_envs, 1).to(self.device)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.all_root_state),
                                                        gymtorch.unwrap_tensor(self.object_handles),
                                                        self.num_envs)


    def init_height_points(self):
        """
        initialize height points in cpu, save self.num_height_points and self.height_points
        self.height_points[:,:self.num_height_points,0] is grid_x
        self.height_points[:,:self.num_height_points,1] is grid_y
        self.num_height_points[:,-1,:] is base (0,0,0)
        """
        cfg_heightmap = self.cfg["env"]["heightmap"]
        x = cfg_heightmap["x"]
        y = cfg_heightmap["y"]

        x = torch.tensor(np.linspace(x['start'], x['end'], x['num']) , dtype=torch.float, device=self.device)
        y = torch.tensor(np.linspace(y['start'], y['end'], y['num']), dtype=torch.float, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        # num_envs, num_points_per_env, xyz
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        self.height_points = points
        # num_envs, num_points_per_env+num_bodies, xyz
        self.heights_absolute = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, device=self.device, dtype=torch.float)
        self.heights_relative = torch.empty_like(self.heights_absolute)
        self._all_height_points_xy = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, 2, device=self.device, dtype=torch.float)

    def get_rays(self):
        base_origin = self.rb_state[:,self.base_id,:3]
        direction = self.rb_state[:,self.foot_ids,:3] - self.rb_state[:,self.base_id,:3].unsqueeze(1)
        foot_origin = self.rb_state[:,self.foot_ids,:3]
        foot_rot = self.rb_state[:,self.foot_ids,3:7]

        self.get_point_cloud_ray_casting_from_foot_batch(base_origin.cpu().detach().numpy(),foot_origin.cpu().detach().numpy(), direction.cpu().detach().numpy(), foot_rot.cpu().detach().numpy())
        
        if self.num_perception_units == 12:
            self.updated_ray_point_clouds[...,:-self.num_perception_units*self.num_points_per_foot,:]=0

        elif self.num_perception_units == 20:
            self.updated_ray_point_clouds[...,self.num_perception_units*self.num_points_per_foot:,:]=0

        if self.enable_ray_visualization:
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, self.sphere_pose, color=(1, 1, 0))
            for location in self.updated_ray_point_clouds[0]:
                transform = gymapi.Transform(p=gymapi.Vec3(location[0],location[1],location[2]+0.6), r=gymapi.Quat(0, 0,0,0)) # camera uses z axis
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.ref_env], transform)
        self.point_cloud_history_buffer[:,:-self.num_points] = self.point_cloud_history_buffer[:,self.num_points:].clone()
        self.point_cloud_history_buffer[:,-self.num_points:] = self.updated_ray_point_clouds.clone()
        self.distance_history_buffer[:,:-self.num_points] = self.distance_history_buffer[:,self.num_points:].clone()
        self.distance_history_buffer[:,-self.num_points:] = self.updated_ray_distance.clone() # update the distance history buffer

    def get_rays_asymmetric(self):
        base_origin = self.rb_state[:,self.base_id,:3]
        base_orientation = self.rb_state[:,self.base_id,3:7]
        #apply orientation to the asymmetric foot position
        foot_origin,foot_rot = transform_asymmetric_joint_positions(base_origin,base_orientation,
                                                                         self.asymmetric_joint_positions,self.asymmetric_joint_orientations)
        direction = foot_origin - base_origin.unsqueeze(1)

        self.get_point_cloud_ray_casting_from_foot_batch(base_origin.cpu().detach().numpy(),foot_origin.cpu().detach().numpy(), direction.cpu().detach().numpy(), foot_rot.cpu().detach().numpy())
        
        if self.num_perception_units == 12:
            self.updated_ray_point_clouds[...,:-self.num_perception_units*self.num_points_per_foot,:]=0

        elif self.num_perception_units == 20:
            self.updated_ray_point_clouds[...,self.num_perception_units*self.num_points_per_foot:,:]=0

        if self.enable_ray_visualization:
                sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, self.sphere_pose, color=(1, 1, 0))
                for location in self.updated_ray_point_clouds[0]:
                    transform = gymapi.Transform(p=gymapi.Vec3(location[0],location[1],location[2]+0.6), r=gymapi.Quat(0, 0,0,0)) # camera uses z axis
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.ref_env], transform)
                sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, self.sphere_pose, color=(0, 1, 1))
                for location in foot_origin[0]:
                    transform = gymapi.Transform(p=gymapi.Vec3(location[0],location[1],location[2]), r=gymapi.Quat(0, 0,0,0)) # camera uses z axis
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.ref_env], transform)

        self.point_cloud_history_buffer[:,:-self.num_points] = self.point_cloud_history_buffer[:,self.num_points:].clone()
        self.point_cloud_history_buffer[:,-self.num_points:] = self.updated_ray_point_clouds.clone()
        self.distance_history_buffer[:,:-self.num_points] = self.distance_history_buffer[:,self.num_points:].clone()
        self.distance_history_buffer[:,-self.num_points:] = self.updated_ray_distance.clone() # update the distance history buffer

    def get_heights(self):
        """get heights at sampled locations"""
        # if self.terrain_type == 'plane': # heights of the plane at sampled locations are all zero
        #     self.heights_absolute = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, device=self.device)
        if self.terrain_type != 'plane':
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_state[:, :3]
            ).unsqueeze(1)
            self._all_height_points_xy[:, :self.num_height_points,:] = points[:, :, :2]
            self._all_height_points_xy[:, self.num_height_points:,:] = self.rb_state[:,:, :2]
            # self.root_states: (num_env,13)
            # points: (num_env,num_points_per_env+1 (root_pos),3 (xyz))
            # ## points = torch.cat((points, self.root_states[:, :3].unsqueeze(1)), dim=1)
            self.heights_absolute = self.terrain.get_heights(self._all_height_points_xy).view(self.num_envs, -1)
            # heights_absolute: (num_env,num_points_per_env+1 (body_com))

        self.heights_relative[:,:self.num_height_points] = self.root_state[:, 2].unsqueeze(1) - self.heights_absolute[:,:self.num_height_points]
        self.heights_relative[:,self.num_height_points:] = self.rb_state[:,:self.num_bodies, 2] - self.heights_absolute[:,self.num_height_points:]
        self.base_height: torch.Tensor = self.heights_relative[:, self.num_height_points+self.base_id]
        self.foot_height: torch.Tensor = self.heights_relative[:, self.num_height_points+self.foot_ids]

def get_matching_str(source, destination, case_sensitive=False, comment=""):
    """Finds case-insensitive partial matches between source and destination lists."""
    def find_matches(src_item):
        if case_sensitive:
            matches = [item for item in destination if src_item in item]
        else:
            matches = [item for item in destination if src_item.lower() in item.lower()]
        if not matches:
            raise KeyError(f"cannot locate {src_item}. [{comment}]")
        elif len(matches) > 1:
            raise KeyError(f"find multiple instances for {src_item}. [{comment}]")
        return matches[0]  # Return just the first match
    if isinstance(source, str):  # one to many
        if case_sensitive:
            matches = [item for item in destination if source in item]
        else:
            matches = [item for item in destination if source.lower() in item.lower()]
        if not matches:
            raise KeyError(f"cannot locate {source} [{comment}\navailables are {destination}")
        return matches
    else:  # one to one
        return [find_matches(item) for item in source]

def slerp(t, q1, q2, epsilon=1e-6):
    """
    Spherical linear interpolation (SLERP) between normalized quaternions q1 and q2.
    Args:
        t: float or (n, 1) tensor — interpolation factor(s) in [0, 1]
        q1, q2: tensors of shape (n, 4), assumed normalized
    Returns:
        Interpolated quaternions of shape (n, 4)
    """
    # Dot product and handle shortest path
    dot = torch.sum(q1 * q2, dim=1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    # Clamp dot for numerical safety
    dot = torch.clamp(dot, -1.0, 1.0)
    # Compute omega
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # SLERP and LERP blend weights
    slerp_mask = sin_omega > epsilon
    scale0 = torch.where(
        slerp_mask,
        torch.sin((1.0 - t) * omega) / sin_omega,
        1.0 - t
    )
    scale1 = torch.where(
        slerp_mask,
        torch.sin(t * omega) / sin_omega,
        t
    )
    # Interpolate and normalize result
    result = scale0 * q1 + scale1 * q2
    return result/torch.norm(result, dim=1, keepdim=True)


# @torch.jit.script
def random_quaternion(len, device):
    temp_vec = torch.empty(len,4, device=device, dtype=torch.float)
    temp_vec.uniform_(-1, 1)
    temp_vec/=temp_vec.norm(dim=1, keepdim=True)
    return temp_vec

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def get_quat_yaw(quat) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_tensor(lower: torch.Tensor, upper: torch.Tensor, shape: Tuple[int, int], device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def square_sum(input: torch.Tensor,dim: int=-1) -> torch.Tensor:
    return torch.square(input).sum(dim=dim)

@torch.jit.script
def square_sum_clamp_max(input: torch.Tensor,dim: int=-1, max: float=1.0) -> torch.Tensor:
    return torch.square(input).sum(dim=dim).clamp_max_(max)

# @torch.jit.script
def exp_square_sum(input: torch.Tensor,exp_scale: float, dim: int=-1) -> torch.Tensor:
    return torch.exp(torch.square(input).sum(dim=dim)*exp_scale)

# @torch.jit.script
def exp_square_mean(input: torch.Tensor,exp_scale: float, dim: int=-1) -> torch.Tensor:
    return torch.exp(torch.square(input).mean(dim=dim)*exp_scale)

@torch.jit.script
def exp_square(input: torch.Tensor,exp_scale: float) -> torch.Tensor:
    return torch.exp(torch.square(input)*exp_scale)

@torch.jit.script
def exp_weighted_square_sum(x: torch.Tensor, exp_scale: torch.Tensor,dim: int=-1):
    return torch.exp(torch.sum(exp_scale*x.square(), dim=dim))


@torch.jit.script
def exp_weighted_abs_sum(x: torch.Tensor, exp_scale: torch.Tensor,dim: int=-1):
    return torch.exp(torch.sum(exp_scale*abs(x), dim=dim))

@torch.jit.script
def out_of_bound_norm(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return torch.norm(input - torch.clamp(input, lower, upper), dim=dim)

# @torch.jit.script
def out_of_bound_square_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return torch.square(input - torch.clamp(input, lower, upper)).sum(dim=dim)

def out_of_bound_exp_square_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, exp_scale: float, dim: int=-1) -> torch.Tensor:
    return torch.exp(torch.square(input - torch.clamp(input, lower, upper)).sum(dim=dim)*exp_scale)

@torch.jit.script
def out_of_bound_abs_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return (input - torch.clamp(input, lower, upper)).abs().sum(dim=dim)


@torch.jit.script
def out_of_float_bound_squared_sum(input: torch.Tensor, lower: float, upper: float, dim: int=-1) -> torch.Tensor:
    return torch.square(input - torch.clamp(input, lower, upper)).sum(dim=dim)


# jit is slower here so do not use jit
def abs_sum(input: torch.Tensor) -> torch.Tensor:
    return input.abs().sum(dim=-1)


# https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
# https://www.mathworks.com/help/fuzzy/gbellmf.html
# https://www.mathworks.com/help/fuzzy/dsigmf.html
# https://www.mathworks.com/help/fuzzy/foundations-of-fuzzy-logic.html
@torch.jit.script
def bell(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    return 1 / (1 + torch.pow(torch.abs(x / a - c), b))

@torch.jit.script
def log_unifrom(tensor:torch.Tensor,log_low:float, log_high:float):
    return tensor.uniform_(log_low, log_high).exp_()

@torch.jit.script
def reverse_bell(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    return 1 - 1 / (1 + torch.pow(torch.abs(x / a - c), b))

@torch.jit.script
def sigmoid_k(x: torch.Tensor, k: float) -> torch.Tensor:
    return 1 / (1 + torch.exp(-k*x))


# TODO verify if this is correct
@torch.jit.script
def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to rotation matrices.
    Input:
        quaternion: (n, 4) tensor where quaternion is (x, y, z, w)
    Output:
        rotation_matrix: (n, 3, 3) tensor
    """
    x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Compute the elements of the rotation matrix
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    rot_matrix = torch.zeros((quaternion.shape[0], 3, 3), device=quaternion.device)

    rot_matrix[:, 0, 0] = 1 - 2 * (yy + zz)
    rot_matrix[:, 0, 1] = 2 * (xy - zw)
    rot_matrix[:, 0, 2] = 2 * (xz + yw)

    rot_matrix[:, 1, 0] = 2 * (xy + zw)
    rot_matrix[:, 1, 1] = 1 - 2 * (xx + zz)
    rot_matrix[:, 1, 2] = 2 * (yz - xw)

    rot_matrix[:, 2, 0] = 2 * (xz - yw)
    rot_matrix[:, 2, 1] = 2 * (yz + xw)
    rot_matrix[:, 2, 2] = 1 - 2 * (xx + yy)

    return rot_matrix

def calculate_angular_velocity_tensor(q1, q2, dt):
    """
    Calculate angular velocity from two consecutive quaternions using PyTorch tensors.

    Parameters:
    q1 (torch.Tensor): First quaternion with shape (num_env, 4) in [x, y, z, w] format
    q2 (torch.Tensor): Second quaternion with shape (num_env, 4) in [x, y, z, w] format
    dt (float or torch.Tensor): Time step between quaternions in seconds

    Returns:
    torch.Tensor: Angular velocity vector with shape (num_env, 3) in the first quaternion's frame
    """

    # Normalize quaternions
    q1_norm = torch.norm(q1, dim=1, keepdim=True)
    q2_norm = torch.norm(q2, dim=1, keepdim=True)
    q1 = q1 / q1_norm
    q2 = q2 / q2_norm

    # Calculate the inverse of q1 (conjugate for unit quaternions)
    q1_inv = torch.cat([-q1[:, :3], q1[:, 3:]], dim=1)

    # Calculate the relative quaternion (q_rel = q1_inv * q2)
    q_rel = quaternion_multiply_batch(q1_inv, q2)

    # Extract components for axis-angle conversion
    vector_part = q_rel[:, :3]  # [x, y, z] components
    scalar_part = q_rel[:, 3:]  # w component

    # Calculate the angle of rotation
    vector_norm = torch.norm(vector_part, dim=1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, scalar_part)

    # Handle small rotations (avoid division by zero)
    mask = (vector_norm > 1e-6).float()
    safe_norm = torch.max(vector_norm, torch.ones_like(vector_norm) * 1e-6)

    # Calculate the axis of rotation (normalized vector part)
    axis = vector_part / safe_norm

    # Default axis for near-zero rotations
    default_axis = torch.zeros_like(vector_part)
    default_axis[:, 2] = 1.0  # z-axis as default

    # Blend between computed axis and default axis based on mask
    axis = axis * mask + default_axis * (1 - mask)

    # Calculate the rotation vector (angle * axis)
    rotvec = axis * angle

    # Angular velocity is the rotation vector divided by time
    if isinstance(dt, torch.Tensor) and dt.dim() > 0:
        # If dt is a batched tensor
        angular_velocity = rotvec / dt.view(-1, 1)
    else:
        # If dt is a scalar or 0-dim tensor
        angular_velocity = rotvec / dt

    return angular_velocity

def quaternion_multiply_batch(q1, q2):
    """
    Multiply two batches of quaternions.

    Parameters:
    q1 (torch.Tensor): First quaternions with shape (num_env, 4) in [x, y, z, w] format
    q2 (torch.Tensor): Second quaternions with shape (num_env, 4) in [x, y, z, w] format

    Returns:
    torch.Tensor: Resulting quaternions with shape (num_env, 4) in [x, y, z, w] format
    """
    # Extract components
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    # Calculate the product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=1)


def random_vector_with_norm(size, norm):
    vec = np.random.randn(*size)  # Generate random vector (normal distribution)
    vec /= np.linalg.norm(vec)    # Normalize to unit vector
    vec *= norm                   # Scale to desired norm
    return vec


def axis_angle_to_quaternion(axis, angle):
    """
    Convert an axis-angle representation to a quaternion.

    Parameters:
        axis (list or np.array): A 3D unit vector representing the axis of rotation (e.g., [x, y, z]).
        angle (float): The rotation angle in radians.

    Returns:
        np.array: The quaternion [w, x, y, z].
    """
    # Ensure the axis is a unit vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Use scipy to calculate the quaternion
    quaternion = R.from_rotvec(angle * axis).as_quat()  # Returns [x, y, z, w]

    # Reorder to [w, x, y, z]
    return np.roll(quaternion, shift=1)

def generate_camera_ray_directions(
        fov_deg=[70, 60],  # Field of view in degrees
        resolution=[25, 25], # Resolution in pixels
        device="cpu"
):

    # Camera parameters
    fov_x_deg, fov_y_deg = fov_deg
    resolution_x,resolution_y = resolution

    # Convert FOV to radians
    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)

    # Calculate the maximum extent in x and y on the image plane at z=1
    max_x = np.tan(fov_x_rad / 2.0)
    max_y = np.tan(fov_y_rad / 2.0)

    # Create a grid of pixel coordinates
    u_ndc = (torch.arange(resolution_x, dtype=torch.float32, device=device) + 0.5) / resolution_x
    v_ndc = (torch.arange(resolution_y, dtype=torch.float32, device=device) + 0.5) / resolution_y
    u_screen = 2.0 * u_ndc - 1.0
    v_screen = 2.0 * v_ndc - 1.0
    screen_x, screen_y = torch.meshgrid(u_screen, v_screen, indexing='xy')

    camera_x = screen_x * max_x
    camera_y = screen_y * max_y # TODO verify this
    camera_z = torch.ones_like(camera_x)

    ray_directions = torch.stack([camera_x, camera_y, camera_z], dim=-1)
    ray_directions_normalized = torch.nn.functional.normalize(ray_directions, dim=-1)

    # plt.plot(screen_x, screen_y, 'o')
    # plt.xlabel('u')
    # plt.ylabel('v')
    # plt.show()

    return ray_directions_normalized # (resolution_y, resolution_x, 3)


def visualize_rays(ray_directions, resolution=[25, 25]):
    resolution_x,resolution_y = resolution
    ray_directions_cpu = ray_directions.cpu().numpy()
    # --- Visualization ---
    import matplotlib.pyplot as plt
    # Camera origin
    camera_origin = torch.tensor([0.0, 0.0, 0.0])
    # Select a subset of rays to visualize
    # We'll select the rays at the corners and the center
    ray_indices_v = [0, resolution_y - 1, resolution_y // 2]
    ray_indices_u = [0, resolution_x - 1, resolution_x // 2]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the camera origin
    ax.scatter(camera_origin[0], camera_origin[1], camera_origin[2], c='red', marker='o', label='Camera Origin')
    # Plot selected rays
    ray_length = 2.0  # Scale the unit direction vector for visualization length
    for v_idx in ray_indices_v:
        for u_idx in ray_indices_u:
            direction = ray_directions_cpu[v_idx, u_idx, :]
            end_point = camera_origin + ray_length * direction

            ax.plot([camera_origin[0], end_point[0]],
                    [camera_origin[1], end_point[1]],
                    [camera_origin[2], end_point[2]],
                    color='blue', alpha=0.6)
    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Camera Rays')
    # Set equal aspect ratio to prevent distortion
    ax.set_box_aspect([1,1,1]) # or use ax.set_aspect('equal') depending on matplotlib version
    # Show the plot
    plt.legend()
    plt.show()

def get_ray_casting_pyramid(origins, directions, quaternions,resolution=[5,5]):
    all_ray_origins = []
    all_ray_directions = []

    # Each origin should have exactly 25 rays (5×5 grid)
    num_origins = len(origins)
    expected_total_rays = num_origins * resolution[0] * resolution[1]

    for origin, direction, quaternion in zip(origins, directions, quaternions):
        # Normalize the direction vector to ensure proper alignment
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # Convert the quaternion to a Rotation object
        rotation = R.from_quat(quaternion)

        # We need to properly incorporate the quaternion rotation
        # First, establish basis vectors in the local space
        local_forward = np.array([0.0, 0.0, 1.0])  # Assuming Z-forward convention
        local_right = np.array([1.0, 0.0, 0.0])   # X is right
        local_up = np.array([0.0, 1.0, 0.0])      # Y is up

        # Apply quaternion rotation to get world-space basis vectors
        forward = rotation.apply(local_forward)
        right = rotation.apply(local_right)
        up = rotation.apply(local_up)

        # Ensure the forward direction matches the provided direction
        # This is important as the quaternion might not align exactly with the direction
        forward = direction

        # Re-orthogonalize the basis vectors to ensure they're perpendicular
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Generate evenly spaced tangent values for horizontal and vertical FOV
        h_max = np.tan(70 * np.pi / 180 / 2)*(1-1/resolution[0])  # Half angle for FoV
        v_max = np.tan(60 * np.pi / 180 / 2)*(1-1/resolution[1])  # Half angle for FoV
        
        h_values = np.linspace(-h_max, h_max, resolution[0])  # Horizontal values
        v_values = np.linspace(-v_max, v_max, resolution[1])  # Vertical values

        # Generate rays for the pyramid
        ray_directions = []

        # Generate a 5×5 grid of rays
        for h in h_values:
            for v in v_values:
                # Calculate ray direction using the orthogonal basis
                # This creates a projection that will appear as a grid on a flat surface
                ray_dir = forward + h * right + v * up
                # import ipdb; ipdb.set_trace()
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                ray_directions.append(ray_dir)

        ray_directions = np.array(ray_directions)
        ray_origins = np.tile(origin, (len(ray_directions), 1))  # Repeat origin for each ray

        all_ray_directions.append(ray_directions)
        all_ray_origins.append(ray_origins)

    # Concatenate all the origins and directions from different origins
    all_ray_origins = np.vstack(all_ray_origins)
    all_ray_directions = np.vstack(all_ray_directions)

    # Verify the shape
    assert all_ray_origins.shape[0] == expected_total_rays, f"Expected {expected_total_rays} rays, got {all_ray_origins.shape[0]}"
    assert all_ray_directions.shape[0] == expected_total_rays, f"Expected {expected_total_rays} rays, got {all_ray_directions.shape[0]}"

    return all_ray_origins, all_ray_directions

def sample_points_with_norm_constraint(low_norm, high_norm, n_samples, device=None):
    """
    Sample x, y points such that their norm is within [low_norm, high_norm]
    """
    # Method 1: Polar coordinates approach
    # Sample radius uniformly from [low_norm, high_norm]
    radius = torch.rand(n_samples, device=device) * (high_norm - low_norm) + low_norm

    # Sample angle uniformly from [0, 2π]
    angle = torch.rand(n_samples, device=device) * 2 * torch.pi

    # Convert to Cartesian coordinates
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)

    return x, y




def convert_ray_distance_to_position(origins, directions, quaternions, robot_root_pos, distance_pixel_normalized, resolution=[5,5], dropout_rate=0.1, noise_std=0.01,apply_noise=True):
    """
    Convert ray distances to 3D positions - FIXED VERSION with dropout and Gaussian noise

    Args:
        origins: Ray origins
        directions: Ray directions
        quaternions: Sensor quaternions
        robot_root_pos: Robot root position
        distance_pixel_normalized: Normalized pixel distances
        resolution: Resolution [height, width]
        dropout_rate: Probability of dropping out a point (default: 0.1 for 10%)
        noise_std: Standard deviation for Gaussian noise (default: 0.01)
    """
    point_positions = []
    num_valid_points = 0
    # Convert all distances from normalized pixel values to meters
    distances_meter = tof_to_depth(np.array(distance_pixel_normalized) * 255)  # Shape: (20, 25)

    # distances_meter = distance_pixel_normalized*2.5
    for sensor_idx, (origin, direction, quaternion) in enumerate(zip(origins, directions, quaternions)):
        # Normalize the direction vector

        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # Convert quaternion to rotation
        rotation = R.from_quat(quaternion)

        # MATCH SIMULATION: Use the same local coordinate frame
        local_forward = np.array([0.0, 0.0, 1.0])  # Z-forward (same as simulation)
        local_right = np.array([1.0, 0.0, 0.0])    # X-right (same as simulation)
        local_up = np.array([0.0, 1.0, 0.0])       # Y-up (same as simulation)

        # Apply quaternion rotation to get world-space basis vectors
        forward = rotation.apply(local_forward)
        right = rotation.apply(local_right)
        up = rotation.apply(local_up)

        # # MATCH SIMULATION: Override forward with provided direction and re-orthogonalize
        forward = direction

        # Re-orthogonalize the basis vectors (same as simulation)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Generate FOV parameters (same as simulation)
        h_max = np.tan(70 * np.pi / 180 / 2)  # Half horizontal FOV
        v_max = np.tan(60 * np.pi / 180 / 2)  # Half vertical FOV

        h_values = np.linspace(-h_max, h_max, resolution[0])  # 5 horizontal values
        v_values = np.linspace(-v_max, v_max, resolution[1])  # 5 vertical values

        # Generate rays for this sensor (5×5 = 25 rays)
        ray_idx = 0  # Index within this sensor's 25 rays
        # MATCH SIMULATION: Same nested loop order (h outer, v inner)
        num_valid_points += sum(distances_meter[sensor_idx, :] < 1.5)

        for h in h_values:
            for v in v_values:
                # Calculate ray direction in world space (same as simulation)
                ray_dir = forward + h * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # Calculate 3D position: origin + distance * ray_direction
                if distances_meter[sensor_idx, ray_idx] < 1.5:
                    # Apply random dropout (10% chance to drop the point)
                    if np.random.random() < dropout_rate:
                        position_3d = np.array([0, 0, 0])  # Drop the point
                    else:
                        # Calculate position normally
                        position_3d = np.array(origin) + distances_meter[sensor_idx, ray_idx] * ray_dir - robot_root_pos

                        # Add Gaussian noise to the position
                        if apply_noise:
                            noise = np.random.normal(0, noise_std, size=3)
                            position_3d = position_3d + noise
                else:
                    position_3d = np.array([0, 0, 0])
                if position_3d[-1] < 0.05 and np.linalg.norm(position_3d[:2]) > 1.5:
                    position_3d = np.array([0,0,0])

                point_positions.append(position_3d)

                ray_idx += 1

    return np.array(point_positions)

def compute_y_axis_velocity_alignment_reward(orientation, velocity_cmd):
    """
    Compute reward for aligning object's x-axis OR y-axis with velocity command direction.
    Takes the maximum alignment between x-axis and y-axis alignment with velocity.
    
    Args:
        orientation: torch.Tensor of shape (num_envs, 4) - quaternions [x, y, z, w]
        velocity_cmd: torch.Tensor of shape (num_envs, 3) - velocity commands [x, y, z]
    
    Returns:
        reward: torch.Tensor of shape (num_envs,) - alignment rewards
    """
    # Normalize quaternions to ensure they're unit quaternions
    orientation = F.normalize(orientation, dim=1)
    
    # Extract quaternion components [x, y, z, w]
    x, y, z, w = orientation[:, 0], orientation[:, 1], orientation[:, 2], orientation[:, 3]
    
    # Y-axis column of rotation matrix from quaternion
    y_axis = torch.stack([
        2 * (x * y - w * z),      # y_axis x-component
        1 - 2 * (x**2 + z**2),    # y_axis y-component  
        2 * (y * z + w * x)       # y_axis z-component
    ], dim=1)
    
    # Normalize velocity command (handle zero velocity case)
    velocity_norm = torch.norm(velocity_cmd, dim=1, keepdim=True)
    velocity_normalized = torch.where(
        velocity_norm > 1e-6,
        velocity_cmd / velocity_norm,
        torch.zeros_like(velocity_cmd)
    )
    
    # Compute dot product (cosine similarity) for both axes
    y_dot_product = torch.sum(y_axis * velocity_normalized, dim=1)
    
    # Clamp to handle numerical errors
    # x_dot_product = torch.clamp(x_dot_product, -1.0, 1.0)
    y_dot_product = torch.clamp(y_dot_product, -1.0, 1.0)
    
    # Take the maximum absolute alignment (best alignment regardless of direction)
    y_alignment = torch.abs(y_dot_product)

    return torch.clamp(-1/y_alignment+1, -0.2, 0.0)
    
def transform_asymmetric_joint_positions(base_pos, base_quat, joint_pos_dict, joint_quat_dict):
    """
    Transform joint positions and orientations from local to world coordinates
    
    Args:
        base_pos: [num_envs, 3] - base position in world frame
        base_quat: [num_envs, 4] - base orientation quaternion (x,y,z,w)
        joint_pos_dict: dict {joint_name: [x,y,z]} - joint positions in local frame
        joint_quat_dict: dict {joint_name: [x,y,z,w]} - joint quaternions in local frame
    
    Returns:
        foot_origin: [num_envs, 20, 3] - all joint positions in world frame
        foot_rot: [num_envs, 20, 4] - all joint quaternions in world frame (x,y,z,w)
    """
    
    joint_names = list(joint_pos_dict.keys())
    num_envs = base_pos.shape[0]
    num_joints = len(joint_names)  # Should be 20
    
    # Convert to tensors
    joint_positions = []
    joint_quaternions = []
    
    for name in joint_names:
        pos = torch.tensor(joint_pos_dict[name], device=base_pos.device, dtype=base_pos.dtype)
        quat = torch.tensor(joint_quat_dict[name], device=base_pos.device, dtype=base_pos.dtype)
        
        # Ensure correct shape
        if pos.dim() > 1:
            pos = pos.flatten()
        if quat.dim() > 1:
            quat = quat.flatten()
            
        joint_positions.append(pos[:3])  # Ensure exactly 3 elements
        joint_quaternions.append(quat[:4])  # Ensure exactly 4 elements
    
    # Stack
    joint_pos_local = torch.stack(joint_positions, dim=0)  # [20, 3]
    joint_quat_local = torch.stack(joint_quaternions, dim=0)  # [20, 4]
    
    # Expand for batch processing
    joint_pos_local = joint_pos_local.unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, 20, 3]
    joint_quat_local = joint_quat_local.unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, 20, 4]
    
    # Transform
    joint_pos_world = torch.zeros_like(joint_pos_local)
    joint_quat_world = torch.zeros_like(joint_quat_local)
    
    # Convert to numpy for scipy operations
    base_pos_np = base_pos.cpu().numpy()
    base_quat_np = base_quat.cpu().numpy()  # [x,y,z,w]
    joint_pos_local_np = joint_pos_local.cpu().numpy()
    joint_quat_local_np = joint_quat_local.cpu().numpy()  # [x,y,z,w]
    
    for env_idx in range(num_envs):
        # Create base rotation object (quaternion is already in [x,y,z,w] format)
        base_rot = Rotation.from_quat(base_quat_np[env_idx])
        
        for joint_idx in range(num_joints):
            # Transform position
            pos_local = joint_pos_local_np[env_idx, joint_idx]
            pos_rotated = base_rot.apply(pos_local)
            joint_pos_world[env_idx, joint_idx] = torch.tensor(
                pos_rotated + base_pos_np[env_idx], 
                device=base_pos.device, dtype=base_pos.dtype
            )
            
            # Transform orientation (quaternion is already in [x,y,z,w] format)
            joint_rot_local = Rotation.from_quat(joint_quat_local_np[env_idx, joint_idx])
            joint_rot_world = base_rot * joint_rot_local
            joint_quat_world[env_idx, joint_idx] = torch.tensor(
                joint_rot_world.as_quat(),  # Returns [x,y,z,w]
                device=base_pos.device, dtype=base_pos.dtype
            )
    
    return joint_pos_world, joint_quat_world  # [num_envs, 20, 3], [num_envs, 20, 4]
## this script is derived from anymal_terrain.py in IsaacGymEnvs
# https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/anymal_terrain.py
# the orignal script contains the copyright notice as below
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.