import os
import sys
import random
import shutil
import time
from dataclasses import dataclass,field
from tqdm import tqdm
import datetime
import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import List, Optional
import tensordict
from torch.utils.tensorboard import SummaryWriter
from torch_utils import Agent, ImageAgent, MixedAgent, ActiveAgent, RayEncoder, RayEncoder_pointnet
from hydra import compose
from hydra.initialize import initialize_config_dir
import train


def yaw_to_quaternion(yaw):
    """
    Convert yaw angle (in radians) to quaternion
    
    Args:
        yaw: yaw angle(s) in radians, can be scalar or tensor
    
    Returns:
        quaternion tensor with shape (..., 4) in format [x, y, z, w]
    """
    # Ensure yaw is a tensor
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=torch.float32)
    
    # Creates quaternions representing rotation around Z-axis (yaw only)
    half_yaw = yaw / 2
    
    # Create quaternion components
    x = torch.zeros_like(half_yaw)
    y = torch.zeros_like(half_yaw)
    z = torch.sin(half_yaw)
    w = torch.cos(half_yaw)
    
    # Stack into quaternion format [x, y, z, w]
    return torch.stack([x, y, z, w], dim=-1)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    rl_device: str = "cuda:0"
    """device to train on, cpu or cuda:0, cuda:1, etc."""
    sim_device: str = "cuda:0"
    """device to simulate physics on, cpu or cuda:0, cuda:1, etc."""
    headless: bool = False
    """if toggled, visualization is disabled"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    config_dir: str = "./cfg"
    """the directory containing the yaml config files"""
    task_name: str = "Ant"
    """the id of the environment"""
    wandb_project_name: str = None
    """the wandb's project name"""
    wandb_run_name: str = "larger_cube"
    """the wandb's run name"""
    wandb_entity: str = 'grl_argus'
    """the entity (team) of wandb's project"""
    run_name: str = None
    """the name of the run. If none, it will be automatically generated"""
    run_dir: str = None
    """the directory to store logs and checkpoints"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    total_timesteps: int = 3000000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.003
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 2
    """coefficient of the value function"""
    bounds_loss_coef: float = 1e-3
    """coefficient of the bound loass"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    kl_threshold: Optional[float] = 0.02
    """the target KL divergence threshold"""
    # reward_scaler: float = 1
    # """the scale factor applied to the reward during training"""
    record_video_step_frequency: int = 1464
    """the frequency at which to record the videos"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint: Optional[str] = None
    """the checkpoint to load"""
    supervise_checkpoint: str = None
    """the checkpoint to load"""
    encoder_checkpoint: str = None
    """the checkpoint to load"""
    checkpoint_save_frquency: int = 50
    """the frequency at which to save the checkpoint"""
    log_interval: int = 5
    """the interval at which to log the statistics"""

    train_mode: str = "train"
    """training mode {train, play, collect}"""

    agent_name: str = "baseline"
    """agent type {baseline, image, mixed}"""

    asymmetric_observations: bool = False
    """asymmetric observations for actor and critic (computed in runtime)"""

    num_obs: int = 0
    """the number of observations excluding image (computed in runtime)"""
    num_state: int = 0
    """the number of states (computed in runtime)"""
    num_action: int = 0
    """the number of actions (computed in runtime)"""

    # # list of strings
    # task_args: List[str] = field(default_factory=lambda: [])

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    @torch.compiler.disable(recursive=False)
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs: int = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    @torch.compiler.disable(recursive=False)
    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return tensordict.TensorDict(observations)
    
    @torch.compiler.disable(recursive=False)
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        dones = dones.bool()
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        not_dones = ~dones
        self.episode_returns *= not_dones
        self.episode_lengths *= not_dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            tensordict.TensorDict(observations),
            rewards,
            dones,
            infos,
        )

class AdaptiveScheduler:
    def __init__(self, kl_threshold = 0.01):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


if __name__ == "__main__":

    Args = tyro.conf.configure( # refence: https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/wisp/config/_tyro.py
        tyro.conf.AvoidSubcommands, # Avoid creating subcommands when a default is provided for unions over nested types.
        tyro.conf.ConsolidateSubcommandArgs, # More robust to reordering of options, ensuring that any new options can simply be placed at the end
        tyro.conf.FlagConversionOff, # support both optional and non-optional boolean args
        tyro.conf.SuppressFixed # Hides fields which are marked as fixed (i.e. predetermined value in dataclass).
    )(Args)

    args, task_args = tyro.cli(Args, return_unknown_args=True)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.task_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    now = datetime.datetime.now()

    if args.wandb_project_name is None:
        args.wandb_project_name = args.task_name
    if args.run_name is None:
        run_name = f"runs/{args.train_mode}/{args.wandb_run_name}_{now.strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = f"{args.run_name}"

    if args.run_dir is None:
        args.run_dir = run_name
        run_dir = os.path.abspath(args.run_dir)
        os.makedirs(run_dir,exist_ok=True)

    # task_args += args.task_args
    config_dir = os.path.abspath(args.config_dir)
    with initialize_config_dir(config_dir,version_base=None):
        cfg_dict = compose(config_name="config", overrides=[f"task={args.task_name}"]+task_args)
    print(f"task={args.task_name}",task_args)
    cfg_dict['task']['env']['numEnvs'] = args.num_envs

    args.asymmetric_observations = cfg_dict["task"]["env"].get("asymmetric_observations",False)


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.empty_cache()

    device = torch.device(args.rl_device)
        
    # env setup
    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.task_name,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=0,
        headless=args.headless,
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=True,
        cfg=cfg_dict
    )
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )

    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    obs_image_shape = getattr(envs, "obs_image_shape", (12,32,32))
    if args.agent_name == "activeagent" or args.agent_name == "mixed_ray":
        obs_ray_shape = (getattr(envs, "num_foot")*getattr(envs, "num_points_per_foot")*getattr(envs, "point_history_length"),)
    args.num_obs = np.array(envs.single_observation_space.shape).prod()
    
    if hasattr(envs, "state_space") and envs.num_states > 0:
        args.num_state = np.array(envs.state_space.shape).prod()
        args.asymmetric_observations = True

    args.num_action = np.array(envs.single_action_space.shape).prod()

    should_use_ray_obs = False 
    
    if args.agent_name == "baseline":
        agent = Agent(
            actor_obs_dim=args.num_obs,
            critic_obs_dim=args.num_state,
            action_dim=args.num_action,
            asymmetric_observations=args.asymmetric_observations
        )
        should_use_image_obs = False
    elif args.agent_name == "image":
        agent = ImageAgent(
            actor_image_obs_dim=args.num_obs,
            critic_obs_dim=args.num_state,
            action_dim=args.num_action,
            asymmetric_observations=args.asymmetric_observations
        )
        should_use_image_obs = True
    elif args.agent_name == "mixed":
        agent = MixedAgent(
            actor_obs_dim=args.num_obs,
            actor_image_obs_dim=obs_image_shape,
            critic_obs_dim=args.num_state,
            action_dim=args.num_action,
            asymmetric_observations=args.asymmetric_observations
        )
        should_use_image_obs = True

    elif args.agent_name == "mixed_ray":
        should_use_image_obs = False
        should_use_ray_obs = True

        # ray_encoder = RayEncoder(
        #     num_stacked_obs_frame=envs.num_stacked_obs_frame,
        #     ray_obs_dim=obs_ray_shape[0],
        # )

        ray_encoder = RayEncoder_pointnet(
            num_stacked_obs_frame=envs.num_stacked_obs_frame,
            ray_obs_dim=obs_ray_shape[0],
            prediction_dim=2 if envs.object_tracking_enabled else 1, # 2 for object tracking, 1 for object pushing
        )

        ray_encoder = torch.compile(ray_encoder)

        agent = Agent(
            actor_obs_dim=args.num_obs,
            critic_obs_dim=args.num_state,
            action_dim=args.num_action,
            asymmetric_observations=args.asymmetric_observations
        )
        ray_encoder: torch.nn.Module = ray_encoder.to(device) 
        agent: torch.nn.Module = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    elif args.agent_name == "activeagent":
        should_use_image_obs = False
        should_use_ray_obs = True

        agent = ActiveAgent(
            actor_obs_dim=args.num_obs,
            critic_obs_dim=args.num_state,
            num_stacked_obs_frame=envs.num_stacked_obs_frame,
            ray_obs_dim=obs_ray_shape[0],
            action_dim=args.num_action,
            asymmetric_observations=args.asymmetric_observations
        )
        agent = torch.compile(agent)
        if args.train_mode != "play":
            # Load checkpoint
            current_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current file
            checkpoint_path = os.path.join(current_path, args.supervise_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            agent_state_dict = checkpoint["agent"]
            # Create a filtered state dictionary that only includes keys present in the new model
            filtered_state_dict = {}
            for name, param in agent_state_dict.items():
                # Skip parameters for ray_encoder which doesn't exist in the checkpoint
                if "ray_encoder" not in name and name in agent.state_dict():
                    filtered_state_dict[name] = param

            # Load the filtered state dictionary
            agent.load_state_dict(filtered_state_dict, strict=False)

            # Move the entire model to the target device
            agent = agent.to(device)

            # Also make sure all buffers like running mean/std are on the correct device
            for buffer in agent.buffers():
                buffer.data = buffer.data.to(device)

            # Freeze the actor_mean parameters after loading
            for param in agent.actor_mean.parameters():
                param.requires_grad = False
            # agent.actor_logstd.requires_grad = False

            for param in agent.critic.parameters():
                param.requires_grad = False

        agent: torch.nn.Module = agent.to(device) 
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.agent_name != "activeagent":
        agent = torch.compile(agent)
        agent: torch.nn.Module = agent.to(device) 
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # # automatic mixed precision https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # scaler = torch.GradScaler()

    if args.checkpoint is not None:
        print('Loading models from {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=True)
        agent_state_dict = checkpoint["agent"]
        # agent_state_dict_current = agent.state_dict()
        # for k in agent_state_dict_current.keys():
        #     if k not in agent_state_dict: # fix missing keys
        #         agent_state_dict[k] = agent_state_dict_current[k]
        agent.load_state_dict(checkpoint["agent"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # scaler.load_state_dict(checkpoint["scaler"])

    if args.encoder_checkpoint is not None:
        print('Loading supervised models from {}'.format(args.encoder_checkpoint))
        checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
        agent_state_dict = checkpoint["agent"]
        agent_state_dict_current = ray_encoder.state_dict()
        
        for k in agent_state_dict_current.keys():
            if k not in agent_state_dict:  # fix missing keys
                agent_state_dict[k] = agent_state_dict_current[k]
    
        ray_encoder.load_state_dict(agent_state_dict)  

    if args.train_mode=="collect": # collect data
        buf_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float,device=device)
        buf_action_mean = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float,device=device)
        buf_action_logstd = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float,device=device)
        buf_next_obs = torch.zeros_like(buf_obs,device=device)
        buf_reward = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float,device=device)
        buf_done = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float,device=device)
        
        should_use_image_obs = False # collect image obs
        if should_use_image_obs:
            buf_obs_image = torch.zeros((args.num_steps, args.num_envs) + obs_image_shape, dtype=torch.float,device=device) 
            buf_next_obs_image = torch.zeros_like(buf_obs_image,device=device)
        
        use_obs2 = False
        if hasattr(envs, "use_obs2") and envs.use_obs2: # collect obs2
            use_obs2 = True
            buf_obs2 = torch.zeros((args.num_steps, args.num_envs) + envs.obs2_space.shape, dtype=torch.float,device=device)
            buf_next_obs2 = torch.zeros_like(buf_obs2,device=device)

        should_use_ray_obs = False
        if hasattr(envs, "use_ray_obs"):
            should_use_ray_obs = True
            buf_obj_velocity = torch.zeros((args.num_steps, args.num_envs) + (3,), dtype=torch.float,device=device)
            buf_obs_ray_point_cloud = torch.zeros_like(envs.point_cloud_history_buffer,device=device).repeat((args.num_steps, 1, 1, 1))
            buf_obs_ray_distance = torch.zeros_like(envs.distance_history_buffer,device=device).repeat((args.num_steps, 1, 1, 1))
            buf_obs_joint_directions = torch.zeros_like(envs.joint_directions,device=device).repeat((args.num_steps, 1, 1, 1))
            buf_obs_joint_origins = torch.zeros_like(envs.joint_origins,device=device).repeat((args.num_steps, 1, 1, 1))
            buf_obs_joint_quaternions = torch.zeros_like(envs.joint_quaternions,device=device).repeat((args.num_steps, 1, 1, 1))

            buf_obs_robot_root_position = torch.zeros_like(envs.robot_root_position,device=device).repeat((args.num_steps, 1, 1, 1))
            buf_obs_robot_root_quaternions = torch.zeros_like(envs.robot_root_quaternions,device=device).repeat((args.num_steps, 1, 1, 1))

            bur_next_obs_ray_point_cloud = torch.zeros_like(buf_obs_ray_point_cloud,device=device).repeat((args.num_steps, 1,1, 1))
            buf_next_obs_ray_distance = torch.zeros_like(buf_obs_ray_distance,device=device).repeat((args.num_steps, 1,1, 1))
        if hasattr(envs, "object_pushing_enabled"):
            buf_obj_orientation = torch.zeros((args.num_steps, args.num_envs) + (4,), dtype=torch.float,device=device)
            buf_contact = torch.zeros_like(envs.foot_contact,device=device).repeat((args.num_steps, 1, 1))
        # collect state as well
        # args.asymmetric_observations = True
        if args.asymmetric_observations:
            buf_state = torch.zeros((args.num_steps, args.num_envs) + envs.state_space.shape, dtype=torch.float,device=device)
            buf_next_state = torch.zeros_like(buf_state,device=device)

        agent.train(mode=False)
        data_dir = run_dir

        def collect_data():
            obs_dict = envs.reset()
            for iteration in tqdm(range(1, args.num_iterations + 1)):
                for step in range(0, args.num_steps):
                    with torch.inference_mode(True):                        
                        action_mean, action_logstd = agent.get_action_mean_and_logstd(obs_dict)
                        next_obs_dict, reward, done, info = envs.step(action_mean)
                        # gather datas
                        if should_use_image_obs:
                            buf_obs_image[step] = obs_dict["obs_image"]
                            # buf_next_obs_image[step] = next_obs_dict["obs_image"]

                        if args.asymmetric_observations and not should_use_ray_obs:
                            buf_state[step] = obs_dict["states"]
                            # buf_next_state[step] = next_obs_dict["states"]
                        if use_obs2:
                            buf_obs2[step] = obs_dict["obs2"]
                            buf_next_obs2[step] = next_obs_dict["obs2"]

                        if should_use_ray_obs:

                            buf_obj_velocity[step] = obs_dict["object_velocity"]
                            buf_obs_ray_point_cloud[step] = obs_dict["ray_point_cloud"]
                            bur_next_obs_ray_point_cloud[step] = next_obs_dict["ray_point_cloud"]    
                            buf_obs_ray_distance[step] = obs_dict["ray_distance"]
                            
                            buf_obs_joint_directions[step] = obs_dict["joint_directions"]
                            buf_obs_joint_origins[step] = obs_dict["joint_origins"]
                            buf_obs_joint_quaternions[step] = obs_dict["joint_quaternions"]
                            
                            buf_obs_robot_root_position[step] = obs_dict["robot_root_position"]
                            buf_obs_robot_root_quaternions[step] = obs_dict["robot_root_quaternions"]
                            if envs.object_pushing_enabled:
                                buf_obj_orientation[step] = obs_dict["object_orientation"]
                                buf_contact[step] = obs_dict["contact"]
                        buf_action_mean[step] = action_mean
                        # buf_action_logstd[step] = action_logstd
                        buf_obs[step] = obs_dict["obs"]
                        # buf_next_obs[step] = next_obs_dict["obs"]
                        # buf_reward[step] = reward
                        # buf_done[step] = done
                        # update next
                        # obs_dict = next_obs_dict.clone()
                        # obs_dict = next_obs_dict.detach()
                        obs_dict = next_obs_dict
                ts = {
                        "obs": buf_obs,
                        "action_mean": buf_action_mean,
                        # "action_logstd": buf_action_logstd,
                        # "next_obs": buf_next_obs,
                        # "reward": buf_reward,
                        # "done": buf_done,
                    }
                if should_use_image_obs:
                    ts["obs_image"] = buf_obs_image
                    ts["next_obs_image"] = buf_next_obs_image
                if args.asymmetric_observations:
                    ts["states"] = buf_state
                    # ts["next_states"] = buf_next_state
                if use_obs2:
                    ts["obs2"] = buf_obs2
                    ts["next_obs2"] = buf_next_obs2
                if should_use_ray_obs:
                    ts["object_velocity"] = buf_obj_velocity
                    ts["ray_point_cloud"] = buf_obs_ray_point_cloud
                    ts["ray_distance"] = buf_obs_ray_distance
                    ts["next_ray_point_cloud"] = bur_next_obs_ray_point_cloud
                    ts["joint_directions"] = buf_obs_joint_directions
                    ts["joint_origins"] = buf_obs_joint_origins
                    ts["joint_quaternions"] = buf_obs_joint_quaternions
                    ts["robot_root_position"] = buf_obs_robot_root_position
                    ts["robot_root_quaternions"] = buf_obs_robot_root_quaternions
                if hasattr(envs, "object_pushing_enabled"):
                    ts["object_orientation"] = buf_obj_orientation
                    ts["contact"] = buf_contact
                torch.save(ts,f"{data_dir}/{iteration}.pt")

        # collect_data = torch.compile(collect_data)
        collect_data()
        try:
            collect_data()
        except KeyboardInterrupt:
            exit()
        finally:
            exit()
    elif args.train_mode=="play": # player only
        if args.encoder_checkpoint is not None:
            ray_encoder.train(mode=False)
            agent.train(mode=False)
            obs_dict = envs.reset()
            prev_velocity = None
            prev_orientation = None
            alpha=1

            loss_fn = torch.nn.MSELoss()

            try:
                with torch.inference_mode(True):
                    for step in tqdm(range(args.total_timesteps//args.num_envs)):

                        #ray_distance
                        # rotation_matrix = obs_dict['obs'][..., -12:-3]
                        # predicted_vel = ray_encoder.get_object_velocity_prediction(rotation_matrix,obs_dict['ray_distance'])
                        if envs.object_tracking_enabled :
                            #ray_point_cloud
                            rotation_matrix = obs_dict['obs'][..., -12:-3]
                            dof_pos = obs_dict['obs'][..., 3:23]
                            worldSpaceAngularVelocity = obs_dict['obs'][..., :3]
                            predicted_vel = ray_encoder.get_object_velocity_prediction_play(obs_dict['ray_point_cloud'])
                            


                            zeros = torch.zeros(*predicted_vel.shape[:-1], 1, device=predicted_vel.device)  # Shape: (a,b,c,1)
                            predicted_vel = torch.cat([predicted_vel, zeros], dim=-1)  # Shape: (a,b,c,3)

                            if prev_velocity is None:
                                smoothed_velocity = predicted_vel.clone()
                            else:
                                smoothed_velocity = alpha * predicted_vel + (1 - alpha) * prev_velocity
                            
                            prev_velocity = smoothed_velocity.clone()   

                            # predicted_vel[...,-1] = 0.0 # set z velocity to 0.0
                            loss = loss_fn(obs_dict["obs"][:,-3:], smoothed_velocity)# + 0.5*loss_fn(predicted_logstd, action_logstd)

                            obs_dict["obs"][:,-3:] = smoothed_velocity                   
                        if  envs.object_pushing_enabled :
                            obs_dict['ray_point_cloud'] = obs_dict['ray_point_cloud'][..., -500:,:]
                            prediction = ray_encoder.get_object_velocity_prediction_play(obs_dict['ray_point_cloud'])

                            #smoothing
                            # obs_dict["obs"][:,-27:] = prediction     
                            # if prev_orientation is None:
                            #     prediction = prediction.clone()
                            # else:
                            #     prediction = alpha * prediction + (1 - alpha) * prev_orientation
                            # prev_velocity = prediction.clone()   

                            prediction = yaw_to_quaternion(prediction).squeeze(1) # convert to quaternion
                            loss = loss_fn(obs_dict["obs"][:,-4:], prediction)# + 0.5*loss_fn(predicted_logstd, action_logstd)
                            obs_dict["obs"][:,-4:] = prediction  

                        action_mean, action_logstd = agent.get_action_mean_and_logstd(obs_dict)


                        next_obs_dict, reward, next_done, info = envs.step(action_mean)
                        obs_dict = next_obs_dict
                        # obs, action_mean, action_logstd, 
            except KeyboardInterrupt:
                exit()
            finally:
                exit()
        else:
            agent.train(mode=False)
            obs_dict = envs.reset()
            try:
                with torch.inference_mode(True):
                    for step in tqdm(range(args.total_timesteps//args.num_envs)):
                        # num_action_dof, 1
                        action_mean, action_logstd = agent.get_action_mean_and_logstd(obs_dict)
                        next_obs_dict, reward, next_done, info = envs.step(action_mean)
                        obs_dict = next_obs_dict
                        # obs, action_mean, action_logstd, 
            except KeyboardInterrupt:
                exit()
            finally:
                exit()

    # train
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            dir=run_dir,
            sync_tensorboard=True,
            config={**vars(args),**cfg_dict},
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        current_folder = os.path.dirname(os.path.abspath(__file__))
        file_to_copy = [f"{current_folder}/tasks/{args.task_name}.py",   
                        f"{current_folder}/cfg/task/{args.task_name}.yaml",
                        f"{current_folder}/cfg/train/{args.task_name}PPO.yaml"]

        for path in file_to_copy:
            destination = os.path.join(run_dir, os.path.basename(path))
            try:
                shutil.copy(path, destination)
                print(f"Copied {path} to {destination}")
            except FileNotFoundError:
                print(f"File {path} not found. Skipping copy.")

    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    if args.anneal_lr and should_use_ray_obs != True:
        lr_scheduler = AdaptiveScheduler(kl_threshold=args.kl_threshold)
    elif should_use_ray_obs:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float,device=device)
    if should_use_ray_obs:
        obs_ray = torch.zeros((args.num_steps, args.num_envs) + obs_ray_shape, dtype=torch.float,device=device)
        obs_object_velocity = torch.zeros((args.num_steps, args.num_envs) + (3,), dtype=torch.float,device=device)
    if should_use_image_obs:
        obs_images = torch.zeros((args.num_steps, args.num_envs) + obs_image_shape, dtype=torch.float,device=device)
    if args.asymmetric_observations:
        states = torch.zeros((args.num_steps, args.num_envs) + envs.state_space.shape, dtype=torch.float,device=device)

    action_means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float,device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float,device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float, device=device)
    advantages = torch.zeros_like(rewards, dtype=torch.float, device=device)

    # statistics
    episode_returns = torch.zeros(args.num_steps,args.num_envs, dtype=torch.float, device=device)
    episode_lengths = torch.zeros(args.num_steps,args.num_envs, dtype=torch.int32, device=device)
    next_dones = torch.zeros(args.num_steps, args.num_envs, dtype=torch.bool, device=device)
    mean_episode_return = torch.zeros(1,dtype=torch.float, device=device)
    mean_episode_length = torch.zeros(1,dtype=torch.float, device=device)
    max_mean_episode_return = -torch.inf
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs_dict = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    def save_checkpoint(checkpoint_path):
        print(f"\nSaving model to {checkpoint_path}")
        torch.save({
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scaler": scaler.state_dict(),
        }, checkpoint_path)

    # @torch.compile
    def bound_loss(mu: torch.Tensor, soft_bound = 1.0):
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
        b_loss = (mu_loss_low + mu_loss_high).mean()
        return b_loss

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        global_step += args.num_envs*args.num_steps
        # agent.train(mode=False)
        for step in range(args.num_steps):
            obs[step] = obs_dict["obs"]
            dones[step] = next_done
            if should_use_ray_obs:
                obs_ray[step] = obs_dict["ray_distance"]
                obs_object_velocity[step] = obs_dict["object_velocity"]
            if should_use_image_obs:
                obs_images[step] = obs_dict["obs_image"]
            if args.asymmetric_observations:
                states[step] = obs_dict["states"]
            # ALGO LOGIC: action logic
            with torch.no_grad():
                if should_use_ray_obs:
                    action_mean, action, logprob, _, value, _ = agent.get_action_and_value(obs_dict)
                else:
                    action_mean, action, logprob, _, value = agent.get_action_and_value(obs_dict)
                values[step] = value.flatten()
            actions[step] = action
            # action_means[step] = action_mean
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_dict, rewards[step], next_done, info = envs.step(action)
            obs_dict = next_obs_dict
            episode_returns[step] = info["r"]
            episode_lengths[step] = info["l"]
            next_dones[step] = next_done


        done_count = next_dones.count_nonzero()
        if done_count:
            mean_episode_return = (episode_returns[next_dones].sum()/done_count)
            mean_episode_length = (episode_lengths[next_dones].sum()/done_count)
        

        if iteration % args.checkpoint_save_frquency == 0:
            if mean_episode_return > max_mean_episode_return:
                max_mean_episode_return = mean_episode_return
                save_checkpoint(f"{run_dir}/{args.wandb_run_name}.pt")
            # else:
            #     save_checkpoint(f"{run_dir}/{args.wandb_run_name}_iter_{iteration:07d}_return_{mean_episode_return.item():.1f}.pt")
            save_checkpoint(f"{run_dir}/{args.wandb_run_name}_newest.pt")

        # agent.train(mode=True)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(obs_dict).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = ~next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = ~dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.flatten(start_dim=0, end_dim=1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.flatten(start_dim=0, end_dim=1)
        # b_action_means = action_means.flatten(start_dim=0, end_dim=1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if should_use_ray_obs:
            b_obs_ray = obs_ray.flatten(start_dim=0, end_dim=1)
            b_obs_object_velocity = obs_object_velocity.flatten(start_dim=0, end_dim=1)
        if should_use_image_obs:
            b_obs_images = obs_images.flatten(start_dim=0, end_dim=1)
        if args.asymmetric_observations:
            b_states = states.flatten(start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Runs the forward pass with autocasting.
                # with torch.autocast(device_type=args.rl_device, enabled=True, dtype=torch.float16):
                b_obs_dict = {"obs": b_obs[mb_inds]}
                if should_use_ray_obs:
                    b_obs_dict["ray_distance"] = b_obs_ray[mb_inds]
                    b_obs_dict["object_velocity"] = b_obs_object_velocity[mb_inds]
                if should_use_image_obs:
                    b_obs_dict["obs_image"] = b_obs_images[mb_inds]
                if args.asymmetric_observations:
                    b_obs_dict["states"] = b_states[mb_inds]

                if should_use_ray_obs:
                    mu, _, newlogprob, entropy, newvalue,latent_loss = agent.get_action_and_value(b_obs_dict, b_actions[mb_inds])
                else:
                    mu, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_dict, b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                b_loss  = bound_loss(mu)

                if should_use_ray_obs:
                    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + latent_loss
                    loss = latent_loss
                    print(f"Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}, ")
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  + b_loss * args.bounds_loss_coef


                optimizer.zero_grad()

                loss.backward()

                if not should_use_ray_obs:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                optimizer.step()

            if args.anneal_lr and should_use_ray_obs != True:
                optimizer.param_groups[0]["lr"] = lr_scheduler.update(optimizer.param_groups[0]["lr"], approx_kl)

            if should_use_ray_obs:
                lr_scheduler.step()
            


        if iteration % args.log_interval == 0:
            # statatics
            print(f"\n step [{global_step}],SPS={int(global_step / (time.time() - start_time)):5d}, mean_episode_return={mean_episode_return.item():.2f}, mean_episode_length={mean_episode_length.item():.1f}")
            writer.add_scalar("charts/episodic_return", mean_episode_return, global_step)
            writer.add_scalar("charts/episodic_length", mean_episode_length, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/bound_loss", b_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            if "episode" in info:
                extra = info["episode"]
                for k, v in extra.items():
                    if 'raw' not in k:
                        writer.add_scalar(f"episode/{k}", v, global_step)

    # envs.close()
    writer.close()
