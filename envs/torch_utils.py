import torch
from torch import nn
from torch.distributions.normal import Normal
import os
import numpy as np
from typing import Tuple
import copy

class RunningMeanStd(nn.Module):
    '''
    updates statistic from a full data
    refer to https://github.com/zplizzi/pytorch-ppo/blob/master/running_mean_std.py
    '''

    def __init__(self, insize, epsilon=1e-05, norm_only=False, clip_limit=5.0):
        super(RunningMeanStd, self).__init__()
        # print('Creating Normalizer RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon
        self.norm_only = norm_only
        self.axis = [0]
        self.clip_min = -clip_limit
        self.clip_max = clip_limit
        in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input: torch.Tensor, denorm=False):
        with torch.no_grad():
            if self.training:
                batch_mean = input.mean(self.axis)
                batch_var = input.var(self.axis)
                batch_count = input.size()[0]
                self.running_mean, self.running_var, self.count = \
                    self.update_mean_var_count_from_moments(
                        self.running_mean, self.running_var, self.count, batch_mean, batch_var, batch_count)

            current_mean = self.running_mean
            current_var = self.running_var
            if denorm:
                y = torch.clamp(input, min=self.clip_min, max=self.clip_max)
                y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
                return y
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=self.clip_min, max=self.clip_max)
            return y


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        asymmetric_observations: bool = False
    ):
        super().__init__()

        self.normalize_obs = RunningMeanStd(actor_obs_dim)

        self.critic_obs_label = "states" if asymmetric_observations else "obs"
        critic_obs_dim = critic_obs_dim if critic_obs_dim else actor_obs_dim
        self.normalize_critic_obs = RunningMeanStd(critic_obs_dim)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(actor_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs_dict):
        x = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation
        return self.critic(x)

    def get_action_and_value(self, obs_dict, action=None):
        x = self.normalize_obs(obs_dict["obs"])
        x_critic = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action_mean,action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_critic)

    def get_action_mean_and_logstd(self, obs_dict):
        x = self.normalize_obs(obs_dict["obs"])  # normalize observation
        action_mean = self.actor_mean(x)
        return action_mean, self.actor_logstd


class CNN(nn.Module):
    def __init__(self, input_shape=(12, 32, 32)):
        super(CNN, self).__init__()
        # Define the layers of the CNN
        # n * 20 *32 *32

        self.base = nn.Sequential(

            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.ELU(),
        )

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        return x

    def calculate_output_shape(self, conv: nn.Conv2d, input_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        # [(W−K+2P)/S]+1.
        shape_1 = (input_shape[1] - conv.kernel_size[0] + 2 * conv.padding[0]) // conv.stride[0] + 1
        shape_2 = (input_shape[2] - conv.kernel_size[1] + 2 * conv.padding[1]) // conv.stride[1] + 1
        return conv.out_channels, shape_1, shape_2


class ImageAgent(nn.Module):
    def __init__(
        self,
        actor_image_obs_dim: tuple,
        critic_obs_dim: int,
        action_dim: int,
        asymmetric_observations: bool = False
        # image_dims: Tuple[int, int, int] = (20, 32, 32),
    ):
        super().__init__()

        self.asymmetric_observations = asymmetric_observations
        self.critic_obs_label = "states" if asymmetric_observations else "obs"
        critic_obs_dim = critic_obs_dim if critic_obs_dim else actor_image_obs_dim
        self.normalize_critic_obs = RunningMeanStd(critic_obs_dim)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.cnn = CNN(actor_image_obs_dim)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs_dict):
        x_critic = self.normalize_critic_obs(obs_dict[self.critic_obs_label])
        return self.critic(x_critic)

    def get_action_and_value(self, obs_dict, action=None):
        x_critic = self.normalize_critic_obs(obs_dict[self.critic_obs_label])
        action_mean = self.actor_mean(self.cnn(obs_dict["obs_image"]))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action_mean, action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_critic)

    def get_action_mean_and_logstd(self, obs_dict):
        # x = self.normalize_obs(x) # normalize observation
        action_mean = self.actor_mean(self.cnn(obs_dict["obs_image"]))
        return action_mean, self.actor_logstd


class MixedAgent(nn.Module):
    def __init__(
        self,
        actor_obs_dim: int,
        actor_image_obs_dim: Tuple,
        critic_obs_dim: int,
        action_dim: int,
        asymmetric_observations: bool = False
        # image_dims: Tuple[int, int, int] = (20, 32, 32),
    ):
        super().__init__()

        self.normalize_obs = RunningMeanStd(actor_obs_dim)
        self.critic_obs_label = "states" if asymmetric_observations else "obs"
        critic_obs_dim = critic_obs_dim if critic_obs_dim else actor_obs_dim
        self.normalize_critic_obs = RunningMeanStd(critic_obs_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.cnn = CNN(actor_image_obs_dim)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512 + actor_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs_dict):
        x = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation
        return self.critic(x)

    def get_action_and_value(self, obs_dict, action=None):
        x = self.normalize_obs(obs_dict["obs"])  # normalize observation
        x_obs = torch.concat((x, self.cnn(obs_dict["obs_image"])), dim=1)
        x_critic = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation
        action_mean = self.actor_mean(x_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action_mean, action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_critic)

    def get_action_mean_and_logstd(self, obs_dict):
        x = self.normalize_obs(obs_dict["obs"])  # normalize observation
        x_obs = torch.concat((x, self.cnn(obs_dict["obs_image"])), dim=1)
        action_mean = self.actor_mean(x_obs)
        return action_mean, self.actor_logstd


class RayEncoder(nn.Module):
    def __init__(self, ray_obs_dim: int, num_stacked_obs_frame: int):
        super().__init__()

        self.ray_obs_dim = ray_obs_dim
        self.normalize_obs = RunningMeanStd(ray_obs_dim)

        self.ray_encoder_1 = nn.Sequential(
            layer_init(nn.Linear(int(ray_obs_dim/3), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 64)),
        )

        self.ray_encoder_2 = nn.Sequential(
            layer_init(nn.Linear(int(ray_obs_dim/3), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 64)),
        )

        self.ray_encoder_3 =nn.Sequential(
            layer_init(nn.Linear(int(ray_obs_dim/3), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 64)),
        )
        self.ray_encoder = nn.Sequential(
            layer_init(nn.Linear(int(64*3+9+20+3), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 2*num_stacked_obs_frame)),
            layer_init(nn.Linear(256, 2*num_stacked_obs_frame)),
        )
        
    def get_object_velocity_prediction(self, worldSpaceAngularVelocity, dof_position, rotation_matrix,ray_point_cloud):

        feature_1 = self.ray_encoder_1(ray_point_cloud[..., 0:int(self.ray_obs_dim/3)])
        feature_2 = self.ray_encoder_2(ray_point_cloud[..., int(self.ray_obs_dim/3):int(2*self.ray_obs_dim/3)])
        feature_3 = self.ray_encoder_3(ray_point_cloud[..., int(2*self.ray_obs_dim/3):])
        feature = torch.cat((worldSpaceAngularVelocity, dof_position,rotation_matrix,
                             feature_1[:,:,0,:,:], feature_2[:,:,0,:,:], feature_3[:,:,0,:,:]), dim=-1)
        return self.ray_encoder(feature)


class RayEncoder_pointnet_mini(nn.Module):
    def __init__(self, ray_obs_dim: int, num_stacked_obs_frame: int):
        super().__init__()
        self.ray_obs_dim = ray_obs_dim
        self.normalize_obs = RunningMeanStd(ray_obs_dim)
        self.latent_dim = 512
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.latent_dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2*num_stacked_obs_frame)
        )
    def get_object_velocity_prediction(self, worldSpaceAngularVelocity, dof_position, rotation_matrix, ray_point_cloud):
        B, env, step, N, coords = ray_point_cloud.shape
        # Process in chunks to avoid OOM
        chunk_size = 128  # Adjust based on your GPU memory
        outputs = []
        for i in range(0, B*env*step, chunk_size):
            end_idx = min(i + chunk_size, B*env*step)
            # Process chunk
            chunk_points = ray_point_cloud.view(B*env*step, N, coords)[i:end_idx]
            chunk_points = chunk_points.transpose(1, 2)
            # chunk_angular = worldSpaceAngularVelocity.view(B*env*step, -1)[i:end_idx]
            # chunk_dof = dof_position.view(B*env*step, -1)[i:end_idx]
            # chunk_rotation = rotation_matrix.view(B*env*step, -1)[i:end_idx]
            # Forward pass on chunk
            points_feature = self.first_conv(chunk_points)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
            points_feature = torch.cat([
                points_feature_global.expand(-1, -1, N),
                points_feature
            ], dim=1)
            points_feature = self.second_conv(points_feature)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
            # feature = torch.cat([chunk_angular, chunk_dof, chunk_rotation, points_feature_global], dim=-1)
            chunk_output = self.mlp(points_feature_global)
            outputs.append(chunk_output)
        output = torch.cat(outputs, dim=0)
        return output.view(B, env, step, -1)
    def get_object_velocity_prediction_play(self,ray_point_cloud):
        env, N, coords = ray_point_cloud.shape
        # Process chunk
        ray_point_cloud = ray_point_cloud.transpose(1, 2)
        # Forward pass on chunk
        points_feature = self.first_conv(ray_point_cloud)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
        points_feature = torch.cat([
            points_feature_global.expand(-1, -1, N),
            points_feature
        ], dim=1)
        points_feature = self.second_conv(points_feature)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
        # feature = torch.cat([worldSpaceAngularVelocity, dof_position, rotation_matrix, points_feature_global], dim=-1)
        output = self.mlp(points_feature_global)
        return output.view(env, -1)

class RayEncoder_pointnet(nn.Module):
    def __init__(self, ray_obs_dim: int, num_stacked_obs_frame: int, prediction_dim: int):
        super().__init__()
        self.ray_obs_dim = ray_obs_dim
        self.normalize_obs = RunningMeanStd(ray_obs_dim)
        self.latent_dim = 1024
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, prediction_dim*num_stacked_obs_frame)
        )

    def get_object_velocity_prediction(self,ray_point_cloud):

        B, env, step, N, coords = ray_point_cloud.shape
        
        # Process in chunks to avoid OOM
        chunk_size = 1  # Adjust based on your GPU memory
        outputs = []
        # import ipdb;ipdb.set_trace()
        for i in range(0, B*env*step, chunk_size):
            end_idx = min(i + chunk_size, B*env*step)
            
            # Process chunk
            chunk_points = ray_point_cloud.view(B*env*step, N, coords)[i:end_idx]
            chunk_points = chunk_points.transpose(1, 2)
            
            # chunk_angular = worldSpaceAngularVelocity.view(B*env*step, -1)[i:end_idx]
            # chunk_dof = dof_position.view(B*env*step, -1)[i:end_idx]  
            # chunk_rotation = rotation_matrix.view(B*env*step, -1)[i:end_idx]
            
            # Forward pass on chunk
            points_feature = self.first_conv(chunk_points)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
            points_feature = torch.cat([
                points_feature_global.expand(-1, -1, N), 
                points_feature
            ], dim=1)
            points_feature = self.second_conv(points_feature)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
            
            # feature = torch.cat([chunk_angular, chunk_dof, chunk_rotation, points_feature_global], dim=-1)
            # feature = torch.cat([points_feature_global], dim=-1)
            chunk_output = self.mlp(points_feature_global)
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=0)
        return output.view(B, env, step, -1)

    def get_object_velocity_prediction_multi_gpu(self, ray_point_cloud):
        B, env, step, N, coords = ray_point_cloud.shape
        
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            return self.get_object_velocity_prediction(ray_point_cloud)
        
        # Keep the main model on cuda:0
        main_device = torch.device('cuda:0')
        self.to(main_device)
        
        # Split data across GPUs
        total_samples = B * env * step
        samples_per_gpu = total_samples // num_gpus
        
        # Create model replicas that share weights (important!)
        replica_models = []
        for gpu_id in range(num_gpus):
            if gpu_id == 0:
                replica_models.append(self)  # Main model stays on cuda:0
            else:
                # Create replica on other GPU but keep weight sharing
                replica = copy.deepcopy(self).cuda(gpu_id)
                # Copy current weights from main model
                replica.load_state_dict(self.state_dict())
                replica_models.append(replica)
        
        # Process chunks on different GPUs
        results = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * samples_per_gpu
            if gpu_id == num_gpus - 1:
                end_idx = total_samples
            else:
                end_idx = (gpu_id + 1) * samples_per_gpu
            
            # Get data slice
            gpu_data = ray_point_cloud.view(total_samples, N, coords)[start_idx:end_idx]
            
            if gpu_id == 0:
                # Main GPU - keep gradients
                gpu_data = gpu_data.to(main_device)
                gpu_result = replica_models[gpu_id].process_chunk_single_gpu(gpu_data)
                results.append(gpu_result)
            else:
                # Other GPUs - detach to avoid gradient issues, then reattach
                gpu_data = gpu_data.cuda(gpu_id)
                with torch.no_grad():
                    gpu_result = replica_models[gpu_id].process_chunk_single_gpu(gpu_data)
                
                # Move result back to main GPU and reattach to graph
                gpu_result = gpu_result.to(main_device)
                # Re-enable gradients by creating new tensor that requires grad
                gpu_result = gpu_result.detach().requires_grad_(True)
                results.append(gpu_result)
        
        # Concatenate results on main GPU
        output = torch.cat(results, dim=0)
        return output.view(B, env, step, -1)

    def get_object_pushing_obs_prediction_multi_gpu(self, ray_point_cloud):
        B, env, step, N, coords = ray_point_cloud.shape
        
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            return self.get_object_velocity_prediction(ray_point_cloud)
        
        # Keep the main model on cuda:0
        main_device = torch.device('cuda:0')
        self.to(main_device)
        
        # Split data across GPUs
        total_samples = B * env * step
        samples_per_gpu = total_samples // num_gpus
        
        # Create model replicas that share weights (important!)
        replica_models = []
        for gpu_id in range(num_gpus):
            if gpu_id == 0:
                replica_models.append(self)  # Main model stays on cuda:0
            else:
                # Create replica on other GPU but keep weight sharing
                replica = copy.deepcopy(self).cuda(gpu_id)
                # Copy current weights from main model
                replica.load_state_dict(self.state_dict())
                replica_models.append(replica)
        
        # Process chunks on different GPUs
        results = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * samples_per_gpu
            if gpu_id == num_gpus - 1:
                end_idx = total_samples
            else:
                end_idx = (gpu_id + 1) * samples_per_gpu
            
            # Get data slice
            gpu_data = ray_point_cloud.view(total_samples, N, coords)[start_idx:end_idx]
            
            if gpu_id == 0:
                # Main GPU - keep gradients
                gpu_data = gpu_data.to(main_device)
                gpu_result = replica_models[gpu_id].process_chunk_single_gpu(gpu_data)
                results.append(gpu_result)
            else:
                # Other GPUs - detach to avoid gradient issues, then reattach
                gpu_data = gpu_data.cuda(gpu_id)
                with torch.no_grad():
                    gpu_result = replica_models[gpu_id].process_chunk_single_gpu(gpu_data)
                
                # Move result back to main GPU and reattach to graph
                gpu_result = gpu_result.to(main_device)
                # Re-enable gradients by creating new tensor that requires grad
                gpu_result = gpu_result.detach().requires_grad_(True)
                results.append(gpu_result)
        
        # Concatenate results on main GPU
        output = torch.cat(results, dim=0)
        return output.view(B, env, step, -1)


    def process_chunk_single_gpu(self, chunk_data):
        # Process the chunk on current GPU
        chunk_data = chunk_data.transpose(1, 2)
        
        points_feature = self.first_conv(chunk_data)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
        points_feature = torch.cat([
            points_feature_global.expand(-1, -1, chunk_data.shape[2]), 
            points_feature
        ], dim=1)
        points_feature = self.second_conv(points_feature)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
        
        return self.mlp(points_feature_global)

    def get_object_velocity_prediction_play(self,ray_point_cloud):
        env, N, coords = ray_point_cloud.shape
    
        # Process chunk
        ray_point_cloud = ray_point_cloud.transpose(1, 2)
        
        
        # Forward pass on chunk
        points_feature = self.first_conv(ray_point_cloud)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
        points_feature = torch.cat([
            points_feature_global.expand(-1, -1, N), 
            points_feature
        ], dim=1)
        points_feature = self.second_conv(points_feature)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
        
        # feature = torch.cat([worldSpaceAngularVelocity, dof_position, rotation_matrix, points_feature_global], dim=-1)
        output = self.mlp(points_feature_global)
    
        return output.view(env, -1)

class RayEncoder_pointnet_mini(nn.Module):
    def __init__(self, ray_obs_dim: int, num_stacked_obs_frame: int):
        super().__init__()
        self.ray_obs_dim = ray_obs_dim
        self.normalize_obs = RunningMeanStd(ray_obs_dim)
        self.latent_dim = 512
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.latent_dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2*num_stacked_obs_frame)
        )


    def get_object_velocity_prediction(self,ray_point_cloud):

        B, env, step, N, coords = ray_point_cloud.shape
        
        # Process in chunks to avoid OOM
        chunk_size = 128  # Adjust based on your GPU memory
        outputs = []
        
        for i in range(0, B*env*step, chunk_size):
            end_idx = min(i + chunk_size, B*env*step)
            
            # Process chunk
            chunk_points = ray_point_cloud.view(B*env*step, N, coords)[i:end_idx]
            chunk_points = chunk_points.transpose(1, 2)
            
            # chunk_angular = worldSpaceAngularVelocity.view(B*env*step, -1)[i:end_idx]
            # chunk_dof = dof_position.view(B*env*step, -1)[i:end_idx]  
            # chunk_rotation = rotation_matrix.view(B*env*step, -1)[i:end_idx]
            
            # Forward pass on chunk
            points_feature = self.first_conv(chunk_points)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
            points_feature = torch.cat([
                points_feature_global.expand(-1, -1, N), 
                points_feature
            ], dim=1)
            points_feature = self.second_conv(points_feature)
            points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
            
            # feature = torch.cat([chunk_angular, chunk_dof, chunk_rotation, points_feature_global], dim=-1)
            chunk_output = self.mlp(points_feature_global)
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=0)
        return output.view(B, env, step, -1)

    def get_object_velocity_prediction_play(self,ray_point_cloud):
        env, N, coords = ray_point_cloud.shape
    
        # Process chunk
        ray_point_cloud = ray_point_cloud.transpose(1, 2)
        
        
        # Forward pass on chunk
        points_feature = self.first_conv(ray_point_cloud)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=True)[0]
        points_feature = torch.cat([
            points_feature_global.expand(-1, -1, N), 
            points_feature
        ], dim=1)
        points_feature = self.second_conv(points_feature)
        points_feature_global = torch.max(points_feature, dim=2, keepdim=False)[0]
        
        # feature = torch.cat([worldSpaceAngularVelocity, dof_position, rotation_matrix, points_feature_global], dim=-1)
        output = self.mlp(points_feature_global)
    
        return output.view(env, -1)
    
class ActiveAgent(nn.Module):
    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        num_stacked_obs_frame: int,
        ray_obs_dim: int,
        action_dim: int,
        asymmetric_observations: bool = False
    ):
        super().__init__()
        self.num_stacked_obs_frame = num_stacked_obs_frame
        actor_obs_dim = actor_obs_dim + 3*num_stacked_obs_frame
        self.normalize_obs = RunningMeanStd(actor_obs_dim)
        self.ray_obs_dim = ray_obs_dim
        self.ray_obs_dim = ray_obs_dim
        self.critic_obs_label = "states" if asymmetric_observations else "obs"
        critic_obs_dim = critic_obs_dim if critic_obs_dim else actor_obs_dim
        self.normalize_critic_obs = RunningMeanStd(critic_obs_dim)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )


        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(actor_obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.ray_encoder = nn.Sequential(
            layer_init(nn.Linear(ray_obs_dim+9, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 3*num_stacked_obs_frame), std=1.0),
        )

        # self.ray_encoder = RayEncoder(ray_obs_dim, num_stacked_obs_frame)

        self.latent_vector_loss = nn.MSELoss()

    def get_value(self, obs_dict):
        x = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation
        return self.critic(x)

    def get_action_and_value(self, obs_dict, action=None):
        x_critic = self.normalize_critic_obs(obs_dict[self.critic_obs_label])  # normalize observation

        actor_obs = obs_dict["obs"].clone()  # Make a copy to avoid modifying the original
        
        ray_point_cloud = obs_dict["ray_point_cloud"]
        object_velocity = obs_dict["object_velocity"]
        rotation_matrix = obs_dict['obs'][..., -12:-3]

        
        object_velocity = object_velocity.repeat(1, self.num_stacked_obs_frame )
        # Process ray point cloud if available
        if ray_point_cloud is not None:
            ray_encoded = self.ray_encoder(torch.cat((rotation_matrix, ray_point_cloud), dim=-1))
            feature = torch.cat((actor_obs, ray_encoded), dim=-1)


        # if ray_point_cloud is not None:
        #     feature_1 = self.ray_encoder.ray_encoder_1(ray_point_cloud[..., 0:int(self.ray_obs_dim/3)])
        #     feature_2 = self.ray_encoder.ray_encoder_2(ray_point_cloud[..., int(self.ray_obs_dim/3):int(2*self.ray_obs_dim/3)])
        #     feature_3 = self.ray_encoder.ray_encoder_3(ray_point_cloud[..., int(2*self.ray_obs_dim/3):])
        #     feature = torch.cat((rotation_matrix,feature_1, feature_2, feature_3), dim=-1)
        #     ray_encoded = self.ray_encoder.ray_encoder(feature)

        #     zeros = torch.zeros(*ray_encoded.shape[:-1], 1, device=ray_encoded.device)  # Shape: (a,b,c,1)
        #     ray_encoded = torch.cat([ray_encoded, zeros], dim=-1)  # Shape: (a,b,c,3)

        #     print('velocity prediction', ray_encoded[..., :])

        #     feature = torch.cat((actor_obs, ray_encoded), dim=1)
            

        latent_loss = ((object_velocity-ray_encoded)**2).mean() 

        x = self.normalize_obs(feature)  # normalize base observation

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()         
        return action_mean, action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_critic), latent_loss

    def get_action_mean_and_logstd(self, obs_dict):
        actor_obs = obs_dict["obs"].clone()  # Make a copy to avoid modifying the original

        ray_point_cloud = obs_dict["ray_point_cloud"]
        rotation_matrix = obs_dict['obs'][..., -12:-3]

        # Process ray point cloud if available
        # if ray_point_cloud is not None:
        #     ray_encoded = self.ray_encoder(ray_point_cloud)
        #     feature = torch.cat((actor_obs, ray_encoded), dim=-1)


        if ray_point_cloud is not None:
            ray_encoded = self.ray_encoder(torch.cat((rotation_matrix, ray_point_cloud), dim=-1))
            feature = torch.cat((actor_obs, ray_encoded), dim=-1)

        # if ray_point_cloud is not None:
        #     rotation_matrix = obs_dict['obs'][..., -12:-3]

        #     feature_1 = self.ray_encoder.ray_encoder_1(ray_point_cloud[..., 0:int(self.ray_obs_dim/3)])
        #     feature_2 = self.ray_encoder.ray_encoder_2(ray_point_cloud[..., int(self.ray_obs_dim/3):int(2*self.ray_obs_dim/3)])
        #     feature_3 = self.ray_encoder.ray_encoder_3(ray_point_cloud[..., int(2*self.ray_obs_dim/3):])
        #     feature = torch.cat((rotation_matrix,feature_1, feature_2, feature_3), dim=-1)
        #     ray_encoded = self.ray_encoder.ray_encoder(feature)
        #     # print('velocity prediction', ray_encoded[..., :])

        #     zeros = torch.zeros(*ray_encoded.shape[:-1], 1, device=ray_encoded.device)  # Shape: (a,b,c,1)
        #     ray_encoded = torch.cat([ray_encoded, zeros], dim=-1)  # Shape: (a,b,c,3)


        #     feature = torch.cat((actor_obs, ray_encoded), dim=1)


        x = self.normalize_obs(feature)  # normalize base observation
        action_mean = self.actor_mean(x)
        return action_mean, self.actor_logstd

    def get_object_velocity_prediction(self, obs_dict):
        ray_point_cloud = obs_dict["ray_point_cloud"]
        ray_encoded = self.ray_encoder(ray_point_cloud)
        return ray_encoded



if __name__ == "__main__":

    image = torch.randn(10, 20, 32, 32)
    cnn = CNN((20, 32, 32))
    print(cnn(image).shape)

    agent = ImageAgent((20, 32, 32), 200, 20)
    agent = torch.compile(agent)
    action_mean, logstd = agent.get_action_mean_and_logstd({"obs_image": image})
    print(action_mean.shape, logstd.shape)

    # for name, param in agent.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} \n")
