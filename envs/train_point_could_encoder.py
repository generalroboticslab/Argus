import os
import time
import datetime
import random
import numpy as np
from dataclasses import dataclass,field
from tqdm import tqdm
import tyro
from typing import List
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_utils import Agent
from torch_utils import ImageAgent, Agent, MixedAgent,ActiveAgent, RayEncoder, RayEncoder_pointnet
from natsort import natsorted
import matplotlib.pyplot as plt

def plot_loss_curves(train_loss_history, val_loss_history, epoch_history, val_epoch_history, run_dir, show_plot=False):
    """
    Plot and save loss curves for both training and validation
    
    Args:
        train_loss_history: List of training loss values
        val_loss_history: List of validation loss values
        epoch_history: List of corresponding epochs for training
        val_epoch_history: List of corresponding epochs for validation
        run_dir: Directory to save plots
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Full loss curves
    plt.subplot(2, 3, 1)
    plt.plot(epoch_history, train_loss_history, 'b-', linewidth=1, alpha=0.7, label='Train')
    if val_loss_history and val_epoch_history:
        plt.plot(val_epoch_history, val_loss_history, 'r-', linewidth=1, alpha=0.7, label='Validation')
    plt.title('MSE Loss Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Loss curves with moving average
    plt.subplot(2, 3, 2)
    plt.plot(epoch_history, train_loss_history, 'b-', linewidth=1, alpha=0.3, label='Train Raw')
    if val_loss_history and val_epoch_history:
        plt.plot(val_epoch_history, val_loss_history, 'r-', linewidth=1, alpha=0.3, label='Val Raw')
    
    if len(train_loss_history) > 20:
        # Calculate moving average
        window_size = min(50, len(train_loss_history) // 10)
        train_moving_avg = np.convolve(train_loss_history, np.ones(window_size)/window_size, mode='valid')
        moving_avg_epochs = epoch_history[window_size-1:]
        plt.plot(moving_avg_epochs, train_moving_avg, 'b-', linewidth=2, label=f'Train MA (w={window_size})')
        
        if val_loss_history and len(val_loss_history) > 20:
            val_moving_avg = np.convolve(val_loss_history, np.ones(window_size)/window_size, mode='valid')
            val_moving_avg_epochs = val_epoch_history[window_size-1:]
            plt.plot(val_moving_avg_epochs, val_moving_avg, 'r-', linewidth=2, label=f'Val MA (w={window_size})')
    
    plt.title('MSE Loss with Moving Average')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Recent loss (last 25% of training)
    plt.subplot(2, 3, 3)
    recent_start = max(0, len(train_loss_history) - len(train_loss_history) // 4)
    recent_train_loss = train_loss_history[recent_start:]
    recent_epochs = epoch_history[recent_start:]
    plt.plot(recent_epochs, recent_train_loss, 'b-', linewidth=1.5, label='Train')
    
    if val_loss_history and val_epoch_history:
        # Find validation points in the recent range
        recent_val_indices = [i for i, epoch in enumerate(val_epoch_history) if epoch >= epoch_history[recent_start]]
        if recent_val_indices:
            recent_val_loss = [val_loss_history[i] for i in recent_val_indices]
            recent_val_epochs = [val_epoch_history[i] for i in recent_val_indices]
            plt.plot(recent_val_epochs, recent_val_loss, 'r-', linewidth=1.5, label='Validation')
    
    plt.title('Recent MSE Loss (Last 25%)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Training loss distribution
    plt.subplot(2, 3, 4)
    plt.hist(train_loss_history, bins=50, alpha=0.7, edgecolor='black', label='Train')
    plt.title('Distribution of Training Loss Values')
    plt.xlabel('MSE Loss')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Validation loss distribution
    if val_loss_history:
        plt.subplot(2, 3, 5)
        plt.hist(val_loss_history, bins=50, alpha=0.7, edgecolor='black', color='red', label='Validation')
        plt.title('Distribution of Validation Loss Values')
        plt.xlabel('MSE Loss')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Training vs Validation comparison
    if val_loss_history and len(val_loss_history) > 1:
        plt.subplot(2, 3, 6)
        # Create corresponding training losses for validation epochs
        train_losses_at_val_epochs = []
        for val_epoch in val_epoch_history:
            # Find the closest training epoch
            closest_train_idx = min(range(len(epoch_history)), 
                                  key=lambda i: abs(epoch_history[i] - val_epoch))
            train_losses_at_val_epochs.append(train_loss_history[closest_train_idx])
        
        plt.scatter(train_losses_at_val_epochs, val_loss_history, alpha=0.5)
        min_loss = min(min(train_losses_at_val_epochs), min(val_loss_history))
        max_loss = max(max(train_losses_at_val_epochs), max(val_loss_history))
        plt.plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.7, label='Perfect correlation')
        plt.xlabel('Training Loss')
        plt.ylabel('Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(run_dir, f'loss_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plot_path

class ArgusDataset(Dataset):
    def __init__(self, file_names: List, device='cuda:0'):
        """
        Custom dataset optimized for speed.
        
        Args:
            file_names (List): List of file paths to .pt files.
            device (str): Device to load tensors on ('cpu' or 'cuda').
        """
        self.file_names = file_names
        self.device = device

        self.data = None
        tmp_data = torch.load(self.file_names[0], weights_only=True)
        self.shapes = {
            k: v.shape[2:] for k, v in tmp_data.items() # first 2 are batches and num_envs
        }

    def _load_file_to_device(self, file_path):
        """
        Load a single .pt file and move the necessary tensors to the target device.
        """
        data = torch.load(file_path, weights_only=True)
        return {
            "obs": data["obs"].to(self.device, non_blocking=True),
            "ray_point_cloud": data["ray_point_cloud"].to(self.device, non_blocking=True),
            "object_velocity": data["object_velocity"].to(self.device, non_blocking=True),
            "ray_distance": data["ray_distance"].to(self.device, non_blocking=True),
            "object_orientation": data["object_orientation"].to(self.device, non_blocking=True),
        }

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        data = self._load_file_to_device(file_path)
        return data

def split_files_train_val(data_dirs: List, total_num_of_file,train_ratio=0.8, seed=42):
    """
    Split files from data directories into training and validation sets.
    
    Args:
        data_dirs: List of data directories
        train_ratio: Ratio of files to use for training (default: 0.8)
        seed: Random seed for reproducible splits
    
    Returns:
        train_files, val_files: Lists of file paths for training and validation
    """
    # Collect all files
    all_files = []
    for data_dir in data_dirs:
        files = natsorted([os.path.join(data_dir, f) for f in natsorted(os.listdir(data_dir))[:-1] if f.endswith('.pt')])
        all_files.extend(files)

    all_files = all_files[:total_num_of_file]
    # Shuffle files with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(all_files)
    
    # Split into train and validation
    n_train = int(len(all_files) * train_ratio)
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    return train_files, val_files

def validate_model(agent, val_dataloader, loss_fn, device, args):
    """
    Validate the model on validation set.
    
    Args:
        agent: The model to validate
        val_dataloader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        args: Arguments containing model configuration
    
    Returns:
        avg_val_loss: Average validation loss
    """
    agent.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            # Apply the same preprocessing as training
            if args.num_perception_units == 12:
                for i in range(3):
                    start_idx = i * 800
                    end_idx = start_idx + 800
                    data['ray_point_cloud'][..., start_idx:end_idx, :][...,:500,:] = 0
            elif args.num_perception_units == 20:
                for i in range(3):
                    start_idx = i * 800
                    end_idx = start_idx + 800
                    data['ray_point_cloud'][..., start_idx:end_idx, :][...,500:,:] = 0
                    
            # Calculate loss
            if args.task == "object_tracking":
                predicted_vel = agent.get_object_velocity_prediction_multi_gpu(data['ray_point_cloud'])
                zeros = torch.zeros(*predicted_vel.shape[:-1], 1, device=predicted_vel.device)
                predicted_vel = torch.cat([predicted_vel, zeros], dim=-1)
                val_loss = loss_fn(predicted_vel, data['object_velocity'])
            elif args.task == "object_pushing":
                data['ray_point_cloud'] = data['ray_point_cloud'][..., -500:,:]
                batch_size, num_views, num_rays, num_points, coord_dim = data['ray_point_cloud'].shape
                num_to_zero = int(num_points * 0.36)  # 180 points out of 500

                modified_data = data['ray_point_cloud'].clone()
                # Iterate through each batch, view, and ray
                for b in range(batch_size):
                    for v in range(num_views):
                        for r in range(num_rays):
                            # Randomly select indices to zero out for this specific ray
                            indices_to_zero = torch.randperm(num_points)[:num_to_zero]
                            modified_data[b, v, r, indices_to_zero, :] = 0
                # Update the data
                data['ray_point_cloud'] = modified_data
                prediction = agent.get_object_pushing_obs_prediction_multi_gpu(data['ray_point_cloud']).squeeze(-1)
                gt = quaternion_to_yaw(data['object_orientation'])
                val_loss = loss_fn(prediction, gt)
            total_val_loss += val_loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
    agent.train()  # Set back to training mode
    return avg_val_loss


def quaternion_to_yaw(q):
    """
    Convert quaternion to yaw angle (in radians)
    
    Args:
        q: quaternion tensor with shape (..., 4) in format [x, y, z, w]
           Can be single quaternion or batch of quaternions
    
    Returns:
        yaw angle(s) in radians with shape (...)
    """
    # Extract components
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Extract yaw from quaternion using atan2
    yaw = torch.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z)
    )
    return yaw

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
    device: str = "cuda:0"
    """device to train on, cpu or cuda:0, cuda:1, etc."""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    agent_name: str = "baseline"
    """the agent to train {"baseline", "image", "mixed"}"""
    learning_rate: float = 0.003
    """the learning rate of the optimizer"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    epochs: int = 4
    """the number of epochs (training steps)"""
    num_perception_units: int = 20
    asymmetric_observations: bool = True
    """asymmetric observations for actor and critic"""
    ref_checkpoint: str = None
    """the reference checkpoint of the baseline RL agent"""
    ref_asymmetric_observations: bool = True
    """asymmetric observations for actor and critic"""
    train_ratio: float = 0.9
    """ratio of data to use for training (rest for validation)"""
    val_frequency: int = 2
    """frequency of validation (every N training steps)"""
    patience: int = 200
    """early stopping patience (number of validation checks without improvement)"""
    task: str = None


if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    Args = tyro.conf.configure(
        tyro.conf.AvoidSubcommands,
        tyro.conf.ConsolidateSubcommandArgs,
        tyro.conf.FlagConversionOff,
        tyro.conf.SuppressFixed
    )(Args)
    args: Args = tyro.cli(Args)
    
    data_dir = 'runs/collect'
    data_list = [
        os.path.join(data_dir, data_path)
        for data_path in natsorted(os.listdir(data_dir))
        if args.task in data_path
    ]
    data_dir = [data_list[-1]]

    now = datetime.datetime.now()
    run_name = f"{args.task}_encoder_{args.agent_name}_{now.strftime('%Y%m%d_%H%M%S')}_{args.num_perception_units}"
    run_dir = f"runs/{run_name}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.empty_cache()

    device = torch.device(args.device)
    
    # Split files into train and validation
    train_files, val_files = split_files_train_val(data_dir, 1000,train_ratio=args.train_ratio, seed=args.seed)
    
    # Create datasets and dataloaders
    train_dataset = ArgusDataset(train_files, device=device)
    val_dataset = ArgusDataset(val_files, device=device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    
    obs_label = "ray_point_cloud"    
    states_label = "states" if args.asymmetric_observations else "obs"
    
    ref_obs_dim = train_dataset.shapes["obs"][0]
    obs_dim = train_dataset.shapes[obs_label][0]
    state_dim = train_dataset.shapes[states_label][0]
    action_dim = train_dataset.shapes["action_mean"][0]
    
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.task == "object_tracking":
        agent = RayEncoder_pointnet(
            num_stacked_obs_frame=1,
            ray_obs_dim=obs_dim,
            prediction_dim=2 
        )
    elif args.task == "object_pushing":
        agent = RayEncoder_pointnet(
                num_stacked_obs_frame=1,
                ray_obs_dim=500,
                prediction_dim=1 
            )

    agent: torch.nn.Module = torch.compile(agent).to(device)

    if args.ref_checkpoint is not None:
        print(f"Loading reference agent from {args.ref_checkpoint}")
        rl_agent = Agent(
            actor_obs_dim=ref_obs_dim,
            critic_obs_dim=state_dim,
            action_dim=action_dim,
            asymmetric_observations=args.asymmetric_observations
        )
        rl_agent: torch.nn.Module = torch.compile(rl_agent).to(device)
        rl_agent.load_state_dict(torch.load(args.ref_checkpoint)["agent"])
        agent.critic.load_state_dict(rl_agent.critic.state_dict())
        agent.normalize_critic_obs.load_state_dict(rl_agent.normalize_critic_obs.state_dict())
        agent.actor_logstd.data.copy_(rl_agent.actor_logstd.data)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0003, last_epoch=-1)

    torch.cuda.empty_cache()

    global_step = 0
    train_loss_history = []
    val_loss_history = []
    epoch_history = []
    val_epoch_history = [] 
    agent.train()

    # Best model tracking
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epochs)):
        for data in train_dataloader:
            if args.task == "object_tracking":
                rotation_matrix = data['obs'][..., -12:-3]
                dof_pos = data['obs'][..., 3:23]
                worldSpaceAngularVelocity = data['obs'][..., :3]

                # Apply preprocessing based on perception units
                if args.num_perception_units == 12:
                    for i in range(3):
                        start_idx = i * 800
                        end_idx = start_idx + 800
                        data['ray_point_cloud'][..., start_idx:end_idx, :][...,:500,:] = 0
                elif args.num_perception_units == 20:
                    for i in range(3):
                        start_idx = i * 800
                        end_idx = start_idx + 800
                        data['ray_point_cloud'][..., start_idx:end_idx, :][...,500:,:] = 0
                
                # Forward pass
                predicted_vel = agent.get_object_velocity_prediction_multi_gpu(data['ray_point_cloud'])
                zeros = torch.zeros(*predicted_vel.shape[:-1], 1, device=predicted_vel.device)
                predicted_vel = torch.cat([predicted_vel, zeros], dim=-1)
                
                # Loss calculation
                loss = loss_fn(predicted_vel, data['object_velocity'])

            elif args.task == "object_pushing":
                data['ray_point_cloud'] = data['ray_point_cloud'][..., -500:,:]
                batch_size, num_views, num_rays, num_points, coord_dim = data['ray_point_cloud'].shape
                num_to_zero = int(num_points * 0.36)  # 180 points out of 500

                modified_data = data['ray_point_cloud'].clone()
                # Iterate through each batch, view, and ray
                for b in range(batch_size):
                    for v in range(num_views):
                        for r in range(num_rays):
                            # Randomly select indices to zero out for this specific ray
                            indices_to_zero = torch.randperm(num_points)[:num_to_zero]
                            modified_data[b, v, r, indices_to_zero, :] = 0
                # Update the data
                data['ray_point_cloud'] = modified_data
                prediction = agent.get_object_pushing_obs_prediction_multi_gpu(data['ray_point_cloud']).squeeze(-1)
                gt = quaternion_to_yaw(data['object_orientation'])
                loss = loss_fn(prediction, gt)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_loss_history.append(loss.cpu().item())
            epoch_history.append(global_step)
            
            # Logging
            writer.add_scalar("losses/train_mse", loss.item(), global_step)
            
            # Validation
            if global_step % args.val_frequency == 0 and global_step > 0:
                avg_val_loss = validate_model(agent, val_dataloader, loss_fn, device, args)
                val_loss_history.append(avg_val_loss)
                val_epoch_history.append(global_step)  # Add this line
                
                writer.add_scalar("losses/val_mse", avg_val_loss, global_step)
                
                print(f"Epoch {epoch}/{args.epochs} | Step {global_step} | Train Loss: {loss.item():.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
                
                # Save best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    checkpoint_path = os.path.abspath(f"{run_dir}/imitation_model_best_val.pt")
                    torch.save({
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "train_loss": loss.item(),
                        },
                        checkpoint_path,
                    )
                    print(f"New best validation loss: {avg_val_loss:.6f} - saving checkpoint {checkpoint_path}")
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {patience_counter} validation checks without improvement")
                    break
            else:
                print(f"Epoch {epoch}/{args.epochs} | Step {global_step} | Train Loss: {loss.item():.6f} | LR: {current_lr:.6f}")

            global_step += data['ray_point_cloud'].shape[0]

            # Save model based on training loss (keep original functionality)
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                checkpoint_path = os.path.abspath(f"{run_dir}/imitation_model_best.pt")
                torch.save({
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_train_loss": best_train_loss,
                    },
                    checkpoint_path,
                )
                print(f"New best training loss: {best_train_loss:.6f} - saving checkpoint {checkpoint_path}")
            
            # Periodic checkpoint saving
            if global_step % 200 == 0:
                checkpoint_path = os.path.abspath(f"{run_dir}/imitation_model_{epoch}_{global_step}_{loss.item():.6f}.pt")
                torch.save({
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    },
                    checkpoint_path,
                )
                print(f"Periodic checkpoint saved: {checkpoint_path}")
            
            # Update loss curves plot
            if global_step % 10 == 0:  # Update plot less frequently to save time
                plot_path = plot_loss_curves(train_loss_history, val_loss_history, epoch_history, val_epoch_history, run_dir, show_plot=False)
            if global_step > 600:
                break        
        # Early stopping check at epoch level
        if patience_counter >= args.patience:
            break


    # Final validation
    final_val_loss = validate_model(agent, val_dataloader, loss_fn, device, args)
    print(f"Final validation loss: {final_val_loss:.6f}")
    writer.add_scalar("losses/final_val_mse", final_val_loss, global_step)

    # Save final model
    final_checkpoint_path = os.path.abspath(f"{run_dir}/imitation_model_final.pt")
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "final_val_loss": final_val_loss,
        },
        final_checkpoint_path,
    )
    print(f"Final model saved: {final_checkpoint_path}")

    # Final plot
    plot_path = plot_loss_curves(train_loss_history, val_loss_history, epoch_history, val_epoch_history, run_dir, show_plot=False)
    writer.close()