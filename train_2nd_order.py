"""
Training script for SecondOrderFlowPolicy.

Trains 2nd-order Flow Matching on (q, q_dot) state space for Franka Panda.

Usage:
    python train_2nd_order.py --config config/franka_gf_fm.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
import time
from tqdm import tqdm

# Add paths (keep local package resolution ahead of shared flowpolicy_curobo)
gf_fm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gf_fm_dir)
sys.path.append(os.path.join(gf_fm_dir, '..', 'flowpolicy_curobo'))

from policy.second_order_flow_policy import SecondOrderFlowPolicy
from data.second_order_dataset import SecondOrderFlowDataset, get_shape_meta_from_dataset



def parse_args():
    parser = argparse.ArgumentParser(description="Train SecondOrderFlowPolicy")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--data_path', type=str, default=None, help="Override dataset path")
    parser.add_argument('--log_dir', type=str, default=None, help="Override log directory")
    parser.add_argument('--device', type=str, default=None, help="Override device")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--wandb_project', type=str, default='gf-fm', help="W&B project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument('--wandb_name', type=str, default=None, help="W&B run name")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_2nd_order_normalization_stats(dataset, mode='limits', output_max=1.0, output_min=-1.0):
    """Compute normalization statistics for 2nd-order dataset.

    Args:
        dataset: SecondOrderFlowDataset instance
        mode: 'limits' or 'gaussian'
        output_max: Max value for normalized output
        output_min: Min value for normalized output

    Returns:
        LinearNormalizer with stats for state, action, and observations
    """
    from model.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

    normalizer = LinearNormalizer()
    stats = dataset.get_normalizer_stats()

    # Process state (used if needed)
    if 'state' in stats:
        state_min = stats['state']['min']
        state_max = stats['state']['max']
        state_range = torch.stack([state_min, state_max], dim=0)
        normalizer['state'] = SingleFieldLinearNormalizer.create_fit(
            state_range,
            output_min=output_min,
            output_max=output_max
        )

    # Process action (velocity field)
    if 'action' in stats:
        action_min = stats['action']['min']
        action_max = stats['action']['max']
        action_range = torch.stack([action_min, action_max], dim=0)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            action_range,
            output_min=output_min,
            output_max=output_max
        )

    # Process observations
    for key in stats:
        if key not in ['state', 'action']:
            obs_min = stats[key]['min']
            obs_max = stats[key]['max']
            obs_range = torch.stack([obs_min, obs_max], dim=0)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                obs_range,
                output_min=output_min,
                output_max=output_max
            )

    return normalizer


def save_checkpoint(
    policy: torch.nn.Module,
    normalizer,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    config: dict,
    shape_meta: dict,
    save_path: str,
    training_stats: dict | None = None,
    task: str | None = None,
) -> None:
    """Save GF-FM checkpoint for training/inference."""
    checkpoint = {
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "shape_meta": shape_meta,
    }
    if task is not None:
        checkpoint["task"] = task
    if training_stats is not None:
        checkpoint["training_stats"] = training_stats

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[save_checkpoint] Checkpoint saved to: {save_path}")


def train(config: dict, args: argparse.Namespace):
    """Main training loop."""
    # Override config with CLI args
    if args.data_path is not None:
        config['data_path'] = args.data_path
    if args.log_dir is not None:
        config['training']['log_dir'] = args.log_dir
    if args.device is not None:
        config['training']['device'] = args.device

    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        cprint("CUDA not available, using CPU", "yellow")
        device = 'cpu'

    # Set seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cprint(f"\n{'='*60}", "cyan")
    cprint(f"Training SecondOrderFlowPolicy (GF-FM)", "cyan")
    cprint(f"Task: {config['task']}", "cyan")
    cprint(f"{'='*60}\n", "cyan")

    # Create directories
    log_dir = config['training']['log_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    # Weights & Biases (optional)
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb_name = args.wandb_name or f"gf-fm_{config['task']}_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_name,
                config={
                    **config,
                    'data_path': config['data_path'],
                    'log_dir': log_dir,
                    'device': device,
                    'seed': seed,
                },
                dir=log_dir,
            )
            # Watch model (optional - logs gradients and parameters)
            # wandb.watch(policy, log='all', log_freq=1000)
            cprint(f"  W&B: Initialized run '{wandb_name}'", "cyan")
        except ImportError:
            cprint("  W&B: wandb not installed, skipping. Install with: pip install wandb", "yellow")
            use_wandb = False
    else:
        cprint("  W&B: Disabled (use --wandb to enable)", "cyan")

    # Load dataset
    cprint(f"[1/6] Loading 2nd-order dataset: {config['data_path']}", "green")

    obs_keys = config.get('obs_keys', ['eef_pos', 'eef_quat', 'obstacle_pos'])
    joint_dim = config['model'].get('joint_dim', 7)

    train_dataset = SecondOrderFlowDataset(
        dataset_path=config['data_path'],
        horizon=config['model']['horizon'],
        n_obs_steps=config['model']['n_obs_steps'],
        n_action_steps=config['model']['n_action_steps'],
        obs_keys=obs_keys,
        joint_dim=joint_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device == 'cuda' else False
    )

    cprint(f"  Dataset size: {len(train_dataset)} sequences", "cyan")
    cprint(f"  State dim: {train_dataset.state_dim}", "cyan")
    cprint(f"  Action dim: {train_dataset.action_dim}", "cyan")

    # Get shape metadata
    cprint(f"\n[2/6] Extracting shape metadata", "green")
    shape_meta = get_shape_meta_from_dataset(
        config['data_path'],
        obs_keys=['joint_pos', 'joint_vel'] + obs_keys,
        joint_dim=joint_dim
    )
    cprint(f"  State shape: {shape_meta['state']['shape']}", "cyan")
    cprint(f"  Action shape: {shape_meta['action']['shape']}", "cyan")

    # Compute normalization
    cprint(f"\n[3/6] Computing normalization statistics", "green")
    normalizer = compute_2nd_order_normalization_stats(
        dataset=train_dataset,
        mode=config['normalization'].get('mode', 'limits'),
        output_max=config['normalization'].get('output_max', 1.0),
        output_min=config['normalization'].get('output_min', -1.0)
    )

    # Create policy
    cprint(f"\n[4/6] Creating SecondOrderFlowPolicy", "green")

    model_config = {
        'shape_meta': shape_meta,
        'horizon': config['model']['horizon'],
        'n_obs_steps': config['model']['n_obs_steps'],
        'n_action_steps': config['model']['n_action_steps'],
        'joint_dim': joint_dim,
        'encoder_output_dim': config['model'].get('encoder_output_dim', 256),
        'diffusion_step_embed_dim': config['model'].get('diffusion_step_embed_dim', 256),
        'down_dims': tuple(config['model'].get('down_dims', [256, 512, 1024])),
        'kernel_size': config['model'].get('kernel_size', 5),
        'n_groups': config['model'].get('n_groups', 8),
        'condition_type': config['model'].get('condition_type', 'film'),
        'use_down_condition': config['model'].get('use_down_condition', True),
        'use_mid_condition': config['model'].get('use_mid_condition', True),
        'use_up_condition': config['model'].get('use_up_condition', True),
        'obs_as_global_cond': config['model'].get('obs_as_global_cond', True),
        'Conditional_ConsistencyFM': config.get('cfm', None),
        'eta': config.get('cfm', {}).get('eta', 0.01)
    }

    policy = SecondOrderFlowPolicy(**model_config)
    policy.set_normalizer(normalizer)
    policy.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    # Print summary
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    cprint(f"  Total parameters: {total_params:,}", "cyan")
    cprint(f"  Trainable parameters: {trainable_params:,}", "cyan")

    # Training loop
    cprint(f"\n[5/6] Starting training", "green")
    cprint(f"  Epochs: {config['training']['num_epochs']}", "cyan")
    cprint(f"  Learning rate: {config['training']['lr']}", "cyan")
    cprint(f"  Device: {device}", "cyan")

    policy.train()
    global_step = 0
    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(config['training']['num_epochs']):
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_consistency_losses = []
        epoch_velocity_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {
                'obs': {k: v.to(device) for k, v in batch['obs'].items()},
                'state': batch['state'].to(device),
                'action': batch['action'].to(device)
            }

            # Forward and loss
            loss, loss_dict = policy.compute_loss(batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record
            epoch_losses.append(loss.item())
            epoch_consistency_losses.append(loss_dict.get('consistency_loss', 0))
            epoch_velocity_losses.append(loss_dict.get('velocity_loss', 0))

            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

            # Log
            if global_step % config['training'].get('log_freq', 100) == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Loss/{key}', value, global_step)

                # W&B logging
                if use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/step': global_step,
                        **{f'train/{key}': value for key, value in loss_dict.items()}
                    }, step=global_step)

            global_step += 1

        # Epoch stats
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start_time

        cprint(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
               f"Loss = {epoch_loss:.6f}, Time = {epoch_time:.2f}s", "green")

        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)

        # W&B epoch logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/loss': epoch_loss,
                'epoch/time': epoch_time,
                'epoch/lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

        # Save checkpoint
        save_freq = config['training'].get('save_freq', 50)
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == config['training']['num_epochs']:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:04d}.pth")
            save_checkpoint(
                policy=policy,
                normalizer=normalizer,
                epoch=epoch,
                optimizer=optimizer,
                config=model_config,
                shape_meta=shape_meta,
                save_path=checkpoint_path,
                training_stats={'epoch_loss': epoch_loss},
                task=config['task']
            )

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint(
                policy=policy,
                normalizer=normalizer,
                epoch=epoch,
                optimizer=optimizer,
                config=model_config,
                shape_meta=shape_meta,
                save_path=best_path,
                training_stats={'epoch_loss': epoch_loss, 'best_epoch': epoch},
                task=config['task']
            )
            cprint(f"  → New best model! Loss: {best_loss:.6f} (Epoch {epoch+1})", "yellow")

            # W&B: Log best model
            if use_wandb and wandb.run is not None:
                wandb.run.summary['best_loss'] = best_loss
                wandb.run.summary['best_epoch'] = epoch + 1
                
                # Log model as Artifact
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", 
                    type="model",
                    description=f"Best model (Loss: {best_loss:.6f})",
                    metadata=config
                )
                artifact.add_file(best_path)
                wandb.log_artifact(artifact, aliases=["best", "latest"])

    # Done
    cprint(f"\n[6/6] Training complete!", "green")
    cprint(f"  Best model: {best_path} (Epoch {best_epoch+1}, Loss: {best_loss:.6f})", "cyan")
    cprint(f"  Logs: {log_dir}", "cyan")

    writer.close()

    # Finish W&B run
    if use_wandb:
        wandb.finish()
        cprint("  W&B: Run finished", "cyan")


def main():
    args = parse_args()
    cprint(f"Loading config: {args.config}", "yellow")
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
