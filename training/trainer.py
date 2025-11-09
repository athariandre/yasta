"""
Training script for PPO-based Tron agent.

This script orchestrates the training loop:
- Environment setup
- Rollout collection
- PPO updates (placeholder for PR-3)
- Checkpointing
- Logging
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import config
from training.env import TronEnv
from training.rollout import RolloutCollector
from training.policy import ActorCriticPolicy
from case_closed_game import GameResult


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    PPO trainer for Tron agent.
    
    Manages:
    - Training loop
    - Checkpoint saving
    - Metric logging
    - Self-play coordination
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize trainer.
        
        Args:
            config_obj: Configuration object (uses default if None)
        """
        self.config = config_obj if config_obj else config
        
        # Set seeds for reproducibility
        set_seed(self.config.SEED)
        
        # Initialize policy (Neural network for PR-3)
        self.policy = ActorCriticPolicy(
            board_h=self.config.BOARD_H,
            board_w=self.config.BOARD_W,
            num_actions=self.config.NUM_ACTIONS,
            latent_size=self.config.LATENT_SIZE,
            device=self.config.DEVICE,
            verbose=True  # Only print on first initialization
        )
        
        # Initialize frozen opponent policy for stable self-play
        if self.config.USE_FROZEN_OPPONENT:
            self.opponent_policy = ActorCriticPolicy(
                board_h=self.config.BOARD_H,
                board_w=self.config.BOARD_W,
                num_actions=self.config.NUM_ACTIONS,
                latent_size=self.config.LATENT_SIZE,
                device=self.config.DEVICE,
                verbose=False
            )
            # Copy weights from main policy
            self.opponent_policy.load_state_dict(self.policy.state_dict())
            # Freeze opponent
            for param in self.opponent_policy.parameters():
                param.requires_grad = False
        else:
            self.opponent_policy = self.policy
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Initialize environment with frozen opponent
        self.env = TronEnv(
            opponent_policy=self.opponent_policy,  # Use frozen opponent
            use_heuristic_opponent=False,
            seed=self.config.SEED
        )
        
        # Initialize rollout collector
        self.rollout_collector = RolloutCollector(
            rollout_length=self.config.ROLLOUT_LENGTH
        )
        
        # Training state
        self.total_steps = 0
        self.num_updates = 0
        self.rollouts_since_opponent_update = 0
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_counts = {'win': 0, 'loss': 0, 'draw': 0}
        self.total_episodes = 0
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"[Trainer] Initialized with seed {self.config.SEED}")
        print(f"[Trainer] Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def train(self, max_steps: int = None):
        """
        Main training loop.
        
        Args:
            max_steps: Maximum training steps (uses config default if None)
        """
        if max_steps is None:
            max_steps = self.config.MAX_TRAINING_STEPS
        
        print(f"\n{'='*60}")
        print(f"Starting training for {max_steps} steps")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        while self.total_steps < max_steps:
            # Collect rollout
            rollout_start = time.time()
            rollout_data = self.rollout_collector.collect(
                self.env,
                self.policy,
                num_steps=self.config.ROLLOUT_LENGTH
            )
            rollout_time = time.time() - rollout_start
            
            # Update episode statistics
            self._update_episode_stats(rollout_data['episode_info'])
            
            # Compute advantages (GAE) with bootstrapping
            rollout_data = self.rollout_collector.compute_advantages(rollout_data, policy=self.policy)
            
            # Prepare data for training
            rollout_data = self.rollout_collector.prepare_batch_data(rollout_data)
            
            # Perform PPO update (placeholder for PR-2)
            update_start = time.time()
            self._ppo_update(rollout_data)
            update_time = time.time() - update_start
            
            # Update counters
            self.total_steps += len(rollout_data['actions'])
            self.num_updates += 1
            self.rollouts_since_opponent_update += 1
            
            # Update frozen opponent periodically for stable self-play
            if self.config.USE_FROZEN_OPPONENT and self.rollouts_since_opponent_update >= self.config.OPPONENT_UPDATE_INTERVAL:
                self.opponent_policy.load_state_dict(self.policy.state_dict())
                self.rollouts_since_opponent_update = 0
                print(f"  [Opponent updated at step {self.total_steps}]")
            
            # Logging
            if self.num_updates % (self.config.LOG_INTERVAL // self.config.ROLLOUT_LENGTH) == 0:
                self._log_metrics(rollout_time, update_time)
            
            # Checkpointing
            if self.total_steps % self.config.CHECKPOINT_INTERVAL < self.config.ROLLOUT_LENGTH:
                self._save_checkpoint()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total steps: {self.total_steps}")
        print(f"Total updates: {self.num_updates}")
        print(f"{'='*60}\n")
        
        # Save final checkpoint
        self._save_checkpoint(final=True)
    
    def _ppo_update(self, rollout_data):
        """
        Perform PPO update with all stability fixes.
        
        Implements the full PPO algorithm with:
        1. Pre-encoded tensors (no re-encoding during update)
        2. Value function clipping (PPO2)
        3. KL divergence tracking and early stopping
        4. Correct entropy bonus (positive for exploration)
        5. Normalized advantages (already done in prepare_batch_data)
        
        Args:
            rollout_data: Dictionary with rollout data and advantages
        """
        # Tracking metrics
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        kl_divs = []
        
        # Get old values for value clipping
        old_values = rollout_data['values']
        
        # PPO update loop
        for epoch in range(self.config.NUM_EPOCHS):
            # Track KL divergence for early stopping
            epoch_kl = 0.0
            num_batches = 0
            
            # Iterate through mini-batches
            for batch in self.rollout_collector.get_mini_batches(rollout_data, batch_size=self.config.BATCH_SIZE):
                # Get batch data - use pre-encoded tensors
                obs_tensors = batch['obs_tensors']  # Already tensors from rollout
                actions = batch['actions']
                old_logprobs_batch = batch['logprobs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values_batch = batch['values']
                
                # Evaluate actions with current policy
                logprobs, values, entropy, kl_div = self.policy.evaluate_actions(
                    obs_tensors, actions, old_values=old_values_batch, old_logprobs=old_logprobs_batch
                )
                
                # Compute ratio for PPO clip
                ratio = torch.exp(logprobs - old_logprobs_batch)
                
                # Compute policy loss (PPO clip objective)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.PPO_CLIP_EPS, 1.0 + self.config.PPO_CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss with clipping (PPO2)
                value_pred_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch,
                    -self.config.VALUE_CLIP_RANGE,
                    self.config.VALUE_CLIP_RANGE
                )
                value_loss_unclipped = F.mse_loss(values, returns)
                value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                
                # Combine losses (correct entropy sign - subtract to encourage exploration)
                total_loss = (
                    policy_loss +
                    self.config.VALUE_COEFF * value_loss -
                    self.config.ENTROPY_COEFF * entropy.mean()  # Subtract for bonus
                )
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.MAX_GRAD_NORM)
                
                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # Track clip fraction and KL divergence
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.config.PPO_CLIP_EPS).float().mean().item()
                    clip_fractions.append(clip_fraction)
                    
                    # Use KL divergence from evaluate_actions (already computed)
                    approx_kl = kl_div.item()
                    
                    # Ensure KL is finite
                    if not torch.isfinite(kl_div):
                        print(f"  Warning: Non-finite KL detected, skipping batch")
                        continue
                    
                    kl_divs.append(approx_kl)
                    epoch_kl += approx_kl
                    num_batches += 1
            
            # Early stopping based on KL divergence
            avg_kl = epoch_kl / max(num_batches, 1)
            if avg_kl > self.config.MAX_KL:
                print(f"  Early stopping at epoch {epoch + 1}/{self.config.NUM_EPOCHS} (KL={avg_kl:.4f} > {self.config.MAX_KL})")
                break
        
        # Print update metrics
        print(f"[Update {self.num_updates + 1}] Policy loss: {np.mean(policy_losses):.4f}")
        print(f"              Value loss: {np.mean(value_losses):.4f}")
        print(f"              Entropy: {np.mean(entropies):.4f}")
        print(f"              Clip frac: {np.mean(clip_fractions):.4f}")
        if kl_divs:
            print(f"              Approx KL: {np.mean(kl_divs):.4f}")
    
    def _update_episode_stats(self, episode_info):
        """
        Update episode statistics from rollout.
        
        Args:
            episode_info: Dictionary with episode information
        """
        for episode in episode_info['completed_episodes']:
            self.episode_rewards.append(episode['reward'])
            self.episode_lengths.append(episode['length'])
            self.total_episodes += 1
            
            # Track outcomes from agent's perspective (using result_pov)
            outcome = episode['outcome']
            # In self-play, AGENT1_WIN means we won (from our POV)
            # because result_pov is already flipped in TronEnv
            if outcome == GameResult.AGENT1_WIN:
                self.win_counts['win'] += 1
            elif outcome == GameResult.AGENT2_WIN:
                self.win_counts['loss'] = self.win_counts.get('loss', 0) + 1
            elif outcome == GameResult.DRAW:
                self.win_counts['draw'] += 1
        
        # Keep only recent episodes for averaging
        max_episodes_to_keep = 100
        if len(self.episode_rewards) > max_episodes_to_keep:
            self.episode_rewards = self.episode_rewards[-max_episodes_to_keep:]
            self.episode_lengths = self.episode_lengths[-max_episodes_to_keep:]
    
    def _log_metrics(self, rollout_time, update_time):
        """
        Log training metrics.
        
        Args:
            rollout_time: Time spent collecting rollout (seconds)
            update_time: Time spent on update (seconds)
        """
        # Compute metrics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        
        total_outcomes = sum(self.win_counts.values())
        win_rate = self.win_counts['win'] / total_outcomes if total_outcomes > 0 else 0.0
        draw_rate = self.win_counts['draw'] / total_outcomes if total_outcomes > 0 else 0.0
        
        print(f"\n[Step {self.total_steps:,}] Update {self.num_updates}")
        print(f"  Episodes completed: {self.total_episodes}")
        print(f"  Avg reward: {avg_reward:.2f}")
        print(f"  Avg length: {avg_length:.1f}")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Draw rate: {draw_rate:.2%}")
        print(f"  Rollout time: {rollout_time:.2f}s")
        print(f"  Update time: {update_time:.2f}s")
    
    def _save_checkpoint(self, final=False):
        """
        Save training checkpoint.
        
        Args:
            final: If True, save as final checkpoint
        """
        if final:
            checkpoint_path = self.checkpoint_dir / 'final_checkpoint.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.total_steps}.pt'
        
        checkpoint = {
            'total_steps': self.total_steps,
            'num_updates': self.num_updates,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'win_counts': self.win_counts,
            'total_episodes': self.total_episodes,
            'config': {
                'SEED': self.config.SEED,
                'LEARNING_RATE': self.config.LEARNING_RATE,
                'GAMMA': self.config.GAMMA,
            },
            # PR-3: Add model and optimizer state
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Add opponent policy if using frozen opponent
        if self.config.USE_FROZEN_OPPONENT:
            checkpoint['opponent_state_dict'] = self.opponent_policy.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n[Checkpoint] Saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        
        self.total_steps = checkpoint['total_steps']
        self.num_updates = checkpoint['num_updates']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.win_counts = checkpoint['win_counts']
        self.total_episodes = checkpoint['total_episodes']
        
        # PR-3: Load model and optimizer state
        if 'model_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.config.USE_FROZEN_OPPONENT and 'opponent_state_dict' in checkpoint:
            self.opponent_policy.load_state_dict(checkpoint['opponent_state_dict'])
        
        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  Resuming from step {self.total_steps}")


def main():
    """Main entry point for training."""
    print("\n" + "="*60)
    print("Tron Agent Training - PR-3 (Neural Policy)")
    print("="*60)
    print(f"Config:")
    print(f"  Seed: {config.SEED}")
    print(f"  Board: {config.BOARD_H}x{config.BOARD_W}")
    print(f"  Rollout length: {config.ROLLOUT_LENGTH}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max steps: {config.MAX_TRAINING_STEPS}")
    print(f"  Device: {config.DEVICE}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
