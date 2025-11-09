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
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import config
from training.env import TronEnv
from training.rollout import RolloutCollector, DummyPolicy
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
        
        # Initialize policy (DummyPolicy for PR-2, will be NN in PR-3)
        self.policy = DummyPolicy(action_space=self.config.NUM_ACTIONS)
        
        # Initialize environment
        self.env = TronEnv(
            opponent_policy=self.policy,  # Self-play
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
            
            # Compute advantages (GAE)
            rollout_data = self.rollout_collector.compute_advantages(rollout_data)
            
            # Prepare data for training
            rollout_data = self.rollout_collector.prepare_batch_data(rollout_data)
            
            # Perform PPO update (placeholder for PR-2)
            update_start = time.time()
            self._ppo_update(rollout_data)
            update_time = time.time() - update_start
            
            # Update counters
            self.total_steps += len(rollout_data['actions'])
            self.num_updates += 1
            
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
        Perform PPO update (placeholder for PR-2).
        
        In PR-3, this will:
        1. Split data into mini-batches
        2. Compute policy and value losses
        3. Backpropagate and update network
        4. Clip gradients
        
        Args:
            rollout_data: Dictionary with rollout data and advantages
        """
        # Placeholder: In PR-3, this will contain actual PPO update logic
        
        # For now, just iterate through mini-batches to validate structure
        for epoch in range(self.config.NUM_EPOCHS):
            for batch in self.rollout_collector.get_mini_batches(rollout_data, batch_size=self.config.BATCH_SIZE):
                # In PR-3: compute losses and update network
                # For now: just validate batch structure
                assert 'observations' in batch
                assert 'actions' in batch
                assert 'advantages' in batch
                assert 'returns' in batch
                # Last batch may be smaller than BATCH_SIZE
                assert len(batch['actions']) > 0
    
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
            
            # Track outcomes
            outcome = episode['outcome']
            if outcome == GameResult.AGENT1_WIN or outcome == GameResult.AGENT2_WIN:
                # In self-play, we need to track from perspective of our agent
                # For now, just count as win (will be refined with player tracking)
                self.win_counts['win'] += 1
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
            }
            # In PR-3: add model weights
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n[Checkpoint] Saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.total_steps = checkpoint['total_steps']
        self.num_updates = checkpoint['num_updates']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.win_counts = checkpoint['win_counts']
        self.total_episodes = checkpoint['total_episodes']
        
        # In PR-3: load model weights
        
        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  Resuming from step {self.total_steps}")


def main():
    """Main entry point for training."""
    print("\n" + "="*60)
    print("Tron Agent Training - PR-2")
    print("="*60)
    print(f"Config:")
    print(f"  Seed: {config.SEED}")
    print(f"  Board: {config.BOARD_H}x{config.BOARD_W}")
    print(f"  Rollout length: {config.ROLLOUT_LENGTH}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max steps: {config.MAX_TRAINING_STEPS}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
