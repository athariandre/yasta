#!/usr/bin/env python3
"""
Validation script to verify all PR-3 requirements are met.

This script tests:
1. Neural policy imports and builds successfully
2. Forward pass works correctly
3. PPO update runs without errors
4. Checkpoint save/load works
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

def test_imports():
    """Test that all modules import correctly."""
    print("\n" + "="*60)
    print("1. Testing Imports")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from training.config import config
        print("✓ training.config imports")
        
        from training.policy import ActorCriticPolicy
        print("✓ training.policy imports")
        
        from training.trainer import Trainer
        print("✓ training.trainer imports")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_build():
    """Test that policy builds successfully."""
    print("\n" + "="*60)
    print("2. Testing Policy Build")
    print("="*60)
    
    try:
        from training.policy import ActorCriticPolicy
        from training.config import config
        
        policy = ActorCriticPolicy(
            board_h=config.BOARD_H,
            board_w=config.BOARD_W,
            num_actions=config.NUM_ACTIONS,
            latent_size=config.LATENT_SIZE,
            device=config.DEVICE,
            verbose=True
        )
        
        print("✓ Policy initialized successfully")
        
        # Check model size
        total_params = sum(p.numel() for p in policy.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        if model_size_mb > 5.0:
            print(f"✗ Model size {model_size_mb:.2f} MB exceeds 5MB limit")
            return False
        else:
            print(f"✓ Model size {model_size_mb:.2f} MB is within 5MB limit")
        
        return True
    except Exception as e:
        print(f"✗ Policy build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that forward pass works correctly."""
    print("\n" + "="*60)
    print("3. Testing Forward Pass")
    print("="*60)
    
    try:
        from training.policy import ActorCriticPolicy
        from training.config import config
        
        policy = ActorCriticPolicy()
        
        # Create fake observation
        obs = {
            'board': np.zeros((config.BOARD_H, config.BOARD_W), dtype=np.float32),
            'my_head': np.array([10, 9], dtype=np.float32),
            'opp_head': np.array([10, 8], dtype=np.float32),
            'boosts': np.array([3, 3], dtype=np.float32),
            'turns': np.array([1], dtype=np.float32),
            'directions': np.array([0, 1], dtype=np.float32),
            'alive': np.array([1, 1], dtype=np.float32),
        }
        
        # Test act()
        action, logprob, value = policy.act(obs)
        print(f"✓ act() returns: action={action}, logprob={logprob:.4f}, value={value:.4f}")
        
        # Validate outputs
        assert isinstance(action, int) and 0 <= action < 4, "Invalid action"
        assert isinstance(logprob, float), "Invalid logprob type"
        assert isinstance(value, float), "Invalid value type"
        print("✓ act() output types are correct")
        
        # Test evaluate_actions() with pre-encoded tensors
        obs_tensors = torch.stack([policy._obs_to_tensor(obs) for _ in range(5)], dim=0)
        actions = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
        
        logprobs, values, entropy, kl_div = policy.evaluate_actions(obs_tensors, actions)
        print(f"✓ evaluate_actions() returns tensors of shape: {logprobs.shape}, {values.shape}, {entropy.shape}, kl_div={kl_div.item():.4f}")
        
        # Validate outputs
        assert logprobs.shape == (5,), "Invalid logprobs shape"
        assert values.shape == (5,), "Invalid values shape"
        assert entropy.shape == (5,), "Invalid entropy shape"
        print("✓ evaluate_actions() output shapes are correct")
        
        # Test get_value() for GAE bootstrapping
        bootstrap_value = policy.get_value(obs)
        print(f"✓ get_value() returns: {bootstrap_value:.4f}")
        assert isinstance(bootstrap_value, float), "Invalid get_value type"
        print("✓ get_value() works for GAE bootstrapping")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_update():
    """Test that PPO update runs without errors."""
    print("\n" + "="*60)
    print("4. Testing PPO Update")
    print("="*60)
    
    try:
        from training.policy import ActorCriticPolicy
        from training.config import config
        import torch.nn.functional as F
        
        policy = ActorCriticPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
        
        # Create fake batch data - use pre-encoded tensors
        batch_size = 32
        obs_tensors_list = []
        for _ in range(batch_size):
            obs = {
                'board': np.random.rand(config.BOARD_H, config.BOARD_W).astype(np.float32),
                'my_head': np.random.rand(2).astype(np.float32) * np.array([config.BOARD_W, config.BOARD_H]),
                'opp_head': np.random.rand(2).astype(np.float32) * np.array([config.BOARD_W, config.BOARD_H]),
                'boosts': np.random.rand(2).astype(np.float32) * config.MAX_BOOST,
                'turns': np.random.rand(1).astype(np.float32) * config.MAX_TURNS,
                'directions': np.random.rand(2).astype(np.float32) * 3,
                'alive': np.ones(2, dtype=np.float32),
            }
            obs_tensors_list.append(policy._obs_to_tensor(obs))
        
        obs_tensors = torch.stack(obs_tensors_list, dim=0)
        actions = torch.randint(0, 4, (batch_size,), dtype=torch.long)
        old_logprobs = torch.randn(batch_size)
        old_values = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        # Run one PPO update step
        logprobs, values, entropy, kl_div = policy.evaluate_actions(obs_tensors, actions, old_values=old_values)
        
        # Compute losses
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with clipping
        value_pred_clipped = old_values + torch.clamp(values - old_values, -config.VALUE_CLIP_RANGE, config.VALUE_CLIP_RANGE)
        value_loss_unclipped = F.mse_loss(values, returns)
        value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        
        total_loss = policy_loss + config.VALUE_COEFF * value_loss - config.ENTROPY_COEFF * entropy.mean()
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        
        print(f"✓ PPO update completed successfully")
        print(f"  Policy loss: {policy_loss.item():.4f}")
        print(f"  Value loss: {value_loss.item():.4f}")
        print(f"  Entropy: {entropy.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ PPO update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """Test that checkpoint save/load works."""
    print("\n" + "="*60)
    print("5. Testing Checkpoint Save/Load")
    print("="*60)
    
    try:
        from training.policy import ActorCriticPolicy
        from training.config import config
        import tempfile
        
        policy = ActorCriticPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        policy2 = ActorCriticPolicy()
        optimizer2 = torch.optim.Adam(policy2.parameters(), lr=config.LEARNING_RATE)
        
        checkpoint_loaded = torch.load(checkpoint_path, map_location=config.DEVICE)
        policy2.load_state_dict(checkpoint_loaded['model_state_dict'])
        optimizer2.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
        print(f"✓ Checkpoint loaded successfully")
        
        # Verify parameters match
        for p1, p2 in zip(policy.parameters(), policy2.parameters()):
            if not torch.allclose(p1, p2):
                print("✗ Loaded parameters don't match saved parameters")
                return False
        
        print("✓ Loaded parameters match saved parameters")
        
        # Clean up
        os.unlink(checkpoint_path)
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("PR-3 Validation: Neural Policy and PPO Implementation")
    print("="*60)
    
    all_pass = True
    
    # Run tests
    all_pass &= test_imports()
    all_pass &= test_policy_build()
    all_pass &= test_forward_pass()
    all_pass &= test_ppo_update()
    all_pass &= test_checkpoint_save_load()
    
    # Summary
    print("\n" + "="*60)
    if all_pass:
        print("✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
    print("="*60 + "\n")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
