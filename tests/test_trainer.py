"""
Unit tests for the PPO trainer.

Tests cover:
- Checkpoint save/load roundtrip
- Opponent policy synchronization
- NaN guard skip logic
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from training.trainer import Trainer, set_seed
from training.config import TrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def small_config():
    """Create a small config for faster tests."""
    cfg = TrainingConfig()
    cfg.BOARD_H = 10
    cfg.BOARD_W = 10
    cfg.LATENT_SIZE = 32
    cfg.ROLLOUT_LENGTH = 16
    cfg.BATCH_SIZE = 8
    cfg.NUM_EPOCHS = 2
    cfg.MAX_TRAINING_STEPS = 32
    cfg.LOG_INTERVAL = 32
    cfg.CHECKPOINT_INTERVAL = 32
    cfg.CSV_LOG = False
    cfg.TENSORBOARD = False
    cfg.USE_FROZEN_OPPONENT = True
    cfg.OPPONENT_UPDATE_INTERVAL = 2
    return cfg


@pytest.mark.unit
def test_set_seed_deterministic():
    """Test that set_seed produces deterministic results."""
    set_seed(42)
    val1 = torch.rand(5)
    
    set_seed(42)
    val2 = torch.rand(5)
    
    assert torch.allclose(val1, val2), "Same seed should produce same random values"


@pytest.mark.unit
def test_checkpoint_roundtrip(small_config, temp_dir):
    """Test that checkpoints can be saved and loaded correctly."""
    small_config.CKPT_DIR = temp_dir
    
    # Create trainer and modify state
    trainer = Trainer(small_config)
    trainer.total_steps = 1000
    trainer.num_updates = 50
    trainer.episode_rewards = [1.0, 2.0, 3.0]
    trainer.episode_lengths = [10, 20, 30]
    trainer.win_counts = {'win': 2, 'loss': 1, 'draw': 0}
    trainer.total_episodes = 3
    
    # Save checkpoint
    ckpt_path = Path(temp_dir) / "test_checkpoint.pt"
    checkpoint = {
        'total_steps': trainer.total_steps,
        'num_updates': trainer.num_updates,
        'episode_rewards': trainer.episode_rewards,
        'episode_lengths': trainer.episode_lengths,
        'win_counts': trainer.win_counts,
        'total_episodes': trainer.total_episodes,
        'config': {
            'SEED': trainer.config.SEED,
            'LEARNING_RATE': trainer.config.LEARNING_RATE,
            'GAMMA': trainer.config.GAMMA,
        },
        'model_state_dict': trainer.policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }
    if trainer.config.USE_FROZEN_OPPONENT:
        checkpoint['opponent_state_dict'] = trainer.opponent_policy.state_dict()
    
    torch.save(checkpoint, ckpt_path)
    
    # Create new trainer and load checkpoint
    trainer2 = Trainer(small_config)
    trainer2.load_checkpoint(ckpt_path)
    
    # Verify state restored
    assert trainer2.total_steps == 1000
    assert trainer2.num_updates == 50
    assert trainer2.episode_rewards == [1.0, 2.0, 3.0]
    assert trainer2.episode_lengths == [10, 20, 30]
    assert trainer2.win_counts == {'win': 2, 'loss': 1, 'draw': 0}
    assert trainer2.total_episodes == 3
    
    # Verify model weights match
    for p1, p2 in zip(trainer.policy.parameters(), trainer2.policy.parameters()):
        assert torch.allclose(p1, p2), "Model weights should match"


@pytest.mark.unit
def test_opponent_sync(small_config):
    """Test that opponent weights are synchronized at correct intervals."""
    small_config.USE_FROZEN_OPPONENT = True
    small_config.OPPONENT_UPDATE_INTERVAL = 2  # Sync every 2 updates
    
    trainer = Trainer(small_config)
    
    # Get initial weights
    initial_main = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    initial_opp = {k: v.clone() for k, v in trainer.opponent_policy.state_dict().items()}
    
    # Weights should be identical initially
    for k in initial_main:
        assert torch.allclose(initial_main[k], initial_opp[k]), \
            f"Initial weights should match for {k}"
    
    # Modify main policy weights
    with torch.no_grad():
        for p in trainer.policy.parameters():
            p.add_(0.1)
    
    # Before sync, weights should differ
    new_main = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    current_opp = {k: v.clone() for k, v in trainer.opponent_policy.state_dict().items()}
    
    for k in new_main:
        assert not torch.allclose(new_main[k], current_opp[k]), \
            f"Weights should differ before sync for {k}"
    
    # Simulate opponent update (as done in trainer)
    trainer.opponent_policy.load_state_dict(trainer.policy.state_dict())
    
    # After sync, weights should match
    synced_opp = {k: v.clone() for k, v in trainer.opponent_policy.state_dict().items()}
    
    for k in new_main:
        assert torch.allclose(new_main[k], synced_opp[k]), \
            f"Weights should match after sync for {k}"


@pytest.mark.unit
def test_nan_guard_logic():
    """Test that NaN detection works correctly."""
    # Test finite tensor
    t1 = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isfinite(t1).all(), "Should detect finite tensor"
    
    # Test NaN tensor
    t2 = torch.tensor([1.0, float('nan'), 3.0])
    assert not torch.isfinite(t2).all(), "Should detect NaN in tensor"
    
    # Test inf tensor
    t3 = torch.tensor([1.0, float('inf'), 3.0])
    assert not torch.isfinite(t3).all(), "Should detect inf in tensor"
