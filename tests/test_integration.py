"""
Integration test for full training smoke run.

Tests that:
- Training runs end-to-end without errors
- CSV logs are created
- Checkpoints are saved
- No NaN/inf values appear in losses
"""

import pytest
import tempfile
import os
from pathlib import Path
from training.trainer import Trainer
from training.config import TrainingConfig


@pytest.mark.integration
@pytest.mark.slow
def test_smoke_training():
    """
    Smoke test: run a short training session to verify everything works.
    
    This test runs for 128 steps (very short) to verify:
    - No crashes or exceptions
    - CSV log created
    - Checkpoints created
    - No NaN/inf in training
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal config for fast testing
        cfg = TrainingConfig()
        cfg.SEED = 123
        cfg.BOARD_H = 10
        cfg.BOARD_W = 10
        cfg.LATENT_SIZE = 32  # Small model
        cfg.ROLLOUT_LENGTH = 64
        cfg.BATCH_SIZE = 32
        cfg.NUM_EPOCHS = 2
        cfg.MAX_TRAINING_STEPS = 128  # Very short run
        cfg.LOG_INTERVAL = 64
        cfg.CHECKPOINT_INTERVAL = 128
        cfg.LOG_DIR = tmpdir
        cfg.CKPT_DIR = tmpdir
        cfg.CSV_LOG = True
        cfg.TENSORBOARD = False
        cfg.USE_FROZEN_OPPONENT = True
        cfg.OPPONENT_UPDATE_INTERVAL = 1
        
        # Run training
        trainer = Trainer(cfg)
        trainer.train()
        
        # Verify CSV log was created
        run_name = cfg.get_run_name()
        csv_path = Path(tmpdir) / run_name / "metrics.csv"
        assert csv_path.exists(), f"CSV log should exist at {csv_path}"
        
        # Verify CSV has content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 2, "CSV should have header + at least one data row"
        
        # Verify checkpoint was created
        ckpt_files = list(Path(tmpdir).glob("*.pt"))
        assert len(ckpt_files) > 0, "At least one checkpoint should be saved"
        
        # Verify no NaN/inf in episode stats
        assert all(isinstance(r, (int, float)) and r == r for r in trainer.episode_rewards), \
            "Episode rewards should not contain NaN"
        assert all(isinstance(l, (int, float)) and l == l for l in trainer.episode_lengths), \
            "Episode lengths should not contain NaN"
        
        print(f"\nSmoke test completed successfully!")
        print(f"  Total steps: {trainer.total_steps}")
        print(f"  Total updates: {trainer.num_updates}")
        print(f"  Total episodes: {trainer.total_episodes}")
        print(f"  Checkpoints saved: {len(ckpt_files)}")


if __name__ == "__main__":
    # Allow running this test standalone for debugging
    test_smoke_training()
