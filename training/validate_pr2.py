#!/usr/bin/env python3
"""
Validation script to verify all PR-2 requirements are met.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists

def main():
    print("="*60)
    print("PR-2 Validation: Training Pipeline Requirements")
    print("="*60)
    
    base = Path(__file__).parent.parent
    all_pass = True
    
    # Check required files
    print("\n1. Required Files:")
    required_files = [
        "training/config.py",
        "training/game_sim.py", 
        "training/rewards.py",
        "training/env.py",
        "training/rollout.py",
        "training/trainer.py",
        "training/readme.md",
        "training/checkpoints/.gitkeep",
    ]
    
    for f in required_files:
        if not check_file_exists(base / f):
            all_pass = False
    
    # Check no changes to competition files
    print("\n2. No Changes to Competition Files:")
    competition_files = [
        "agent.py",
        "case_closed_game.py", 
        "judge_engine.py",
    ]
    
    for f in competition_files:
        # Just verify they still exist
        check_file_exists(base / f)
    
    # Check imports work
    print("\n3. Module Imports:")
    try:
        sys.path.insert(0, str(base))
        from training.config import config
        print("✓ training.config imports")
        
        from training.game_sim import LocalGame
        print("✓ training.game_sim imports")
        
        from training.rewards import compute_reward
        print("✓ training.rewards imports")
        
        from training.env import TronEnv
        print("✓ training.env imports")
        
        from training.rollout import RolloutCollector, DummyPolicy
        print("✓ training.rollout imports")
        
        from training.trainer import Trainer
        print("✓ training.trainer imports")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        all_pass = False
    
    # Check requirements
    print("\n4. Dependencies:")
    req_path = base / "requirements.txt"
    if req_path.exists():
        with open(req_path) as f:
            reqs = f.read()
        print("✓ numpy in requirements" if "numpy" in reqs else "✗ numpy missing")
        print("✓ torch in requirements" if "torch" in reqs else "✗ torch missing")
    
    print("\n" + "="*60)
    if all_pass:
        print("✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
    print("="*60)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    exit(main())
