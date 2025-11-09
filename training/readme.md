# Training Pipeline - PR-2

This directory contains the reinforcement learning training stack for the Tron agent using self-play and PPO (Proximal Policy Optimization).

## Overview

The training pipeline is designed to be:
- **Deterministic**: All random seeds are controlled for reproducibility
- **Offline**: Runs independently of the Flask server
- **CPU-only**: Compatible with CPU-only PyTorch
- **Modular**: Clean separation between game simulation, rewards, environment, and training

## Directory Structure

```
training/
├── config.py          # Hyperparameters and configuration
├── game_sim.py        # Local game simulation wrapper
├── rewards.py         # Reward shaping functions
├── env.py            # RL environment with self-play
├── rollout.py        # PPO rollout collector + DummyPolicy
├── trainer.py        # Main training script
├── readme.md         # This file
└── checkpoints/      # Saved model checkpoints
```

## Module Descriptions

### `config.py`
Contains all configuration and hyperparameters:
- Board dimensions (18x20)
- PPO hyperparameters (learning rate, gamma, GAE lambda, etc.)
- Rollout settings (length, batch size)
- Reward weights for shaping
- Random seed for reproducibility
- Checkpoint and logging intervals

### `game_sim.py`
Provides `LocalGame` class that wraps the official `case_closed_game.py`:
- Deterministic game resets with seeding
- Step-by-step game progression
- Observations as numpy arrays (CPU-only)
- State cloning for lookahead (optional)
- No dependencies on Flask or judge engine

### `rewards.py`
Implements sophisticated reward shaping:
1. **Survival Reward**: +1 per turn alive
2. **Terminal Rewards**: +100 win, -100 loss, 0 draw
3. **Space Control**: Voronoi territory margin (BFS-based)
4. **Mobility**: Reachable area from current position
5. **Trap Avoidance**: Penalty when degree ≤ 1
6. **Head-on Collision**: Penalty for predicted collisions
7. **Boost Efficiency**: Reward/penalty based on mobility change

### `env.py`
Single-agent RL environment with self-play:
- Wraps `LocalGame` for RL interface
- Supports self-play (agent vs. copy of itself)
- Supports heuristic opponent mode
- Symmetric experience: 50% as player 1, 50% as player 2
- Converts action indices (0-3) to Direction enums
- Computes shaped rewards using `rewards.py`

### `rollout.py`
Contains `RolloutCollector` and `DummyPolicy`:
- **RolloutCollector**: Collects trajectories for PPO
  - Stores observations, actions, log probs, rewards, dones, values
  - Computes GAE (Generalized Advantage Estimation)
  - Handles episode boundaries correctly
  - Preallocates memory for efficiency
  - All CPU tensors (no GPU)
  - Provides mini-batch iterator
  
- **DummyPolicy**: Placeholder for PR-2
  - Random action selection
  - Fake log probabilities and value estimates
  - Compatible interface for PPO (will swap with NN in PR-3)

### `trainer.py`
Main training orchestration:
- Sets up environment, policy, and rollout collector
- Runs main training loop
- Collects rollouts using self-play
- Computes advantages (GAE)
- Performs PPO updates (placeholder structure for PR-3)
- Logs metrics (win rate, avg reward, avg episode length)
- Saves checkpoints every N steps
- Seeds all RNG sources for reproducibility

## How to Run Training

### 1. Install Dependencies

First, ensure you have the required packages:

```bash
pip install -r requirements.txt
```

This should include:
- `numpy` (for observations)
- `torch` (CPU-only version)
- `Flask` and `requests` (for agent server)

### 2. Run Training

From the repository root:

```bash
python training/trainer.py
```

Or from within the training directory:

```bash
cd training
python trainer.py
```

### 3. Monitor Progress

The trainer will print logs every N steps showing:
- Current step count
- Episodes completed
- Average reward
- Average episode length
- Win/draw rates
- Timing information

Example output:
```
[Step 10,000] Update 5
  Episodes completed: 127
  Avg reward: 45.32
  Avg length: 78.5
  Win rate: 48.00%
  Draw rate: 4.00%
  Rollout time: 12.34s
  Update time: 0.56s
```

### 4. Checkpoints

Checkpoints are saved to `training/checkpoints/`:
- `checkpoint_step_N.pt`: Saved every N steps (configurable in `config.py`)
- `final_checkpoint.pt`: Saved at the end of training

Each checkpoint contains:
- Training state (steps, updates, metrics)
- Episode statistics
- Configuration used
- (In PR-3: model weights)

## Configuration

Edit `training/config.py` to customize:

```python
# Change training duration
MAX_TRAINING_STEPS = 100_000  # Default: 1,000,000

# Change rollout length
ROLLOUT_LENGTH = 1024  # Default: 2048

# Change learning rate
LEARNING_RATE = 1e-4  # Default: 3e-4

# Change reward weights
REWARD_SPACE_CONTROL = 0.02  # Default: 0.01
```

## How to Evaluate Checkpoints

To evaluate a trained checkpoint (will be implemented in PR-3):

```python
from training.trainer import Trainer

trainer = Trainer()
trainer.load_checkpoint('training/checkpoints/checkpoint_step_100000.pt')

# Run evaluation games
# (Evaluation code to be added in PR-3)
```

## Future Integration with agent.py (PR-3/PR-4)

The training pipeline is designed to integrate with the live agent in future PRs:

1. **PR-3**: Replace `DummyPolicy` with real neural network
   - Implement policy network (actor-critic)
   - Implement actual PPO update logic
   - Train network using this pipeline
   
2. **PR-4**: Integrate trained model with `agent.py`
   - Add checkpoint loading to `agent.py`
   - Replace heuristic logic with NN inference
   - Support model hot-swapping during competition

The key interface is:
```python
# Policy interface (already compatible)
action, logprob, value = policy.act(observation)
logprobs, values, entropy = policy.evaluate_actions(obs_batch, actions)
```

## Testing Determinism

To verify deterministic behavior:

```bash
# Run training twice with same seed
python training/trainer.py  # First run
python training/trainer.py  # Second run

# Compare checkpoints - should be identical
```

## Troubleshooting

### ImportError
If you get import errors, ensure you're running from the repository root or that the path is set correctly.

### Memory Issues
If you run out of memory, reduce:
- `ROLLOUT_LENGTH` in `config.py`
- `BATCH_SIZE` in `config.py`

### Slow Training
Training with `DummyPolicy` should be fast. If it's slow:
- Check `ROLLOUT_LENGTH` (longer = more time per update)
- Check for debug prints in the game simulation

### CUDA Errors
This pipeline is CPU-only. If you see CUDA errors:
- Ensure PyTorch CPU version is installed
- Check that no tensors are being moved to GPU

## Notes

- **PR-2 Status**: This PR implements the full training infrastructure but uses a `DummyPolicy` (random agent) instead of a neural network.
- **No Changes to Competition Code**: This PR does not modify `agent.py`, `case_closed_game.py`, or any judge engine files.
- **CPU-Only**: All operations use CPU tensors for compatibility.
- **Deterministic**: All random seeds are set for reproducibility.

## Contact

For questions or issues with the training pipeline, please open an issue on the repository.
