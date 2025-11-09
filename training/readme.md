# Training Pipeline - PR-3 (Neural Policy)

This directory contains the reinforcement learning training stack for the Tron agent using self-play and PPO (Proximal Policy Optimization) with a neural network policy.

## Overview

The training pipeline is designed to be:
- **Deterministic**: All random seeds are controlled for reproducibility
- **Offline**: Runs independently of the Flask server
- **CPU-only**: Compatible with CPU-only PyTorch (no CUDA)
- **Modular**: Clean separation between game simulation, rewards, environment, and training
- **Efficient**: Small neural network (<5MB) for fast inference

## Directory Structure

```
training/
├── config.py          # Hyperparameters and configuration
├── game_sim.py        # Local game simulation wrapper
├── rewards.py         # Reward shaping functions
├── env.py            # RL environment with self-play
├── rollout.py        # PPO rollout collector
├── policy.py         # Neural network policy (Actor-Critic)
├── trainer.py        # Main training script with PPO updates
├── validate_pr3.py   # Validation script for PR-3
├── readme.md         # This file
└── checkpoints/      # Saved model checkpoints
```

## Module Descriptions

### `config.py`
Contains all configuration and hyperparameters:
- Board dimensions (18x20)
- Neural network architecture (latent size: 128)
- PPO hyperparameters (learning rate, clip epsilon, value/entropy coefficients)
- Rollout settings (length: 2048, batch size: 512, epochs: 4)
- Reward weights for shaping
- Random seed for reproducibility
- Checkpoint and logging intervals
- Device specification (CPU-only)

### `policy.py` (New in PR-3)
Implements the neural network policy:

**Architecture:**
- **Shared Encoder**: 
  - Input: Flattened observation (board + metadata)
  - 2-layer MLP: input → 128 → 128
  - ReLU activations
  - Output: 128-dimensional latent vector

- **Actor Head** (Policy):
  - Input: 128-dimensional latent vector
  - 2-layer MLP: 128 → 64 → 4
  - Output: Logits over 4 actions (UP, DOWN, LEFT, RIGHT)

- **Critic Head** (Value Function):
  - Input: 128-dimensional latent vector
  - 2-layer MLP: 128 → 64 → 1
  - Output: Scalar value estimate

**Key Methods:**
- `act(obs)`: Select action given observation → (action, logprob, value)
- `evaluate_actions(obs_batch, actions)`: Evaluate actions for PPO → (logprobs, values, entropy)
- `get_value(obs)`: Get value estimate for observation

**Design Constraints:**
- Total parameters: ~81K (0.31 MB)
- CPU-only operations (no CUDA)
- Fast inference (<5ms per step)
- Uses only MLPs (no convolutions)

### `game_sim.py`
Provides `LocalGame` class that wraps the official `case_closed_game.py`:
- Deterministic game resets with seeding
- Step-by-step game progression
- Observations as numpy arrays (CPU-only, float32)
- State cloning for lookahead (optional)
- No dependencies on Flask or judge engine
- PR-3: All observations guaranteed to be stable for NN input

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
- PR-3: All observations are float32 for NN stability

### `rollout.py`
Contains `RolloutCollector`:
- **RolloutCollector**: Collects trajectories for PPO
  - Stores observations, actions, log probs, rewards, dones, values
  - Computes GAE (Generalized Advantage Estimation)
  - Handles episode boundaries correctly
  - Preallocates memory for efficiency
  - All CPU tensors (no GPU)
  - Provides mini-batch iterator with shuffling
  - Converts observation dicts to batched tensors for NN

### `trainer.py`
Main training orchestration with full PPO implementation:
- Sets up neural policy and optimizer (Adam)
- Runs main training loop
- Collects rollouts using self-play
- Computes advantages (GAE)
- **Performs PPO updates** (PR-3):
  - Mini-batch iteration with shuffling
  - Computes policy loss (PPO clip objective)
  - Computes value loss (MSE)
  - Computes entropy bonus
  - Backpropagation with gradient clipping
  - Logs update metrics (policy loss, value loss, entropy, clip fraction)
- Saves/loads checkpoints with model and optimizer state
- Seeds all RNG sources for reproducibility

## Neural Network Architecture Details

### Input Processing
The network processes observations as follows:
1. **Board**: 18×20 grid → flattened to 360 values
2. **My Head**: (x, y) → 2 values
3. **Opponent Head**: (x, y) → 2 values
4. **Boosts**: [my_boosts, opp_boosts] → 2 values
5. **Turns**: current turn count → 1 value
6. **Directions**: [my_dir, opp_dir] (encoded 0-3) → 2 values
7. **Alive**: [my_alive, opp_alive] (binary) → 2 values

**Total input size**: 360 + 2 + 2 + 2 + 1 + 2 + 2 = **371 features**

All inputs are normalized float32 values for numerical stability.

### Output
- **Actor**: 4 logits (one per action)
- **Critic**: 1 scalar (value estimate)

## PPO Update Explanation

The PPO (Proximal Policy Optimization) algorithm updates the policy by:

1. **Collect Rollouts**: Gather trajectories using current policy
2. **Compute Advantages**: Use GAE (Generalized Advantage Estimation)
3. **Mini-batch Updates**: For each epoch:
   - Shuffle rollout data
   - Split into mini-batches (size 512)
   - For each mini-batch:
     - Evaluate actions with current policy → new logprobs, values, entropy
     - Compute importance sampling ratio: `ratio = exp(new_logprob - old_logprob)`
     - Compute clipped policy loss:
       ```
       surr1 = ratio * advantages
       surr2 = clip(ratio, 1-ε, 1+ε) * advantages
       policy_loss = -min(surr1, surr2)
       ```
     - Compute value loss: `MSE(values, returns)`
     - Compute entropy bonus: `-entropy.mean()`
     - Total loss = `policy_loss + 0.5*value_loss + 0.01*entropy_loss`
     - Backpropagate and clip gradients (max norm: 0.5)
     - Update parameters with Adam optimizer

### PPO Hyperparameters
- **Clip epsilon** (ε): 0.2
- **Learning rate**: 3e-4
- **Batch size**: 512
- **Epochs per rollout**: 4
- **Value coefficient**: 0.5
- **Entropy coefficient**: 0.01
- **Max gradient norm**: 0.5
- **Discount factor** (γ): 0.99
- **GAE lambda** (λ): 0.95

## Checkpoint Saving and Loading

### What Gets Saved
Checkpoints include:
- **Training state**: steps, updates, episode stats
- **Model state**: All neural network parameters
- **Optimizer state**: Adam optimizer state (momentum, variance)
- **Configuration**: Hyperparameters used

### Checkpoint Files
- `checkpoint_step_N.pt`: Saved every 10,000 steps
- `final_checkpoint.pt`: Saved at end of training

### How to Load
```python
from training.trainer import Trainer

trainer = Trainer()
trainer.load_checkpoint('training/checkpoints/checkpoint_step_10000.pt')
# Continue training or evaluate
```

## Expected Training Stability

### Normal Behavior
- **Policy loss**: Should decrease over time, typically 0.0-0.5 range
- **Value loss**: Should decrease, typically 0.0-1.0 range after convergence
- **Entropy**: Should slowly decrease (exploration → exploitation)
- **Clip fraction**: Should be 0.1-0.3 (indicates learning is happening)

### Warning Signs
- **NaN losses**: Check learning rate, gradient clipping
- **Exploding gradients**: Increase gradient clipping threshold
- **Value loss explosion**: Reduce value coefficient or learning rate
- **Zero entropy**: Policy has collapsed, may need to restart

### Stability Features
1. **Gradient clipping**: Prevents exploding gradients (max norm 0.5)
2. **PPO clipping**: Prevents large policy updates (ε = 0.2)
3. **Advantage normalization**: Stabilizes policy gradients
4. **CPU-only**: Consistent numerics (no CUDA non-determinism)
5. **Fixed seeds**: Fully reproducible runs

## CPU-Only Constraints

### Why CPU-Only?
Per competition rules:
- Docker images must be small (<5GB)
- GPU libraries (CUDA, cuDNN) are disallowed
- Inference must be fast (<5ms per step)
- Models must be portable across platforms

### Design Choices for CPU Efficiency
1. **Small network**: Only ~81K parameters
2. **No convolutions**: MLPs are faster on CPU
3. **Moderate hidden sizes**: 128 latent, 64 in heads
4. **Batch processing**: Vectorized operations in PyTorch
5. **NumPy → Torch conversion**: Minimal overhead

### Performance Targets
- **Inference**: <5ms per forward pass
- **Training throughput**: ~1000 steps/second (depends on hardware)
- **Memory usage**: <500MB during training

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

**Important**: Install CPU-only PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

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

The trainer will print logs showing:
- **Update metrics** (after each PPO update):
  - Policy loss
  - Value loss
  - Entropy
  - Clip fraction
  
- **Episode metrics** (every N steps):
  - Current step count
  - Episodes completed
  - Average reward
  - Average episode length
  - Win/draw rates
  - Timing information

Example output:
```
[Update 1] Policy loss: 0.3245
              Value loss: 0.8912
              Entropy: 1.3862
              Clip frac: 0.2156

[Step 2,048] Update 1
  Episodes completed: 15
  Avg reward: 45.32
  Avg length: 78.5
  Win rate: 48.00%
  Draw rate: 4.00%
  Rollout time: 12.34s
  Update time: 0.56s
```

### 4. Validate Implementation

Before running full training, validate the implementation:

```bash
python training/validate_pr3.py
```

This runs quick tests to ensure:
- All modules import correctly
- Policy builds successfully and is <5MB
- Forward pass works
- PPO update runs without errors
- Checkpoint save/load works

### 5. Checkpoints

Checkpoints are saved to `training/checkpoints/`:
- `checkpoint_step_N.pt`: Saved every 10,000 steps (configurable in `config.py`)
- `final_checkpoint.pt`: Saved at the end of training

Each checkpoint contains:
- Training state (steps, updates, metrics)
- Model weights (all network parameters)
- Optimizer state (Adam momentum and variance)
- Configuration used

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

# Change network size
LATENT_SIZE = 256  # Default: 128 (increases model size)

# Change PPO hyperparameters
PPO_CLIP_EPS = 0.1  # Default: 0.2 (more conservative updates)
BATCH_SIZE = 256  # Default: 512
NUM_EPOCHS = 8  # Default: 4
```

**Note**: Changing `LATENT_SIZE` will affect model size. Ensure it stays under 5MB.

## How to Evaluate Checkpoints

To evaluate a trained checkpoint:

```python
from training.trainer import Trainer
from training.config import config

# Load checkpoint
trainer = Trainer(config)
trainer.load_checkpoint('training/checkpoints/checkpoint_step_100000.pt')

# Run evaluation games
# (Use the env to play games with the loaded policy)
```

## Future Integration with agent.py (PR-4)

The training pipeline is designed to integrate with the live agent in future PRs:

**PR-4 will**:
- Add checkpoint loading to `agent.py`
- Replace heuristic logic with NN inference
- Support model hot-swapping during competition
- Optimize inference for <5ms per move

The key interface is already compatible:
```python
# Policy interface (ready for integration)
action, logprob, value = policy.act(observation)
```

## Testing Determinism

To verify deterministic behavior:

```bash
# Run training twice with same seed
python training/trainer.py  # First run
# Save checkpoint location

python training/trainer.py  # Second run
# Compare checkpoints - parameters should be identical
```

All random sources are seeded:
- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed`

## Troubleshooting

### ImportError
If you get import errors, ensure you're running from the repository root or that the path is set correctly.

### Memory Issues
If you run out of memory, reduce:
- `ROLLOUT_LENGTH` in `config.py` (e.g., 1024 instead of 2048)
- `BATCH_SIZE` in `config.py` (e.g., 256 instead of 512)

### Slow Training
Training speed depends on hardware. On a typical CPU:
- Rollout collection: ~10-20 seconds per 2048 steps
- PPO update: ~1-2 seconds per update

If training is very slow:
- Check `ROLLOUT_LENGTH` (longer = more time per update)
- Reduce `NUM_EPOCHS` (fewer gradient steps per rollout)

### CUDA Errors
This pipeline is CPU-only. If you see CUDA errors:
- Ensure PyTorch CPU version is installed
- Check that `DEVICE = "cpu"` in config
- Verify no tensors are being moved to GPU

### NaN Losses
If you see NaN in losses:
- Reduce learning rate (try 1e-4)
- Increase gradient clipping threshold
- Check for invalid observations (inf, nan values)

### Policy Collapse (Zero Entropy)
If entropy drops to zero too quickly:
- Increase `ENTROPY_COEFF` (try 0.02)
- Reduce learning rate
- Add observation normalization

## Notes

- **PR-3 Status**: This PR implements a full neural network policy with PPO training.
- **No Changes to Competition Code**: This PR does not modify `agent.py`, `case_closed_game.py`, or any judge engine files.
- **CPU-Only**: All operations use CPU tensors for compatibility with competition rules.
- **Deterministic**: All random seeds are set for reproducibility.
- **Model Size**: ~0.31 MB (well under 5MB limit)
- **Inference Speed**: <5ms per forward pass on typical CPUs

## Contact

For questions or issues with the training pipeline, please open an issue on the repository.

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
