"""
Configuration file for training hyperparameters and environment settings.
"""

import os
from datetime import datetime


class TrainingConfig:
    """Configuration class for training pipeline."""
    
    # Board dimensions (spec requires 15x15)
    BOARD_H = 15
    BOARD_W = 15
    
    # Random seed for reproducibility
    SEED = 42
    
    # Network architecture
    LATENT_SIZE = 128  # Keep model <5MB
    DEVICE = "cpu"  # device for training (CPU-only per competition rules)
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99  # discount factor
    GAE_LAMBDA = 0.95  # GAE lambda for advantage estimation
    PPO_CLIP_EPS = 0.2  # PPO clipping parameter
    VALUE_COEFF = 0.5  # value loss coefficient
    ENTROPY_COEFF = 0.01  # entropy bonus coefficient
    MAX_GRAD_NORM = 0.5  # gradient clipping
    MAX_KL = 0.03  # maximum KL divergence before early stopping
    VALUE_CLIP_RANGE = 0.2  # clipping range for value function (PPO2)
    
    # Observation normalization
    OBS_NORMALIZATION_EPS = 1e-8  # epsilon for numerical stability
    
    # Rollout settings
    ROLLOUT_LENGTH = 2048  # number of steps to collect per rollout
    BATCH_SIZE = 256  # minibatch size for PPO updates
    NUM_EPOCHS = 10  # number of optimization epochs per rollout
    
    # Training settings
    MAX_TRAINING_STEPS = 200_000  # maximum environment steps
    CHECKPOINT_INTERVAL = 50_000  # save checkpoint every N steps
    LOG_INTERVAL = 4096  # log metrics every N steps
    
    # Reward weights
    REWARD_SURVIVAL = 1.0  # reward per turn alive
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_DRAW = 0.0
    REWARD_SPACE_CONTROL = 0.01  # per Voronoi margin cell
    REWARD_MOBILITY = 0.002  # per reachable cell
    PENALTY_TRAP = -0.05  # when degree <= 1
    PENALTY_HEAD_ON = -1.0  # predicted head-on collision
    REWARD_BOOST_GOOD = 0.1  # boost increases reachable area
    REWARD_BOOST_BAD = -0.1  # boost decreases reachable area
    
    # Self-play settings
    USE_FROZEN_OPPONENT = True  # freeze opponent policy for stable self-play
    OPPONENT_UPDATE_INTERVAL = 5  # update opponent snapshot every N updates (not rollouts)
    OPPONENT_TYPE = "self"  # "self" or "heuristic"
    PLAYER_SWAP_PROB = 0.5  # probability of playing as player 2
    
    # Environment settings
    MAX_TURNS = 1000  # max turns per episode
    NUM_ACTIONS = 4  # Action space: 0=up, 1=right, 2=down, 3=left
    MAX_BOOST = 3  # maximum boosts per agent
    
    # Performance settings
    VORONOI_CAP = 150  # max cells to explore per player in Voronoi BFS
    FLOOD_FILL_CAP = 200  # max cells to explore in flood fill (mobility)
    
    # Logging and checkpointing
    RUN_NAME = None  # auto-generated if None: tronPPO_{YYYYmmdd_HHMMSS}_seed{SEED}
    LOG_DIR = "runs/"  # directory for logs
    CKPT_DIR = "training/checkpoints/"  # directory for checkpoints
    CSV_LOG = True  # enable CSV logging
    TENSORBOARD = False  # enable TensorBoard logging (optional)
    
    @classmethod
    def get_run_name(cls):
        """Generate run name if not set."""
        if cls.RUN_NAME is not None:
            return cls.RUN_NAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tronPPO_{timestamp}_seed{cls.SEED}"


# Create default config instance
config = TrainingConfig()
