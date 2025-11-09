"""
Configuration file for training hyperparameters and environment settings.
"""

class TrainingConfig:
    """Configuration class for training pipeline."""
    
    # Board dimensions (from case_closed_game.py)
    BOARD_H = 18
    BOARD_W = 20
    
    # Random seed for reproducibility
    SEED = 42
    
    # Network architecture
    LATENT_SIZE = 128  # size of latent feature vector
    DEVICE = "cpu"  # device for training (CPU-only per competition rules)
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99  # discount factor
    GAE_LAMBDA = 0.95  # GAE lambda for advantage estimation
    PPO_CLIP_EPS = 0.2  # PPO clipping parameter (renamed for clarity)
    CLIP_EPSILON = 0.2  # PPO clipping parameter (kept for backward compatibility)
    VALUE_COEFF = 0.5  # value loss coefficient (renamed for consistency)
    VALUE_COEF = 0.5  # value loss coefficient (kept for backward compatibility)
    ENTROPY_COEFF = 0.01  # entropy bonus coefficient (renamed for consistency)
    ENTROPY_COEF = 0.01  # entropy bonus coefficient (kept for backward compatibility)
    MAX_GRAD_NORM = 0.5  # gradient clipping
    
    # Observation normalization
    OBS_NORMALIZATION_EPS = 1e-8  # epsilon for numerical stability
    
    # Rollout settings
    ROLLOUT_LENGTH = 2048  # number of steps to collect per rollout
    BATCH_SIZE = 512  # minibatch size for PPO updates
    NUM_EPOCHS = 4  # number of optimization epochs per rollout
    
    # Training settings
    MAX_TRAINING_STEPS = 1_000_000  # maximum environment steps
    CHECKPOINT_INTERVAL = 10_000  # save checkpoint every N steps
    LOG_INTERVAL = 1000  # log metrics every N steps
    
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
    OPPONENT_TYPE = "self"  # "self" or "heuristic"
    PLAYER_SWAP_PROB = 0.5  # probability of playing as player 2
    
    # Environment settings
    MAX_TURNS = 200  # max turns per episode (from game rules)
    NUM_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT
    
    # Performance settings
    VORONOI_CAP = 150  # max cells to explore per player in Voronoi BFS
    FLOOD_FILL_CAP = 200  # max cells to explore in flood fill (mobility)
    
    # PPO stability settings (PR-3 fixes)
    KL_TARGET = 0.01  # target KL divergence for early stopping
    MAX_KL = 0.05  # maximum KL divergence before stopping epoch
    VALUE_CLIP_RANGE = 0.2  # clipping range for value function (PPO2)
    OPPONENT_UPDATE_INTERVAL = 50  # update opponent snapshot every N rollouts (was 10, increased for stability)
    USE_FROZEN_OPPONENT = True  # freeze opponent policy for stable self-play
    MAX_BOOST = 3  # maximum boosts per agent

# Create default config instance
config = TrainingConfig()
