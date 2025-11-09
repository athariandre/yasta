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
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99  # discount factor
    GAE_LAMBDA = 0.95  # GAE lambda for advantage estimation
    CLIP_EPSILON = 0.2  # PPO clipping parameter
    VALUE_COEF = 0.5  # value loss coefficient
    ENTROPY_COEF = 0.01  # entropy bonus coefficient
    MAX_GRAD_NORM = 0.5  # gradient clipping
    
    # Rollout settings
    ROLLOUT_LENGTH = 2048  # number of steps to collect per rollout
    BATCH_SIZE = 64  # minibatch size for PPO updates
    NUM_EPOCHS = 10  # number of optimization epochs per rollout
    
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

# Create default config instance
config = TrainingConfig()
