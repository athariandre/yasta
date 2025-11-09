"""
Reward shaping module for Tron training.

Implements sophisticated reward terms to guide the agent towards good strategies:
- Survival rewards
- Space control (Voronoi)
- Mobility (reachable area)
- Trap avoidance
- Head-on collision avoidance
- Boost efficiency
"""

import numpy as np
from collections import deque
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_closed_game import GameResult
from training.config import config


def compute_voronoi_margin(my_head, opp_head, blocked, board_h=18, board_w=20, cap=150):
    """
    Compute Voronoi territory difference using dual BFS.
    
    Args:
        my_head: My head position (x, y)
        opp_head: Opponent head position (x, y)
        blocked: Set of blocked cells
        board_h: Board height
        board_w: Board width
        cap: Maximum cells to explore per player
    
    Returns:
        Margin (my_cells - opp_cells)
    """
    def torus(x, y):
        return (x % board_w, y % board_h)
    
    q = deque()
    seen = {}
    
    # Start BFS from both heads
    q.append((my_head, 1))  # player 1
    seen[my_head] = 1
    
    q.append((opp_head, 2))  # player 2
    seen[opp_head] = 2
    
    my_cells = 1
    opp_cells = 1
    
    total_explored = 0
    max_total = cap * 2
    
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
    
    while q and total_explored < max_total:
        (x, y), player = q.popleft()
        total_explored += 1
        
        for dx, dy in dirs:
            nx, ny = torus(x + dx, y + dy)
            if (nx, ny) not in blocked and (nx, ny) not in seen:
                seen[(nx, ny)] = player
                q.append(((nx, ny), player))
                if player == 1:
                    my_cells += 1
                else:
                    opp_cells += 1
    
    return my_cells - opp_cells


def compute_reachable_area(start, blocked, board_h=18, board_w=20, cap=200):
    """
    Compute reachable area from a starting position using flood fill.
    
    Args:
        start: Starting position (x, y)
        blocked: Set of blocked cells
        board_h: Board height
        board_w: Board width
        cap: Maximum cells to explore
    
    Returns:
        Number of reachable cells
    """
    def torus(x, y):
        return (x % board_w, y % board_h)
    
    if start in blocked:
        return 0
    
    q = deque([start])
    seen = {start}
    count = 0
    
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while q and count < cap:
        x, y = q.popleft()
        count += 1
        for dx, dy in dirs:
            nx, ny = torus(x + dx, y + dy)
            if (nx, ny) not in seen and (nx, ny) not in blocked:
                seen.add((nx, ny))
                q.append((nx, ny))
    
    return count


def compute_neighbor_degree(pos, blocked, board_h=18, board_w=20):
    """
    Count number of free neighbors around a position.
    
    Args:
        pos: Position to check (x, y)
        blocked: Set of blocked cells
        board_h: Board height
        board_w: Board width
    
    Returns:
        Number of free neighbors (0-4)
    """
    def torus(x, y):
        return (x % board_w, y % board_h)
    
    degree = 0
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for dx, dy in dirs:
        nx, ny = torus(pos[0] + dx, pos[1] + dy)
        if (nx, ny) not in blocked:
            degree += 1
    
    return degree


def get_blocked_cells(obs: Dict[str, np.ndarray]):
    """
    Extract set of blocked cells from observation.
    
    Args:
        obs: Observation dictionary with 'board' key
    
    Returns:
        Set of (x, y) tuples representing blocked cells
    """
    blocked = set()
    board = obs['board']
    h, w = board.shape
    
    for y in range(h):
        for x in range(w):
            if board[y, x] != 0:  # Non-empty cell
                blocked.add((x, y))
    
    return blocked


def compute_reward(prev_obs: Dict[str, np.ndarray], 
                   action: int, 
                   next_obs: Dict[str, np.ndarray], 
                   done: bool, 
                   outcome: GameResult = None,
                   boost_used: bool = False) -> float:
    """
    Compute shaped reward for a state transition.
    
    This function combines multiple reward terms:
    1. Survival reward (+1 per turn)
    2. Win/Loss/Draw terminal rewards
    3. Space control (Voronoi margin)
    4. Mobility (reachable area)
    5. Trap avoidance penalty
    6. Head-on collision penalty
    7. Boost efficiency
    
    Args:
        prev_obs: Previous observation dictionary
        action: Action taken (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        next_obs: Next observation dictionary
        done: Whether episode terminated
        outcome: GameResult if done, None otherwise
        boost_used: Whether boost was used this step
    
    Returns:
        Scalar reward (float)
    """
    reward = 0.0
    
    # 1. Survival Reward
    if not done:
        reward += config.REWARD_SURVIVAL
    else:
        # Terminal rewards
        if outcome == GameResult.AGENT1_WIN:
            reward += config.REWARD_WIN
        elif outcome == GameResult.AGENT2_WIN:
            reward += config.REWARD_LOSS
        elif outcome == GameResult.DRAW:
            reward += config.REWARD_DRAW
        
        # Early return if terminal state
        return reward
    
    # Extract state information
    my_head = tuple(next_obs['my_head'].astype(int))
    opp_head = tuple(next_obs['opp_head'].astype(int))
    blocked = get_blocked_cells(next_obs)
    
    prev_my_head = tuple(prev_obs['my_head'].astype(int))
    prev_blocked = get_blocked_cells(prev_obs)
    
    # 2. Space Control Reward (Voronoi margin)
    voronoi_margin = compute_voronoi_margin(
        my_head, opp_head, blocked,
        board_h=config.BOARD_H, board_w=config.BOARD_W
    )
    total_cells = config.BOARD_H * config.BOARD_W
    space_reward = config.REWARD_SPACE_CONTROL * (voronoi_margin / total_cells)
    reward += space_reward
    
    # 3. Mobility Reward (reachable area)
    reachable = compute_reachable_area(
        my_head, blocked,
        board_h=config.BOARD_H, board_w=config.BOARD_W
    )
    mobility_reward = config.REWARD_MOBILITY * reachable
    reward += mobility_reward
    
    # 4. Trap Avoidance Penalty
    degree = compute_neighbor_degree(
        my_head, blocked,
        board_h=config.BOARD_H, board_w=config.BOARD_W
    )
    if degree <= 1:
        trap_penalty = config.PENALTY_TRAP
        reward += trap_penalty
    
    # 5. Head-on Collision Penalty
    # Check if we moved into opponent's predicted position
    # Simple heuristic: if opponent was moving towards us and we're now at same position
    if my_head == opp_head:
        head_on_penalty = config.PENALTY_HEAD_ON
        reward += head_on_penalty
    
    # 6. Boost Efficiency
    if boost_used:
        # Compare reachable area before and after boost
        prev_reachable = compute_reachable_area(
            prev_my_head, prev_blocked,
            board_h=config.BOARD_H, board_w=config.BOARD_W
        )
        
        if reachable > prev_reachable:
            # Boost increased our mobility - good!
            reward += config.REWARD_BOOST_GOOD
        else:
            # Boost didn't help or made things worse
            reward += config.REWARD_BOOST_BAD
    
    return reward


def compute_batch_rewards(prev_obs_batch, actions_batch, next_obs_batch, 
                          dones_batch, outcomes_batch, boosts_batch):
    """
    Compute rewards for a batch of transitions.
    
    Args:
        prev_obs_batch: List of previous observations
        actions_batch: List of actions
        next_obs_batch: List of next observations
        dones_batch: List of done flags
        outcomes_batch: List of GameResult or None
        boosts_batch: List of boost flags
    
    Returns:
        numpy array of rewards
    """
    rewards = []
    
    for i in range(len(prev_obs_batch)):
        reward = compute_reward(
            prev_obs_batch[i],
            actions_batch[i],
            next_obs_batch[i],
            dones_batch[i],
            outcomes_batch[i],
            boosts_batch[i]
        )
        rewards.append(reward)
    
    return np.array(rewards, dtype=np.float32)
