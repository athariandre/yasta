"""
RL environment wrapper for Tron game with self-play support.

This module provides a single-agent RL interface that handles:
- Self-play against a copy of the agent
- Self-play against a heuristic agent
- Symmetric experience collection (50% as player 1, 50% as player 2)
- Action space conversion (indices to Direction enums)
"""

import random
import numpy as np
from typing import Dict, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_closed_game import Direction, GameResult
from training.game_sim import LocalGame
from training.rewards import compute_reward
from training.config import config


class TronEnv:
    """
    Single-agent RL environment for Tron with self-play.
    
    The environment manages:
    - Game simulation
    - Opponent agent (copy or heuristic)
    - Player role swapping (player 1 vs player 2)
    - Observation and reward computation
    """
    
    # Action space: 0=up, 1=right, 2=down, 3=left (per spec)
    ACTION_TO_DIRECTION = {
        0: Direction.UP,
        1: Direction.RIGHT,
        2: Direction.DOWN,
        3: Direction.LEFT,
    }
    
    DIRECTION_TO_ACTION = {
        Direction.UP: 0,
        Direction.RIGHT: 1,
        Direction.DOWN: 2,
        Direction.LEFT: 3,
    }
    
    def __init__(self, opponent_policy=None, use_heuristic_opponent=False, seed=None):
        """
        Initialize the Tron environment.
        
        Args:
            opponent_policy: Policy object for opponent (if using self-play)
            use_heuristic_opponent: If True, use heuristic agent as opponent
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else config.SEED
        self.rng = random.Random(self.seed)
        # Note: Global numpy seeding is handled by Trainer.set_seed()
        # We use a local random.Random instance for env-specific randomness
        
        self.opponent_policy = opponent_policy
        self.use_heuristic_opponent = use_heuristic_opponent
        
        self.game = LocalGame(seed=self.seed)
        
        # Track which player we are (1 or 2) - swap for symmetric self-play
        self.my_player_number = 1
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Previous observation for reward computation
        self.prev_obs = None
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation dictionary
        """
        # Randomly swap player roles for symmetric self-play
        if self.rng.random() < config.PLAYER_SWAP_PROB:
            self.my_player_number = 2
        else:
            self.my_player_number = 1
        
        # Reset game
        self.game.reset()
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Get initial observation
        obs = self.game.get_obs(self.my_player_number)
        self.prev_obs = obs
        
        return obs
    
    def _get_opponent_action(self, obs):
        """
        Get opponent's action based on current observation.
        
        Args:
            obs: Opponent's observation
        
        Returns:
            Opponent action (0-3) and boost flag
        """
        if self.use_heuristic_opponent:
            # Use simple heuristic for opponent
            return self._heuristic_action(obs), False
        elif self.opponent_policy is not None:
            # Use policy for opponent (self-play)
            action, _, _ = self.opponent_policy.act(obs)
            return action, False
        else:
            # Random opponent as fallback
            return self.rng.randint(0, 3), False
    
    def _heuristic_action(self, obs):
        """
        Simple heuristic policy for opponent.
        
        This is a basic survival heuristic that tries to:
        1. Avoid immediate collisions
        2. Prefer moves with more open space
        
        TODO (PR-4): Allow head-on when winning by trail length or when fully boxed
        Currently this heuristic avoids all blocked cells including opponent head,
        which means it will never select a deliberate head-on collision even when
        that would be optimal (e.g., winning on length in endgame).
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Action index (0-3)
        """
        my_head = tuple(obs['my_head'].astype(int))
        board = obs['board']
        
        # Get blocked cells
        blocked = set()
        h, w = board.shape
        for y in range(h):
            for x in range(w):
                if board[y, x] != 0:
                    blocked.add((x, y))
        
        # Try each action and count free neighbors
        best_action = 0
        best_score = -1
        
        for action in range(4):
            direction = self.ACTION_TO_DIRECTION[action]
            dx, dy = direction.value
            new_x = (my_head[0] + dx) % config.BOARD_W
            new_y = (my_head[1] + dy) % config.BOARD_H
            new_pos = (new_x, new_y)
            
            # Check if move is valid
            if new_pos in blocked:
                continue
            
            # Count free neighbors from new position
            free_count = 0
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx = (new_pos[0] + ddx) % config.BOARD_W
                ny = (new_pos[1] + ddy) % config.BOARD_H
                if (nx, ny) not in blocked:
                    free_count += 1
            
            if free_count > best_score:
                best_score = free_count
                best_action = action
        
        return best_action
    
    def step(self, action_self: int, use_boost_self: bool = False):
        """
        Execute one environment step.
        
        Args:
            action_self: Our action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            use_boost_self: Whether we use boost
        
        Returns:
            Tuple of (obs, reward, done, info)
        """
        # Get opponent's observation and action
        opp_player_number = 2 if self.my_player_number == 1 else 1
        opp_obs = self.game.get_obs(opp_player_number)
        action_opp, use_boost_opp = self._get_opponent_action(opp_obs)
        
        # Convert actions to Directions
        dir_self = self.ACTION_TO_DIRECTION[action_self]
        dir_opp = self.ACTION_TO_DIRECTION[action_opp]
        
        # Determine which player is which
        if self.my_player_number == 1:
            dir_p1 = dir_self
            dir_p2 = dir_opp
            boost_p1 = use_boost_self
            boost_p2 = use_boost_opp
        else:
            dir_p1 = dir_opp
            dir_p2 = dir_self
            boost_p1 = use_boost_opp
            boost_p2 = use_boost_self
        
        # Execute game step
        obs_p1, obs_p2, reward_p1, reward_p2, done, info = self.game.step(
            dir_p1, dir_p2, boost_p1, boost_p2
        )
        
        # Get our observation and reward
        if self.my_player_number == 1:
            obs = obs_p1
            outcome = info['result']
        else:
            obs = obs_p2
            # Flip outcome for player 2 perspective
            if info['result'] == GameResult.AGENT1_WIN:
                outcome = GameResult.AGENT2_WIN
            elif info['result'] == GameResult.AGENT2_WIN:
                outcome = GameResult.AGENT1_WIN
            else:
                outcome = info['result']
        
        # Compute shaped reward
        reward = compute_reward(
            self.prev_obs,
            action_self,
            obs,
            done,
            outcome,
            use_boost_self
        )
        
        # Update tracking
        self.episode_steps += 1
        self.episode_reward += reward
        self.prev_obs = obs
        
        # Add episode info with both engine result and POV result
        info['episode_steps'] = self.episode_steps
        info['episode_reward'] = self.episode_reward
        info['my_player_number'] = self.my_player_number
        info['result_pov'] = outcome  # My perspective (win/loss/draw)
        # info['result'] remains the engine's perspective for debugging
        
        return obs, reward, done, info
    
    def render(self):
        """
        Render the current game state (optional, for debugging).
        
        Returns:
            String representation of the board
        """
        return str(self.game.game.board)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_observation_space(self):
        """
        Get information about observation space.
        
        Returns:
            Dictionary describing observation structure
        """
        return {
            'board': (config.BOARD_H, config.BOARD_W),
            'my_head': (2,),
            'opp_head': (2,),
            'boosts': (2,),
            'turns': (1,),
            'directions': (2,),
            'alive': (2,),
        }
    
    def get_action_space(self):
        """
        Get information about action space.
        
        Returns:
            Number of discrete actions
        """
        return config.NUM_ACTIONS
