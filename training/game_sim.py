"""
Local game simulation wrapper over the official case_closed_game.py classes.

This module provides a clean API for running deterministic game simulations
without depending on Flask or the judge engine.
"""

import random
import copy
import numpy as np
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_closed_game import Game, Direction, GameResult, Agent, GameBoard


class LocalGame:
    """
    A wrapper around the official Game class that provides an RL-friendly interface.
    
    This class enables:
    - Deterministic game resets with seeding
    - State observation as numpy arrays
    - Step-by-step game progression
    - Game state cloning for lookahead
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize a new local game.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.game = None
        self.reset()
    
    def reset(self):
        """
        Reset the game to initial state.
        
        Returns:
            Initial observation for player 1
        """
        # Note: Global RNG seeding is handled by Trainer.set_seed()
        # We rely on the trainer's seed policy for deterministic behavior
        # rather than re-seeding here to avoid conflicts with other modules
        
        # Create new game instance
        self.game = Game()
        self.game.reset()
        
        return self.get_obs(player_number=1)
    
    def step(self, action_p1: Direction, action_p2: Direction, 
             boost_p1: bool = False, boost_p2: bool = False):
        """
        Execute one game step with both players' actions.
        
        Args:
            action_p1: Direction for player 1
            action_p2: Direction for player 2
            boost_p1: Whether player 1 uses boost
            boost_p2: Whether player 2 uses boost
        
        Returns:
            Tuple of (obs_p1, obs_p2, reward_p1, reward_p2, done, info)
        """
        # Execute the step in the official game
        result = self.game.step(action_p1, action_p2, boost_p1, boost_p2)
        
        # Get observations
        obs_p1 = self.get_obs(player_number=1)
        obs_p2 = self.get_obs(player_number=2)
        
        # Determine if game is done
        done = result is not None
        
        # Compute basic rewards (detailed reward shaping is in rewards.py)
        reward_p1 = 0.0
        reward_p2 = 0.0
        
        if done:
            if result == GameResult.AGENT1_WIN:
                reward_p1 = 100.0
                reward_p2 = -100.0
            elif result == GameResult.AGENT2_WIN:
                reward_p1 = -100.0
                reward_p2 = 100.0
            elif result == GameResult.DRAW:
                reward_p1 = 0.0
                reward_p2 = 0.0
        else:
            # Survival reward
            reward_p1 = 1.0
            reward_p2 = 1.0
        
        # Create info dict
        info = {
            'result': result,
            'turns': self.game.turns,
            'agent1_alive': self.game.agent1.alive,
            'agent2_alive': self.game.agent2.alive,
            'agent1_boosts': self.game.agent1.boosts_remaining,
            'agent2_boosts': self.game.agent2.boosts_remaining,
            'agent1_length': self.game.agent1.length,
            'agent2_length': self.game.agent2.length,
        }
        
        return obs_p1, obs_p2, reward_p1, reward_p2, done, info
    
    def get_obs(self, player_number: int):
        """
        Get current observation for a player as numpy arrays.
        
        Observation includes:
        - Board grid (H x W)
        - Both head positions (2 x 2)
        - Trail positions for both agents (lists)
        - Boosts remaining for both agents (2,)
        - Turn count (scalar)
        - Previous direction for both agents (2,)
        
        Args:
            player_number: 1 or 2
        
        Returns:
            Dictionary of numpy arrays (CPU only)
        """
        # Get board grid as numpy array
        board = np.array(self.game.board.grid, dtype=np.float32)
        
        # Get agent references
        my_agent = self.game.agent1 if player_number == 1 else self.game.agent2
        opp_agent = self.game.agent2 if player_number == 1 else self.game.agent1
        
        # Get head positions
        my_head = np.array(my_agent.trail[-1] if my_agent.trail else [0, 0], dtype=np.float32)
        opp_head = np.array(opp_agent.trail[-1] if opp_agent.trail else [0, 0], dtype=np.float32)
        
        # Get trails
        my_trail = np.array(list(my_agent.trail), dtype=np.float32) if my_agent.trail else np.zeros((0, 2), dtype=np.float32)
        opp_trail = np.array(list(opp_agent.trail), dtype=np.float32) if opp_agent.trail else np.zeros((0, 2), dtype=np.float32)
        
        # Get boosts remaining
        boosts = np.array([my_agent.boosts_remaining, opp_agent.boosts_remaining], dtype=np.float32)
        
        # Get turn count
        turns = np.array([self.game.turns], dtype=np.float32)
        
        # Get previous directions (encoded as integers)
        direction_encoding = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3,
        }
        
        my_dir = direction_encoding.get(my_agent.direction, 0)
        opp_dir = direction_encoding.get(opp_agent.direction, 0)
        directions = np.array([my_dir, opp_dir], dtype=np.float32)
        
        # Get alive status
        alive = np.array([my_agent.alive, opp_agent.alive], dtype=np.float32)
        
        # Return observation dictionary
        obs = {
            'board': board,
            'my_head': my_head,
            'opp_head': opp_head,
            'my_trail': my_trail,
            'opp_trail': opp_trail,
            'boosts': boosts,
            'turns': turns,
            'directions': directions,
            'alive': alive,
        }
        
        return obs
    
    def is_done(self):
        """
        Check if the game is finished.
        
        Returns:
            True if game is over, False otherwise
        """
        # Game is done if either agent is dead or max turns reached
        if not self.game.agent1.alive or not self.game.agent2.alive:
            return True
        if self.game.turns >= 200:
            return True
        return False
    
    def clone(self):
        """
        Create a deep copy of the current game state.
        
        This is useful for lookahead search or rollout simulations.
        
        Returns:
            A new LocalGame instance with copied state
        """
        cloned = LocalGame(seed=self.seed)
        
        # Deep copy the game state
        cloned.game = Game()
        
        # Copy board
        cloned.game.board = GameBoard(height=self.game.board.height, width=self.game.board.width)
        cloned.game.board.grid = [row[:] for row in self.game.board.grid]
        
        # Copy agent 1
        cloned.game.agent1 = Agent(
            agent_id=self.game.agent1.agent_id,
            start_pos=self.game.agent1.trail[0],
            start_dir=self.game.agent1.direction,
            board=cloned.game.board
        )
        cloned.game.agent1.trail = deque(self.game.agent1.trail)
        cloned.game.agent1.direction = self.game.agent1.direction
        cloned.game.agent1.alive = self.game.agent1.alive
        cloned.game.agent1.length = self.game.agent1.length
        cloned.game.agent1.boosts_remaining = self.game.agent1.boosts_remaining
        
        # Copy agent 2
        cloned.game.agent2 = Agent(
            agent_id=self.game.agent2.agent_id,
            start_pos=self.game.agent2.trail[0],
            start_dir=self.game.agent2.direction,
            board=cloned.game.board
        )
        cloned.game.agent2.trail = deque(self.game.agent2.trail)
        cloned.game.agent2.direction = self.game.agent2.direction
        cloned.game.agent2.alive = self.game.agent2.alive
        cloned.game.agent2.length = self.game.agent2.length
        cloned.game.agent2.boosts_remaining = self.game.agent2.boosts_remaining
        
        # Copy turns
        cloned.game.turns = self.game.turns
        
        return cloned
    
    def get_result(self):
        """
        Get the final game result (if game is done).
        
        Returns:
            GameResult enum or None if game is not done
        """
        if not self.is_done():
            return None
        
        if not self.game.agent1.alive and not self.game.agent2.alive:
            return GameResult.DRAW
        elif not self.game.agent1.alive:
            return GameResult.AGENT2_WIN
        elif not self.game.agent2.alive:
            return GameResult.AGENT1_WIN
        else:
            # Max turns reached, compare trail lengths
            if self.game.agent1.length > self.game.agent2.length:
                return GameResult.AGENT1_WIN
            elif self.game.agent2.length > self.game.agent1.length:
                return GameResult.AGENT2_WIN
            else:
                return GameResult.DRAW
