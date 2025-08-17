"""
Abstract base classes for games in Student of Games.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import numpy as np


class Action:
    """Base class for game actions."""
    
    def __init__(self, value: Any):
        self.value = value
    
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"Action({self.value})"
    
    def __str__(self):
        return str(self.value)


class GameState:
    """Base class for game states."""
    
    def __init__(self):
        self.history = []
        self.current_player = 0
        self.terminal = False
    
    def apply_action(self, action: Action):
        """Apply an action and return a new state."""
        raise NotImplementedError
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.terminal
    
    def get_current_player(self) -> int:
        """Get the current player to act."""
        return self.current_player
    
    def clone(self):
        """Create a deep copy of this state."""
        raise NotImplementedError


class Game(ABC):
    """
    Abstract base class for games that can be used with Student of Games.
    
    This interface supports both perfect and imperfect information games.
    """
    
    @abstractmethod
    def initial_state(self) -> Any:
        """Return the initial state of the game."""
        pass
    
    @abstractmethod
    def current_player(self, state: Any) -> int:
        """Return the current player to act in the given state."""
        pass
    
    @abstractmethod
    def legal_actions(self, state: Any) -> List[Any]:
        """Return a list of legal actions in the given state."""
        pass
    
    @abstractmethod
    def apply_action(self, state: Any, action: Any) -> Any:
        """Apply an action to a state and return the new state."""
        pass
    
    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if the given state is terminal."""
        pass
    
    @abstractmethod
    def returns(self, state: Any) -> np.ndarray:
        """Return the utility for each player in a terminal state."""
        pass
    
    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players in the game."""
        pass
    
    @abstractmethod
    def num_actions(self) -> int:
        """Return the maximum number of actions in any state."""
        pass
    
    @abstractmethod
    def information_state_string(self, state: Any, player: int) -> str:
        """
        Return a string representation of the information state for a player.
        
        This should capture all information available to the player, but not
        information that should be hidden (like opponent's private cards).
        """
        pass
    
    @abstractmethod
    def state_to_features(self, state: Any) -> np.ndarray:
        """
        Convert a game state to a feature vector for neural networks.
        
        This should create a fixed-size numerical representation suitable
        for input to the neural network.
        """
        pass
    
    def observation_string(self, state: Any, player: int) -> str:
        """
        Return a string representation of what a player observes.
        
        Default implementation uses information_state_string.
        """
        return self.information_state_string(state, player)
    
    def is_chance_node(self, state: Any) -> bool:
        """
        Check if the current state is a chance node.
        
        Chance nodes are where random events occur (like dealing cards).
        Default implementation assumes no chance nodes.
        """
        return False
    
    def chance_outcomes(self, state: Any) -> List[tuple]:
        """
        Return chance outcomes and their probabilities.
        
        Returns list of (outcome, probability) tuples.
        Only relevant for games with chance nodes.
        """
        return []
    
    def get_public_state(self, state: Any) -> Any:
        """
        Get the public state visible to all players.
        
        For perfect information games, this is the same as the full state.
        For imperfect information games, this excludes private information.
        """
        return state
    
    def get_private_state(self, state: Any, player: int) -> Any:
        """
        Get the private state for a specific player.
        
        This includes information known only to that player.
        """
        return None
    
    def max_game_length(self) -> int:
        """
        Return the maximum possible length of a game.
        
        Used for allocating arrays and setting timeouts.
        """
        return 1000  # Default reasonable value
    
    def game_type(self) -> Dict[str, Any]:
        """
        Return information about the game type.
        
        This can include properties like:
        - perfect_information: bool
        - zero_sum: bool
        - simultaneous: bool
        - etc.
        """
        return {
            "perfect_information": True,
            "zero_sum": True,
            "simultaneous": False,
            "chance_mode": False
        }
    
    def __str__(self) -> str:
        """String representation of the game."""
        return self.__class__.__name__