"""
Kuhn Poker implementation for Student of Games.

Kuhn Poker is a simplified poker game that serves as a classic example
in game theory and computational game theory research.
"""

import copy
import numpy as np
from typing import List, Any, Dict
from .base import Game, Action


class KuhnPokerState:
    """State representation for Kuhn Poker."""
    
    def __init__(self):
        # Cards: 0=Jack, 1=Queen, 2=King
        self.cards = [0, 1, 2]
        self.player_cards = [None, None]  # Cards for each player
        self.history = []  # Sequence of actions
        self.pot = [1, 1]  # Ante for each player
        self.current_player = 0
        self.terminal = False
        self.folded = [False, False]
        
        # Deal cards randomly
        np.random.shuffle(self.cards)
        self.player_cards[0] = self.cards[0]
        self.player_cards[1] = self.cards[1]
    
    def clone(self):
        """Create a deep copy of the state."""
        new_state = KuhnPokerState()
        new_state.cards = self.cards.copy()
        new_state.player_cards = self.player_cards.copy()
        new_state.history = self.history.copy()
        new_state.pot = self.pot.copy()
        new_state.current_player = self.current_player
        new_state.terminal = self.terminal
        new_state.folded = self.folded.copy()
        return new_state


class KuhnPoker(Game):
    """
    Kuhn Poker game implementation.
    
    Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - 2 players, each dealt 1 card
    - Each player antes 1 chip
    - Player 0 acts first, then Player 1
    - Actions: Check/Call (0), Bet/Fold (1)
    - If both check, higher card wins
    - If one bets and other calls, higher card wins
    - If one bets and other folds, better wins
    """
    
    def __init__(self):
        self.num_actions_per_state = 2  # Check/Call or Bet/Fold
        self.num_players_val = 2
    
    def initial_state(self) -> KuhnPokerState:
        """Return initial game state."""
        return KuhnPokerState()
    
    def current_player(self, state: KuhnPokerState) -> int:
        """Return current player."""
        return state.current_player
    
    def legal_actions(self, state: KuhnPokerState) -> List[int]:
        """Return legal actions. 0=Check/Call, 1=Bet/Fold"""
        if state.terminal:
            return []
        return [0, 1]
    
    def apply_action(self, state: KuhnPokerState, action: int) -> KuhnPokerState:
        """Apply action and return new state."""
        new_state = state.clone()
        new_state.history.append(action)
        
        # Check if game should end
        if len(new_state.history) == 1:
            # First action
            if action == 0:  # Check
                new_state.current_player = 1
            else:  # Bet
                new_state.pot[0] += 1  # Player 0 bets
                new_state.current_player = 1
        
        elif len(new_state.history) == 2:
            # Second action
            if new_state.history[0] == 0:  # First player checked
                if action == 0:  # Second player checks
                    new_state.terminal = True
                else:  # Second player bets
                    new_state.pot[1] += 1
                    new_state.current_player = 0  # Back to player 0 for response
            else:  # First player bet
                if action == 0:  # Second player calls
                    new_state.pot[1] += 1
                    new_state.terminal = True
                else:  # Second player folds
                    new_state.folded[1] = True
                    new_state.terminal = True
        
        elif len(new_state.history) == 3:
            # Third action (response to bet)
            if action == 0:  # Call
                new_state.pot[0] += 1
                new_state.terminal = True
            else:  # Fold
                new_state.folded[0] = True
                new_state.terminal = True
        
        return new_state
    
    def is_terminal(self, state: KuhnPokerState) -> bool:
        """Check if state is terminal."""
        return state.terminal
    
    def returns(self, state: KuhnPokerState) -> np.ndarray:
        """Return utilities for each player."""
        if not state.terminal:
            return np.array([0.0, 0.0])
        
        # Check for folds
        if state.folded[0]:
            # Player 0 folded, Player 1 wins
            return np.array([-state.pot[0], state.pot[0]])
        elif state.folded[1]:
            # Player 1 folded, Player 0 wins
            return np.array([state.pot[1], -state.pot[1]])
        
        # Showdown - higher card wins
        if state.player_cards[0] > state.player_cards[1]:
            # Player 0 wins
            pot_size = state.pot[1]
            return np.array([pot_size, -pot_size])
        else:
            # Player 1 wins
            pot_size = state.pot[0]
            return np.array([-pot_size, pot_size])
    
    def num_players(self) -> int:
        """Return number of players."""
        return self.num_players_val
    
    def num_actions(self) -> int:
        """Return maximum number of actions."""
        return self.num_actions_per_state
    
    def information_state_string(self, state: KuhnPokerState, player: int) -> str:
        """Return information state string for player."""
        # Player sees their own card and the history
        card = state.player_cards[player]
        history_str = "".join(map(str, state.history))
        return f"card{card}_history{history_str}_player{player}"
    
    def state_to_features(self, state: KuhnPokerState) -> np.ndarray:
        """Convert state to feature vector."""
        # Feature vector: [my_card_0, my_card_1, my_card_2, history_features, pot_features]
        features = np.zeros(12)  # Adjust size as needed
        
        # Current player's card (one-hot)
        if state.current_player == 0:
            card = state.player_cards[0]
        else:
            card = state.player_cards[1]
        
        features[card] = 1.0  # One-hot encoding for card
        
        # History features
        for i, action in enumerate(state.history):
            if i < 4:  # Max 4 actions
                features[3 + i] = action
        
        # Pot features
        features[7] = state.pot[0] / 4.0  # Normalize pot sizes
        features[8] = state.pot[1] / 4.0
        
        # Game phase
        features[9] = len(state.history) / 4.0  # Normalize history length
        
        # Current player
        features[10] = state.current_player
        
        # Terminal flag
        features[11] = 1.0 if state.terminal else 0.0
        
        return features
    
    def get_public_state(self, state: KuhnPokerState) -> Dict[str, Any]:
        """Get public information visible to all players."""
        return {
            'history': state.history.copy(),
            'pot': state.pot.copy(),
            'current_player': state.current_player,
            'terminal': state.terminal,
            'folded': state.folded.copy()
        }
    
    def get_private_state(self, state: KuhnPokerState, player: int) -> Dict[str, Any]:
        """Get private information for a player."""
        return {
            'card': state.player_cards[player]
        }
    
    def max_game_length(self) -> int:
        """Maximum possible game length."""
        return 3  # At most 3 actions in Kuhn Poker
    
    def game_type(self) -> Dict[str, Any]:
        """Game type information."""
        return {
            "perfect_information": False,
            "zero_sum": True,
            "simultaneous": False,
            "chance_mode": True  # Cards are dealt randomly
        }
    
    def chance_outcomes(self, state: KuhnPokerState) -> List[tuple]:
        """Return chance outcomes for card dealing."""
        # This would be used at the start of the game for card dealing
        # For simplicity, we handle this in initial_state()
        return []
    
    def is_chance_node(self, state: KuhnPokerState) -> bool:
        """Check if this is a chance node."""
        # Card dealing happens in initial_state, so no chance nodes during play
        return False
    
    def __str__(self) -> str:
        return "KuhnPoker"


def print_game_tree():
    """Print the complete game tree for Kuhn Poker (for debugging/analysis)."""
    game = KuhnPoker()
    
    def print_node(state, depth=0, prefix=""):
        indent = "  " * depth
        if state.terminal:
            returns = game.returns(state)
            print(f"{indent}{prefix}Terminal: {returns}")
            return
        
        player = game.current_player(state)
        card = state.player_cards[player]
        history = "".join(map(str, state.history))
        print(f"{indent}{prefix}Player {player} (Card {card}) History: {history}")
        
        for action in game.legal_actions(state):
            new_state = game.apply_action(state, action)
            action_name = "Check/Call" if action == 0 else "Bet/Fold"
            print_node(new_state, depth + 1, f"{action_name} -> ")
    
    # This would be too large to print for all possible card deals
    # So we'll just show the structure for one deal
    state = game.initial_state()
    print(f"Initial state: P0 has card {state.player_cards[0]}, P1 has card {state.player_cards[1]}")
    print_node(state)


if __name__ == "__main__":
    # Example usage
    game = KuhnPoker()
    state = game.initial_state()
    
    print("=== Kuhn Poker Example ===")
    print(f"Initial state: P0 card={state.player_cards[0]}, P1 card={state.player_cards[1]}")
    
    # Play a sample game
    while not game.is_terminal(state):
        player = game.current_player(state)
        legal_actions = game.legal_actions(state)
        print(f"Player {player} to act. Legal actions: {legal_actions}")
        print(f"Information state: {game.information_state_string(state, player)}")
        
        # Random action for demo
        action = np.random.choice(legal_actions)
        action_name = "Check/Call" if action == 0 else "Bet/Fold"
        print(f"Player {player} chooses: {action_name}")
        
        state = game.apply_action(state, action)
    
    returns = game.returns(state)
    print(f"Game over! Returns: {returns}")
    print(f"Winner: Player {np.argmax(returns)}")