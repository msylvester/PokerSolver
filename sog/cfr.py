"""
Growing-Tree Counterfactual Regret Minimization (GT-CFR) implementation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import copy
from .core import PublicBeliefState


class CFRNode:
    """Node in the CFR game tree."""
    
    def __init__(self, info_state: str, legal_actions: List[Any]):
        self.info_state = info_state
        self.legal_actions = legal_actions
        self.num_actions = len(legal_actions)
        
        # Regret and strategy sums
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        
        # Current strategy
        self.strategy = np.ones(self.num_actions) / self.num_actions
        
        # Network predictions cache
        self.network_policy = None
        self.network_values = None
        
    def update_strategy(self):
        """Update strategy based on regret matching."""
        # Regret matching
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            self.strategy = positive_regrets / normalizing_sum
        else:
            # Uniform random strategy
            self.strategy = np.ones(self.num_actions) / self.num_actions
    
    def get_average_strategy(self) -> np.ndarray:
        """Get the average strategy over all iterations."""
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions


class GrowingTreeCFR:
    """
    Growing-Tree Counterfactual Regret Minimization algorithm.
    
    This is the core algorithm that combines CFR with neural network
    function approximation for the Student of Games approach.
    """
    
    def __init__(
        self,
        game,
        network=None,
        use_network: bool = True,
        network_threshold: int = 100,
        cfr_iterations: int = 100
    ):
        self.game = game
        self.network = network
        self.use_network = use_network
        self.network_threshold = network_threshold
        self.cfr_iterations = cfr_iterations
        
        # Game tree nodes
        self.nodes: Dict[str, CFRNode] = {}
        
        # Public belief states mapping
        self.belief_states: Dict[str, PublicBeliefState] = {}
        
        # Iteration counter
        self.iteration = 0
        
    def cfr(
        self, 
        state,
        reach_probabilities: np.ndarray,
        player: int
    ) -> np.ndarray:
        """
        Counterfactual Regret Minimization recursive algorithm.
        
        Args:
            state: Current game state
            reach_probabilities: Reach probabilities for each player
            player: Player to compute CFR for
            
        Returns:
            Expected utility for each player
        """
        if self.game.is_terminal(state):
            return self.game.returns(state)
        
        current_player = self.game.current_player(state)
        info_state = self.game.information_state_string(state, current_player)
        legal_actions = self.game.legal_actions(state)
        
        # Get or create node
        if info_state not in self.nodes:
            self.nodes[info_state] = CFRNode(info_state, legal_actions)
        
        node = self.nodes[info_state]
        
        # Update strategy
        node.update_strategy()
        
        # Use network predictions if available and appropriate
        if (self.use_network and self.network is not None and 
            self.iteration > self.network_threshold):
            self._update_node_with_network(node, state)
        
        # Compute action utilities
        action_utilities = np.zeros((len(legal_actions), self.game.num_players()))
        
        for i, action in enumerate(legal_actions):
            new_state = self.game.apply_action(state, action)
            
            # Update reach probabilities
            new_reach_probs = reach_probabilities.copy()
            if current_player == player:
                new_reach_probs[current_player] *= node.strategy[i]
            
            action_utilities[i] = self.cfr(new_state, new_reach_probs, player)
        
        # Compute expected utility
        expected_utility = np.sum(
            action_utilities * node.strategy.reshape(-1, 1), 
            axis=0
        )
        
        # Update regrets and strategy sum
        if current_player == player:
            for i in range(len(legal_actions)):
                regret = action_utilities[i, player] - expected_utility[player]
                node.regret_sum[i] += reach_probabilities[1 - player] * regret
            
            # Update strategy sum
            node.strategy_sum += reach_probabilities[player] * node.strategy
        
        return expected_utility
    
    def _update_node_with_network(self, node: CFRNode, state):
        """Update node with network predictions."""
        if self.network is None:
            return
        
        # Get state features (this would be game-specific)
        state_features = self._state_to_features(state)
        legal_mask = self._get_legal_actions_mask(state)
        
        # Get network predictions
        policy_probs, values = self.network.predict(state_features, legal_mask)
        
        # Cache network predictions
        node.network_policy = policy_probs
        node.network_values = values
        
        # Blend network policy with CFR strategy
        alpha = 0.8  # Network weight
        node.strategy = alpha * policy_probs + (1 - alpha) * node.strategy
    
    def _state_to_features(self, state) -> np.ndarray:
        """Convert game state to feature vector for neural network."""
        return self.game.state_to_features(state)
    
    def _get_legal_actions_mask(self, state) -> np.ndarray:
        """Get binary mask for legal actions."""
        legal_actions = self.game.legal_actions(state)
        mask = np.zeros(self.game.num_actions())
        for action in legal_actions:
            mask[action] = 1
        return mask.astype(bool)
    
    def solve(self, num_iterations: int = None) -> Dict[str, np.ndarray]:
        """
        Run CFR for specified number of iterations.
        
        Args:
            num_iterations: Number of CFR iterations to run
            
        Returns:
            Dictionary mapping info states to average strategies
        """
        if num_iterations is None:
            num_iterations = self.cfr_iterations
        
        initial_state = self.game.initial_state()
        
        for i in range(num_iterations):
            self.iteration += 1
            
            # Run CFR for each player
            for player in range(self.game.num_players()):
                reach_probs = np.ones(self.game.num_players())
                self.cfr(initial_state, reach_probs, player)
        
        # Return average strategies
        strategies = {}
        for info_state, node in self.nodes.items():
            strategies[info_state] = node.get_average_strategy()
        
        return strategies
    
    def get_policy(self, state) -> Dict[Any, float]:
        """Get the current policy at a given state."""
        current_player = self.game.current_player(state)
        info_state = self.game.information_state_string(state, current_player)
        legal_actions = self.game.legal_actions(state)
        
        if info_state in self.nodes:
            node = self.nodes[info_state]
            strategy = node.get_average_strategy()
            return {action: strategy[i] for i, action in enumerate(legal_actions)}
        else:
            # Uniform random policy for unseen states
            prob = 1.0 / len(legal_actions)
            return {action: prob for action in legal_actions}
    
    def get_value(self, state, player: int) -> float:
        """Get the estimated value for a player at a given state."""
        info_state = self.game.information_state_string(state, player)
        
        if info_state in self.nodes:
            node = self.nodes[info_state]
            if node.network_values is not None:
                return node.network_values[player]
        
        # Fallback: estimate using Monte Carlo rollout
        return self._monte_carlo_value(state, player)
    
    def _monte_carlo_value(self, state, player: int, num_rollouts: int = 10) -> float:
        """Estimate value using Monte Carlo rollouts."""
        total_value = 0.0
        
        for _ in range(num_rollouts):
            current_state = copy.deepcopy(state)
            
            while not self.game.is_terminal(current_state):
                legal_actions = self.game.legal_actions(current_state)
                action = np.random.choice(legal_actions)
                current_state = self.game.apply_action(current_state, action)
            
            returns = self.game.returns(current_state)
            total_value += returns[player]
        
        return total_value / num_rollouts
    
    def compute_exploitability(self, num_samples: int = 1000) -> float:
        """Compute exploitability of current strategy."""
        total_exploitability = 0.0
        
        for _ in range(num_samples):
            initial_state = self.game.initial_state()
            for player in range(self.game.num_players()):
                best_response_value = self._compute_best_response_value(
                    initial_state, player
                )
                current_value = self._compute_strategy_value(
                    initial_state, player
                )
                total_exploitability += max(0, best_response_value - current_value)
        
        return total_exploitability / (num_samples * self.game.num_players())
    
    def _compute_best_response_value(self, state, player: int) -> float:
        """Compute best response value for a player."""
        if self.game.is_terminal(state):
            return self.game.returns(state)[player]
        
        current_player = self.game.current_player(state)
        legal_actions = self.game.legal_actions(state)
        
        if current_player == player:
            # Maximize over actions
            best_value = float('-inf')
            for action in legal_actions:
                new_state = self.game.apply_action(state, action)
                value = self._compute_best_response_value(new_state, player)
                best_value = max(best_value, value)
            return best_value
        else:
            # Use current strategy of opponent
            policy = self.get_policy(state)
            expected_value = 0.0
            for action in legal_actions:
                new_state = self.game.apply_action(state, action)
                value = self._compute_best_response_value(new_state, player)
                expected_value += policy[action] * value
            return expected_value
    
    def _compute_strategy_value(self, state, player: int) -> float:
        """Compute value of current strategy for a player."""
        if self.game.is_terminal(state):
            return self.game.returns(state)[player]
        
        policy = self.get_policy(state)
        legal_actions = self.game.legal_actions(state)
        
        expected_value = 0.0
        for action in legal_actions:
            new_state = self.game.apply_action(state, action)
            value = self._compute_strategy_value(new_state, player)
            expected_value += policy[action] * value
        
        return expected_value
    
    def update_network(self, network):
        """Update the neural network."""
        self.network = network
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing."""
        return {
            "nodes": {info_state: {
                "regret_sum": node.regret_sum.tolist(),
                "strategy_sum": node.strategy_sum.tolist(),
                "legal_actions": node.legal_actions
            } for info_state, node in self.nodes.items()},
            "iteration": self.iteration
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.iteration = state["iteration"]
        self.nodes = {}
        
        for info_state, node_data in state["nodes"].items():
            legal_actions = node_data["legal_actions"]
            node = CFRNode(info_state, legal_actions)
            node.regret_sum = np.array(node_data["regret_sum"])
            node.strategy_sum = np.array(node_data["strategy_sum"])
            node.update_strategy()
            self.nodes[info_state] = node