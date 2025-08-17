"""
Sound Self-Play implementation for Student of Games.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import copy
import random
from .core import PublicBeliefState


class SoundSelfPlay:
    """
    Sound Self-Play algorithm that generates training data by playing
    the game using the current strategy and collecting experiences.
    
    This implements the "sound" property ensuring the algorithm converges
    to Nash equilibrium strategies.
    """
    
    def __init__(
        self,
        game,
        cfr,
        buffer_size: int = 10000,
        min_buffer_size: int = 1000,
        exploration_epsilon: float = 0.1,
        use_importance_sampling: bool = True
    ):
        self.game = game
        self.cfr = cfr
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.exploration_epsilon = exploration_epsilon
        self.use_importance_sampling = use_importance_sampling
        
        # Experience replay buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        
        # Metrics
        self.episodes_generated = 0
        self.total_rewards = []
    
    def generate_episode(self) -> List[Dict[str, Any]]:
        """
        Generate a single episode of self-play and return training data.
        
        Returns:
            List of training samples containing state features, policy targets,
            value targets, and other relevant information.
        """
        trajectory = []
        state = self.game.initial_state()
        
        # Track reach probabilities for importance sampling
        reach_probabilities = np.ones(self.game.num_players())
        
        while not self.game.is_terminal(state):
            current_player = self.game.current_player(state)
            legal_actions = self.game.legal_actions(state)
            
            # Get current policy from CFR
            policy = self.cfr.get_policy(state)
            policy_probs = np.array([policy.get(action, 0.0) for action in legal_actions])
            
            # Add exploration noise
            if random.random() < self.exploration_epsilon:
                policy_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
            # Sample action
            action_idx = np.random.choice(len(legal_actions), p=policy_probs)
            action = legal_actions[action_idx]
            
            # Store transition information
            transition = {
                'state': copy.deepcopy(state),
                'player': current_player,
                'legal_actions': legal_actions.copy(),
                'policy': policy_probs.copy(),
                'action': action,
                'action_idx': action_idx,
                'reach_prob': reach_probabilities[current_player]
            }
            trajectory.append(transition)
            
            # Update reach probabilities
            reach_probabilities[current_player] *= policy_probs[action_idx]
            
            # Apply action
            state = self.game.apply_action(state, action)
        
        # Get final returns
        final_returns = self.game.returns(state)
        
        # Convert trajectory to training data
        training_data = self._process_trajectory(trajectory, final_returns)
        
        # Add to experience buffer
        self._add_to_buffer(training_data)
        
        self.episodes_generated += 1
        self.total_rewards.append(final_returns)
        
        return training_data
    
    def _process_trajectory(
        self, 
        trajectory: List[Dict[str, Any]], 
        final_returns: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Process a trajectory into training data.
        
        Args:
            trajectory: List of transitions from the episode
            final_returns: Final returns for each player
            
        Returns:
            List of training samples
        """
        training_data = []
        
        for i, transition in enumerate(trajectory):
            state = transition['state']
            player = transition['player']
            legal_actions = transition['legal_actions']
            policy = transition['policy']
            
            # Convert state to features
            state_features = self._state_to_features(state)
            
            # Create legal actions mask
            legal_mask = np.zeros(self.game.num_actions())
            for action in legal_actions:
                legal_mask[action] = 1.0
            
            # Compute value targets using Monte Carlo returns
            value_targets = self._compute_value_targets(
                trajectory[i:], final_returns, player
            )
            
            # Create policy target (current CFR strategy)
            policy_target = np.zeros(self.game.num_actions())
            for j, action in enumerate(legal_actions):
                policy_target[action] = policy[j]
            
            # Normalize policy target
            if np.sum(policy_target) > 0:
                policy_target = policy_target / np.sum(policy_target)
            
            sample = {
                'state_features': state_features,
                'policy_target': policy_target,
                'value_targets': value_targets,
                'legal_actions_mask': legal_mask.astype(bool),
                'player': player,
                'importance_weight': 1.0 / max(transition['reach_prob'], 1e-10)
            }
            
            training_data.append(sample)
        
        return training_data
    
    def _state_to_features(self, state) -> np.ndarray:
        """Convert game state to feature vector."""
        return self.game.state_to_features(state)
    
    def _compute_value_targets(
        self, 
        remaining_trajectory: List[Dict[str, Any]], 
        final_returns: np.ndarray,
        player: int
    ) -> np.ndarray:
        """
        Compute value targets for training.
        
        For now, we use Monte Carlo returns (final game outcome).
        More sophisticated methods could use TD-learning or GAE.
        """
        # Simple Monte Carlo return
        value_targets = final_returns.copy()
        
        # Could add temporal difference learning here:
        # value_targets[player] = reward + gamma * next_value
        
        return value_targets
    
    def _add_to_buffer(self, training_data: List[Dict[str, Any]]):
        """Add training data to experience replay buffer."""
        for sample in training_data:
            if len(self.experience_buffer) >= self.buffer_size:
                # Remove oldest sample
                self.experience_buffer.pop(0)
            self.experience_buffer.append(sample)
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of training data from the experience buffer.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            List of training samples
        """
        if len(self.experience_buffer) < self.min_buffer_size:
            return []
        
        # Sample with or without importance sampling
        if self.use_importance_sampling:
            return self._importance_sample(batch_size)
        else:
            return random.sample(
                self.experience_buffer, 
                min(batch_size, len(self.experience_buffer))
            )
    
    def _importance_sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch using importance sampling weights."""
        if not self.experience_buffer:
            return []
        
        # Extract importance weights
        weights = np.array([sample['importance_weight'] for sample in self.experience_buffer])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Sample indices
        indices = np.random.choice(
            len(self.experience_buffer),
            size=min(batch_size, len(self.experience_buffer)),
            replace=False,
            p=weights
        )
        
        return [self.experience_buffer[i] for i in indices]
    
    def generate_training_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of training data by running episodes or sampling from buffer.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            List of training samples
        """
        # If buffer is too small, generate new episodes
        if len(self.experience_buffer) < self.min_buffer_size:
            training_data = []
            while len(training_data) < batch_size:
                episode_data = self.generate_episode()
                training_data.extend(episode_data)
            return training_data[:batch_size]
        
        # Otherwise sample from buffer and optionally add new episodes
        batch = self.sample_batch(batch_size)
        
        # Occasionally add fresh data
        if random.random() < 0.1:  # 10% chance to add fresh episode
            fresh_data = self.generate_episode()
            # Replace some samples with fresh data
            replace_count = min(len(fresh_data), len(batch) // 4)
            if replace_count > 0:
                batch[-replace_count:] = fresh_data[:replace_count]
        
        return batch
    
    def evaluate_current_strategy(self, num_games: int = 100) -> Dict[str, float]:
        """
        Evaluate the current strategy by playing games.
        
        Args:
            num_games: Number of evaluation games to play
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_returns = np.zeros(self.game.num_players())
        game_lengths = []
        
        for _ in range(num_games):
            state = self.game.initial_state()
            moves = 0
            
            while not self.game.is_terminal(state):
                policy = self.cfr.get_policy(state)
                legal_actions = self.game.legal_actions(state)
                
                # Select action greedily (no exploration)
                best_action = max(policy.items(), key=lambda x: x[1])[0]
                state = self.game.apply_action(state, best_action)
                moves += 1
            
            returns = self.game.returns(state)
            total_returns += returns
            game_lengths.append(moves)
        
        return {
            'average_returns': (total_returns / num_games).tolist(),
            'average_game_length': np.mean(game_lengths),
            'win_rates': [(total_returns[i] > 0).sum() / num_games 
                         for i in range(self.game.num_players())]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about self-play training."""
        if not self.total_rewards:
            return {}
        
        recent_rewards = self.total_rewards[-100:] if len(self.total_rewards) > 100 else self.total_rewards
        
        return {
            'episodes_generated': self.episodes_generated,
            'buffer_size': len(self.experience_buffer),
            'recent_average_returns': np.mean(recent_rewards, axis=0).tolist(),
            'recent_return_std': np.std(recent_rewards, axis=0).tolist(),
            'total_episodes': len(self.total_rewards)
        }
    
    def reset_buffer(self):
        """Clear the experience buffer."""
        self.experience_buffer.clear()
    
    def save_buffer(self, filepath: str):
        """Save experience buffer to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.experience_buffer, f)
    
    def load_buffer(self, filepath: str):
        """Load experience buffer from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.experience_buffer = pickle.load(f)