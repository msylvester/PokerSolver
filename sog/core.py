"""
Core classes for Student of Games implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from .cfr import GrowingTreeCFR
from .network import CounterfactualValuePolicyNetwork
from .selfplay import SoundSelfPlay


@dataclass
class PublicBeliefState:
    """
    Represents a public belief state β = (spub, r) where:
    - spub: public state sequence 
    - r: regret-to-go values
    """
    spub: List[Any]  # Public state sequence
    r: np.ndarray    # Regret-to-go values
    
    def __hash__(self):
        return hash((tuple(self.spub), tuple(self.r.flatten())))
    
    def __eq__(self, other):
        if not isinstance(other, PublicBeliefState):
            return False
        return (self.spub == other.spub and 
                np.allclose(self.r, other.r))


class StudentOfGames:
    """
    Main Student of Games algorithm implementation.
    
    This unified algorithm works for both perfect and imperfect information games
    by using Growing-Tree CFR with public belief states and sound self-play.
    """
    
    def __init__(
        self,
        game,
        network_config: Dict[str, Any],
        cfr_config: Dict[str, Any] = None,
        selfplay_config: Dict[str, Any] = None,
        device: str = "cpu"
    ):
        self.game = game
        self.device = device
        
        # Initialize neural network
        self.network = CounterfactualValuePolicyNetwork(
            input_size=network_config.get("input_size", 128),
            hidden_size=network_config.get("hidden_size", 256),
            num_actions=game.num_actions(),
            device=device
        )
        
        # Initialize CFR solver
        cfr_config = cfr_config or {}
        self.cfr = GrowingTreeCFR(
            game=game,
            network=self.network,
            **cfr_config
        )
        
        # Initialize self-play
        selfplay_config = selfplay_config or {}
        self.selfplay = SoundSelfPlay(
            game=game,
            cfr=self.cfr,
            **selfplay_config
        )
        
        self.training_history = []
    
    def train(
        self, 
        num_iterations: int,
        save_interval: int = 1000,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the Student of Games model.
        
        Args:
            num_iterations: Number of training iterations
            save_interval: How often to save checkpoints
            checkpoint_path: Path to save checkpoints
            
        Returns:
            Dictionary containing training metrics
        """
        metrics = {
            "loss": [],
            "policy_loss": [], 
            "value_loss": [],
            "exploitability": []
        }
        
        for iteration in range(num_iterations):
            # Generate training data through sound self-play
            training_data = self.selfplay.generate_episode()
            
            # Train the network on the generated data
            loss_info = self.network.train_step(training_data)
            
            # Update CFR with new network
            self.cfr.update_network(self.network)
            
            # Record metrics
            metrics["loss"].append(loss_info["total_loss"])
            metrics["policy_loss"].append(loss_info["policy_loss"])
            metrics["value_loss"].append(loss_info["value_loss"])
            
            # Evaluate exploitability periodically
            if iteration % 100 == 0:
                exploitability = self._compute_exploitability()
                metrics["exploitability"].append(exploitability)
                
                print(f"Iteration {iteration}: Loss={loss_info['total_loss']:.4f}, "
                      f"Exploitability={exploitability:.6f}")
            
            # Save checkpoint
            if checkpoint_path and iteration % save_interval == 0:
                self.save_checkpoint(f"{checkpoint_path}/iteration_{iteration}.pt")
        
        self.training_history.append(metrics)
        return metrics
    
    def get_policy(self, state) -> Dict[Any, float]:
        """Get the current policy at a given state."""
        return self.cfr.get_policy(state)
    
    def get_value(self, state, player: int) -> float:
        """Get the estimated value for a player at a given state."""
        return self.cfr.get_value(state, player)
    
    def _compute_exploitability(self) -> float:
        """Compute the exploitability of the current strategy."""
        return self.cfr.compute_exploitability()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "cfr_state": self.cfr.get_state(),
            "training_history": self.training_history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.cfr.load_state(checkpoint["cfr_state"])
        self.training_history = checkpoint.get("training_history", [])
    
    def play_against_human(self, human_player: int = 0):
        """Interactive play against human player."""
        state = self.game.initial_state()
        
        while not self.game.is_terminal(state):
            current_player = self.game.current_player(state)
            
            if current_player == human_player:
                # Human turn
                legal_actions = self.game.legal_actions(state)
                print(f"Legal actions: {legal_actions}")
                action = input("Enter your action: ")
                try:
                    action = type(legal_actions[0])(action)  # Convert to same type
                    if action not in legal_actions:
                        print("Invalid action!")
                        continue
                except:
                    print("Invalid action format!")
                    continue
            else:
                # AI turn
                policy = self.get_policy(state)
                action = max(policy.items(), key=lambda x: x[1])[0]
                print(f"AI plays: {action}")
            
            state = self.game.apply_action(state, action)
            print(f"Current state: {state}")
        
        # Game over
        returns = self.game.returns(state)
        winner = np.argmax(returns)
        if winner == human_player:
            print("You won!")
        elif returns[winner] > returns[human_player]:
            print("AI won!")
        else:
            print("It's a tie!")
        
        return returns