"""
Counterfactual Value-Policy Network (CVPN) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import numpy as np


class CounterfactualValuePolicyNetwork(nn.Module):
    """
    Counterfactual Value-Policy Network (CVPN) that estimates both
    counterfactual values and policies for Student of Games.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 4,
        num_players: int = 2,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        
        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Value heads - one for each player
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(num_players)
        ])
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
    
    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state_features: Tensor of shape (batch_size, input_size)
            
        Returns:
            policy_logits: Tensor of shape (batch_size, num_actions)
            values: Tensor of shape (batch_size, num_players)
        """
        # Extract features
        features = self.feature_net(state_features)
        
        # Compute policy logits
        policy_logits = self.policy_head(features)
        
        # Compute values for each player
        values = torch.cat([
            value_head(features) for value_head in self.value_heads
        ], dim=1)
        
        return policy_logits, values
    
    def get_policy_and_values(
        self, 
        state_features: torch.Tensor,
        legal_actions_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy probabilities and values.
        
        Args:
            state_features: Input state features
            legal_actions_mask: Binary mask for legal actions
            
        Returns:
            policy_probs: Action probabilities
            values: Estimated values for each player
        """
        policy_logits, values = self.forward(state_features)
        
        # Apply legal actions mask if provided
        if legal_actions_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_actions_mask, float('-inf'))
        
        # Convert to probabilities
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        return policy_probs, values
    
    def train_step(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform one training step on a batch of data.
        
        Args:
            training_data: List of training samples, each containing:
                - 'state_features': State representation
                - 'policy_target': Target policy distribution
                - 'value_targets': Target values for each player
                - 'legal_actions_mask': Mask for legal actions
                
        Returns:
            Dictionary with loss information
        """
        if not training_data:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # Prepare batch data
        batch_size = len(training_data)
        state_features = torch.zeros(batch_size, self.input_size, device=self.device)
        policy_targets = torch.zeros(batch_size, self.num_actions, device=self.device)
        value_targets = torch.zeros(batch_size, self.num_players, device=self.device)
        legal_masks = torch.ones(batch_size, self.num_actions, device=self.device, dtype=torch.bool)
        
        for i, sample in enumerate(training_data):
            state_features[i] = torch.tensor(sample['state_features'], device=self.device)
            policy_targets[i] = torch.tensor(sample['policy_target'], device=self.device)
            value_targets[i] = torch.tensor(sample['value_targets'], device=self.device)
            if 'legal_actions_mask' in sample:
                legal_masks[i] = torch.tensor(sample['legal_actions_mask'], device=self.device)
        
        # Forward pass
        policy_logits, predicted_values = self.forward(state_features)
        
        # Apply legal actions mask
        policy_logits = policy_logits.masked_fill(~legal_masks, float('-inf'))
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Compute losses
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            policy_targets,
            reduction='batchmean'
        )
        
        value_loss = F.mse_loss(predicted_values, value_targets)
        
        # Total loss (weighted combination)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def predict(self, state_features: np.ndarray, legal_actions_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a single state.
        
        Args:
            state_features: State representation as numpy array
            legal_actions_mask: Binary mask for legal actions
            
        Returns:
            policy_probs: Action probabilities as numpy array
            values: Estimated values as numpy array
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state_features, device=self.device).unsqueeze(0)
            mask_tensor = None
            if legal_actions_mask is not None:
                mask_tensor = torch.tensor(legal_actions_mask, device=self.device).unsqueeze(0)
            
            policy_probs, values = self.get_policy_and_values(state_tensor, mask_tensor)
            
            return policy_probs.cpu().numpy()[0], values.cpu().numpy()[0]
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the network for checkpointing."""
        return {
            "state_dict": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_actions": self.num_actions,
                "num_players": self.num_players
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load network state from checkpoint."""
        self.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state"])