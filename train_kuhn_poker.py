#!/usr/bin/env python3
"""
Training script for Student of Games on Kuhn Poker.

This script demonstrates how to use the Student of Games algorithm
to learn to play Kuhn Poker from self-play.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sog import StudentOfGames
from sog.games import KuhnPoker


def plot_training_metrics(metrics, save_path=None):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(metrics['loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    
    # Policy and Value loss
    axes[0, 1].plot(metrics['policy_loss'], label='Policy Loss')
    axes[0, 1].plot(metrics['value_loss'], label='Value Loss')
    axes[0, 1].set_title('Policy and Value Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Exploitability
    if metrics['exploitability']:
        x_exploitability = range(0, len(metrics['loss']), 100)
        axes[1, 0].plot(x_exploitability, metrics['exploitability'])
        axes[1, 0].set_title('Exploitability')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Exploitability')
    
    # Recent loss (smoothed)
    if len(metrics['loss']) > 50:
        window_size = min(50, len(metrics['loss']) // 10)
        smoothed_loss = np.convolve(metrics['loss'], 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
        axes[1, 1].plot(smoothed_loss)
        axes[1, 1].set_title(f'Smoothed Loss (window={window_size})')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Smoothed Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()


def evaluate_against_random(sog, num_games=1000):
    """Evaluate the trained strategy against a random player."""
    game = sog.game
    wins = 0
    total_utility = 0
    
    for _ in range(num_games):
        state = game.initial_state()
        
        while not game.is_terminal(state):
            current_player = game.current_player(state)
            legal_actions = game.legal_actions(state)
            
            if current_player == 0:
                # Use trained strategy
                policy = sog.get_policy(state)
                action_probs = [policy.get(action, 0.0) for action in legal_actions]
                action = np.random.choice(legal_actions, p=action_probs)
            else:
                # Random strategy
                action = np.random.choice(legal_actions)
            
            state = game.apply_action(state, action)
        
        returns = game.returns(state)
        if returns[0] > returns[1]:
            wins += 1
        total_utility += returns[0]
    
    win_rate = wins / num_games
    avg_utility = total_utility / num_games
    
    return win_rate, avg_utility


def main():
    parser = argparse.ArgumentParser(description='Train Student of Games on Kuhn Poker')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for neural network')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden layer size for neural network')
    parser.add_argument('--cfr-iterations', type=int, default=50,
                       help='CFR iterations per training step')
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='How often to save checkpoints')
    parser.add_argument('--output-dir', type=str, default='./kuhn_poker_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Student of Games - Kuhn Poker Training ===")
    print(f"Training iterations: {args.iterations}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"CFR iterations: {args.cfr_iterations}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize game
    game = KuhnPoker()
    print(f"Game: {game}")
    print(f"Players: {game.num_players()}")
    print(f"Max actions: {game.num_actions()}")
    print()
    
    # Network configuration
    network_config = {
        'input_size': 12,  # Feature size for Kuhn Poker
        'hidden_size': args.hidden_size,
        'lr': args.lr
    }
    
    # CFR configuration
    cfr_config = {
        'cfr_iterations': args.cfr_iterations,
        'use_network': True,
        'network_threshold': 100
    }
    
    # Initialize Student of Games
    sog = StudentOfGames(
        game=game,
        network_config=network_config,
        cfr_config=cfr_config,
        device=args.device
    )
    
    print("Training started...")
    
    # Train the model
    metrics = sog.train(
        num_iterations=args.iterations,
        save_interval=args.save_interval,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints")
    )
    
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    sog.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training metrics
    plot_path = os.path.join(args.output_dir, "training_metrics.png")
    plot_training_metrics(metrics, plot_path)
    
    # Evaluate against random player
    print("\nEvaluating against random player...")
    win_rate, avg_utility = evaluate_against_random(sog, num_games=10000)
    print(f"Win rate against random: {win_rate:.3f}")
    print(f"Average utility against random: {avg_utility:.3f}")
    
    # Print some example policies
    print("\n=== Example Policies ===")
    test_state = game.initial_state()
    
    # Print policies for a few information states
    for i in range(min(3, 10)):  # Show a few examples
        if not game.is_terminal(test_state):
            player = game.current_player(test_state)
            info_state = game.information_state_string(test_state, player)
            policy = sog.get_policy(test_state)
            
            print(f"Info state: {info_state}")
            print(f"Policy: {policy}")
            print()
            
            # Take a random action to see next state
            legal_actions = game.legal_actions(test_state)
            if legal_actions:
                action = np.random.choice(legal_actions)
                test_state = game.apply_action(test_state, action)
        else:
            break
    
    # Save evaluation results
    results = {
        'win_rate_vs_random': win_rate,
        'avg_utility_vs_random': avg_utility,
        'final_loss': metrics['loss'][-1] if metrics['loss'] else 0,
        'final_exploitability': metrics['exploitability'][-1] if metrics['exploitability'] else 0,
        'training_iterations': args.iterations
    }
    
    import json
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()