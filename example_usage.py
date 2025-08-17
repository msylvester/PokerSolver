#!/usr/bin/env python3
"""
Example usage of the Student of Games implementation.

This script shows how to use the Student of Games algorithm
for different scenarios including training, evaluation, and interactive play.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sog import StudentOfGames
from sog.games import KuhnPoker


def basic_training_example():
    """Basic example of training Student of Games on Kuhn Poker."""
    print("=== Basic Training Example ===")
    
    # Initialize game
    game = KuhnPoker()
    
    # Configure the algorithm
    network_config = {
        'input_size': 12,
        'hidden_size': 64,
        'lr': 1e-3
    }
    
    # Initialize Student of Games
    sog = StudentOfGames(
        game=game,
        network_config=network_config
    )
    
    # Train for a small number of iterations
    print("Training for 100 iterations...")
    metrics = sog.train(num_iterations=100)
    
    print(f"Final loss: {metrics['loss'][-1]:.4f}")
    print(f"Training completed!\n")
    
    return sog


def policy_analysis_example(sog):
    """Analyze the learned policy."""
    print("=== Policy Analysis ===")
    
    game = sog.game
    
    # Create a few example states and show the learned policy
    for trial in range(3):
        state = game.initial_state()
        print(f"\nTrial {trial + 1}:")
        print(f"Player 0 card: {state.player_cards[0]} (0=Jack, 1=Queen, 2=King)")
        print(f"Player 1 card: {state.player_cards[1]}")
        
        move_count = 0
        while not game.is_terminal(state) and move_count < 3:
            player = game.current_player(state)
            info_state = game.information_state_string(state, player)
            policy = sog.get_policy(state)
            
            print(f"  Player {player} turn:")
            print(f"    Information state: {info_state}")
            print(f"    Policy: Check/Call={policy.get(0, 0):.3f}, Bet/Fold={policy.get(1, 0):.3f}")
            
            # Take the most likely action
            best_action = max(policy.items(), key=lambda x: x[1])[0]
            action_name = "Check/Call" if best_action == 0 else "Bet/Fold"
            print(f"    Chosen action: {action_name}")
            
            state = game.apply_action(state, best_action)
            move_count += 1
        
        if game.is_terminal(state):
            returns = game.returns(state)
            winner = "Player 0" if returns[0] > returns[1] else "Player 1"
            print(f"  Game result: {winner} wins, returns: {returns}")


def interactive_play_example(sog):
    """Play interactively against the AI."""
    print("=== Interactive Play Example ===")
    print("You are Player 0. Actions: 0=Check/Call, 1=Bet/Fold")
    print("Type 'q' to quit\n")
    
    while True:
        try:
            # Start a new game
            game = sog.game
            state = game.initial_state()
            
            print(f"New game! Your card: {state.player_cards[0]} (0=Jack, 1=Queen, 2=King)")
            print(f"AI has card: {state.player_cards[1]} (hidden from you)")
            
            while not game.is_terminal(state):
                player = game.current_player(state)
                legal_actions = game.legal_actions(state)
                
                if player == 0:  # Human player
                    print(f"\nYour turn. History: {state.history}")
                    print("Legal actions: 0=Check/Call, 1=Bet/Fold")
                    
                    action_input = input("Enter your action (0 or 1, or 'q' to quit): ").strip()
                    
                    if action_input.lower() == 'q':
                        return
                    
                    try:
                        action = int(action_input)
                        if action not in legal_actions:
                            print("Invalid action! Try again.")
                            continue
                    except ValueError:
                        print("Please enter 0, 1, or 'q'")
                        continue
                        
                else:  # AI player
                    policy = sog.get_policy(state)
                    action = max(policy.items(), key=lambda x: x[1])[0]
                    action_name = "Check/Call" if action == 0 else "Bet/Fold"
                    print(f"\nAI's turn: {action_name}")
                
                state = game.apply_action(state, action)
            
            # Game over
            returns = game.returns(state)
            if returns[0] > returns[1]:
                print(f"\nYou win! Final returns: {returns}")
            elif returns[1] > returns[0]:
                print(f"\nAI wins! Final returns: {returns}")
            else:
                print(f"\nTie! Final returns: {returns}")
            
            play_again = input("\nPlay again? (y/n): ").strip().lower()
            if play_again != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def evaluation_example(sog):
    """Evaluate the AI against different strategies."""
    print("=== Evaluation Example ===")
    
    game = sog.game
    
    # Evaluate against random strategy
    print("Evaluating against random strategy...")
    wins = 0
    games = 1000
    
    for _ in range(games):
        state = game.initial_state()
        
        while not game.is_terminal(state):
            player = game.current_player(state)
            legal_actions = game.legal_actions(state)
            
            if player == 0:
                # Use AI strategy
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
    
    win_rate = wins / games
    print(f"Win rate against random: {win_rate:.3f}")
    
    # Evaluate against always-fold strategy
    print("\nEvaluating against always-fold strategy...")
    wins = 0
    
    for _ in range(games):
        state = game.initial_state()
        
        while not game.is_terminal(state):
            player = game.current_player(state)
            legal_actions = game.legal_actions(state)
            
            if player == 0:
                # Use AI strategy
                policy = sog.get_policy(state)
                action_probs = [policy.get(action, 0.0) for action in legal_actions]
                action = np.random.choice(legal_actions, p=action_probs)
            else:
                # Always fold (action 1 when possible, otherwise 0)
                action = 1 if 1 in legal_actions else 0
            
            state = game.apply_action(state, action)
        
        returns = game.returns(state)
        if returns[0] > returns[1]:
            wins += 1
    
    win_rate_vs_fold = wins / games
    print(f"Win rate against always-fold: {win_rate_vs_fold:.3f}")


def save_load_example():
    """Demonstrate saving and loading models."""
    print("=== Save/Load Example ===")
    
    # Train a small model
    game = KuhnPoker()
    network_config = {'input_size': 12, 'hidden_size': 32}
    sog = StudentOfGames(game=game, network_config=network_config)
    
    print("Training a small model...")
    sog.train(num_iterations=50)
    
    # Save the model
    model_path = "example_model.pt"
    sog.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")
    
    # Create a new instance and load the model
    sog_loaded = StudentOfGames(game=game, network_config=network_config)
    sog_loaded.load_checkpoint(model_path)
    print("Model loaded successfully!")
    
    # Verify they give the same policy
    test_state = game.initial_state()
    policy1 = sog.get_policy(test_state)
    policy2 = sog_loaded.get_policy(test_state)
    
    print(f"Original policy: {policy1}")
    print(f"Loaded policy: {policy2}")
    print(f"Policies match: {policy1 == policy2}")
    
    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")


def main():
    """Main function to run all examples."""
    print("Student of Games - Example Usage")
    print("=" * 40)
    
    # Basic training
    sog = basic_training_example()
    
    # Policy analysis
    policy_analysis_example(sog)
    
    # Evaluation
    evaluation_example(sog)
    
    # Save/load demonstration
    save_load_example()
    
    # Interactive play (optional)
    play_interactive = input("\nWould you like to play against the AI? (y/n): ").strip().lower()
    if play_interactive == 'y':
        interactive_play_example(sog)
    
    print("\nExample usage complete!")


if __name__ == "__main__":
    main()