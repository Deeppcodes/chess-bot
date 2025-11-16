"""
Stockfish Benchmark Suite for Chess Bot Evaluation.

This script tests your chess bot against Stockfish at various ELO levels
to objectively measure its playing strength.

Requirements:
    - Stockfish chess engine installed
    - python-chess package (already in requirements.txt)

Usage:
    python benchmark_stockfish.py --games 10 --elo 1200
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import chess
import chess.engine
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.board_encoder import board_to_tensor_torch
from src.main import _model, _mcts, _load_model


class StockfishBenchmark:
    """Benchmark chess bot against Stockfish."""
    
    def __init__(self, stockfish_path=None, model_path=None):
        """
        Initialize benchmark.
        
        Args:
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            model_path: Path to your trained model (defaults to models/chess_model.pth)
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(project_root, 'models', 'chess_model.pth')
        self.stockfish_path = stockfish_path or self.find_stockfish()
        
        if not self.stockfish_path:
            print("❌ Stockfish not found! Please install it first.")
            print("\nInstallation instructions:")
            print("  macOS:   brew install stockfish")
            print("  Ubuntu:  sudo apt-get install stockfish")
            print("  Windows: Download from https://stockfishchess.org/download/")
            sys.exit(1)
        
        print(f"✓ Found Stockfish at: {self.stockfish_path}")
        
        # Load your model
        print(f"Loading your model from {model_path}...")
        _load_model()
        
        if _model is None:
            print("❌ Failed to load your model!")
            sys.exit(1)
        
        print("✓ Model loaded successfully")
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "stockfish_version": "Unknown",
            "games": []
        }
    
    @staticmethod
    def find_stockfish():
        """Try to find Stockfish in common locations."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(project_root, "benchmarks", "stockfish", "stockfish"),  # Local installation
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "stockfish",  # In PATH
            "C:\\Program Files\\Stockfish\\stockfish.exe",  # Windows
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                return path
        
        return None
    
    def play_game(self, stockfish_elo, bot_is_white, time_limit=1.0, mcts_sims=50):
        """
        Play one game against Stockfish.
        
        Args:
            stockfish_elo: Stockfish's ELO rating
            bot_is_white: Whether bot plays as white
            time_limit: Time limit per move for Stockfish (seconds)
            mcts_sims: Number of MCTS simulations for bot
            
        Returns:
            dict: Game result with details
        """
        board = chess.Board()
        moves_played = []
        
        # Start Stockfish engine
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            
            # Configure Stockfish to play at specific ELO
            # Note: Stockfish 16.1+ has minimum ELO of 1320
            if stockfish_elo < 1320:
                print(f"  ⚠️ Warning: Stockfish minimum ELO is 1320, adjusting from {stockfish_elo}")
                stockfish_elo = 1320
            
            engine.configure({"UCI_LimitStrength": True})
            engine.configure({"UCI_Elo": stockfish_elo})
            
            # Get Stockfish version
            if self.results["stockfish_version"] == "Unknown":
                self.results["stockfish_version"] = engine.id.get("name", "Stockfish")
        
        except Exception as e:
            print(f"❌ Failed to start Stockfish: {e}")
            return None
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        print(f"  {'White: Bot, Black: Stockfish' if bot_is_white else 'White: Stockfish, Black: Bot'}")
        
        while not board.is_game_over() and move_count < max_moves:
            current_player = "Bot" if (board.turn == bot_is_white) else "Stockfish"
            
            # Get move
            if (board.turn and bot_is_white) or (not board.turn and not bot_is_white):
                # Bot's turn
                try:
                    # Temporarily set MCTS simulations
                    old_sims = _mcts.num_simulations if _mcts else 50
                    if _mcts:
                        _mcts.num_simulations = mcts_sims
                    
                    move, _ = _mcts.search(board)
                    
                    if _mcts:
                        _mcts.num_simulations = old_sims
                except Exception as e:
                    print(f"  ❌ Bot error: {e}")
                    engine.quit()
                    return None
            else:
                # Stockfish's turn
                try:
                    result = engine.play(board, chess.engine.Limit(time=time_limit))
                    move = result.move
                except Exception as e:
                    print(f"  ❌ Stockfish error: {e}")
                    engine.quit()
                    return None
            
            # Make move
            moves_played.append(move.uci())
            board.push(move)
            move_count += 1
            
            # Print progress every 10 moves
            if move_count % 10 == 0:
                print(f"  Move {move_count}...")
        
        engine.quit()
        
        # Determine result
        result = board.result()
        
        if result == "1-0":
            outcome = "win" if bot_is_white else "loss"
        elif result == "0-1":
            outcome = "loss" if bot_is_white else "win"
        else:
            outcome = "draw"
        
        game_result = {
            "stockfish_elo": stockfish_elo,
            "bot_color": "white" if bot_is_white else "black",
            "outcome": outcome,
            "moves": move_count,
            "pgn_moves": moves_played,
            "termination": board.outcome().termination.name if board.outcome() else "MAX_MOVES"
        }
        
        return game_result
    
    def run_benchmark(self, stockfish_elo=1500, num_games=10, time_limit=1.0, mcts_sims=50):
        """
        Run benchmark against Stockfish at specific ELO.
        
        Args:
            stockfish_elo: Stockfish's ELO rating
            num_games: Number of games to play
            time_limit: Time per move for Stockfish
            mcts_sims: MCTS simulations for bot
            
        Returns:
            dict: Summary statistics
        """
        print("\n" + "=" * 70)
        print(f"Benchmark: {num_games} games vs Stockfish {stockfish_elo} ELO")
        print("=" * 70)
        
        wins = losses = draws = 0
        
        for i in range(num_games):
            bot_is_white = (i % 2 == 0)  # Alternate colors
            
            print(f"\nGame {i+1}/{num_games}:")
            
            start_time = time.time()
            game_result = self.play_game(stockfish_elo, bot_is_white, time_limit, mcts_sims)
            elapsed = time.time() - start_time
            
            if game_result is None:
                print(f"  ❌ Game aborted")
                continue
            
            game_result["game_duration"] = elapsed
            self.results["games"].append(game_result)
            
            # Update stats
            if game_result["outcome"] == "win":
                wins += 1
                print(f"  ✓ Bot wins! ({elapsed:.1f}s)")
            elif game_result["outcome"] == "loss":
                losses += 1
                print(f"  ✗ Bot loses ({elapsed:.1f}s)")
            else:
                draws += 1
                print(f"  = Draw ({elapsed:.1f}s)")
        
        # Calculate statistics
        total_games = wins + losses + draws
        if total_games == 0:
            print("\n❌ No games completed!")
            return None
        
        win_rate = wins / total_games * 100
        score = (wins + 0.5 * draws) / total_games * 100
        
        stats = {
            "stockfish_elo": stockfish_elo,
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "score": score,
            "mcts_simulations": mcts_sims
        }
        
        self.results["summary"] = stats
        
        # Print summary
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        print(f"Games Played: {total_games}")
        print(f"Wins:   {wins} ({win_rate:.1f}%)")
        print(f"Losses: {losses} ({losses/total_games*100:.1f}%)")
        print(f"Draws:  {draws} ({draws/total_games*100:.1f}%)")
        print(f"\nScore: {score:.1f}% (vs {stockfish_elo} ELO)")
        print("=" * 70)
        
        return stats
    
    def estimate_elo(self, stats):
        """
        Estimate bot's ELO based on performance against Stockfish.
        
        Args:
            stats: Statistics from benchmark
            
        Returns:
            int: Estimated ELO
        """
        stockfish_elo = stats["stockfish_elo"]
        score = stats["score"] / 100  # Convert to decimal
        
        # ELO calculation based on expected score
        # score = 1 / (1 + 10^((opponent_elo - player_elo)/400))
        # Solving for player_elo:
        if score >= 0.99:
            score = 0.99  # Avoid division by zero
        elif score <= 0.01:
            score = 0.01
        
        estimated_elo = stockfish_elo - 400 * math.log10((1 / score) - 1)
        
        return int(estimated_elo)
    
    def progressive_benchmark(self, games_per_level=5):
        """
        Run progressive benchmark starting from low ELO and going up.
        Stops when bot starts struggling (< 30% score).
        
        Args:
            games_per_level: Number of games per ELO level
        """
        # Note: Stockfish 16.1+ minimum ELO is 1320
        levels = [
            (1320, "Minimum Stockfish"),
            (1400, "Intermediate"),
            (1600, "Club Player"),
            (1800, "Advanced"),
            (2000, "Expert"),
            (2200, "Master"),
            (2400, "International Master"),
        ]
        
        all_results = []
        
        for elo, name in levels:
            print(f"\n{'='*70}")
            print(f"Testing against {name} ({elo} ELO)")
            print(f"{'='*70}")
            
            stats = self.run_benchmark(
                stockfish_elo=elo,
                num_games=games_per_level,
                time_limit=1.0,
                mcts_sims=200
            )
            
            if stats:
                all_results.append(stats)
                
                # Stop if performing poorly
                if stats["score"] < 30:
                    print(f"\n⚠️ Bot struggles at {elo} ELO level")
                    print(f"Estimated bot strength: ~{elo - 200} ELO")
                    break
        
        # Save all results
        self.save_results()
        
        return all_results
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(benchmarks_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, f"stockfish_benchmark_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
        return filename


import math

def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description='Benchmark chess bot against Stockfish')
    parser.add_argument('--elo', type=int, default=1200,
                       help='Stockfish ELO level (default: 1200)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games to play (default: 10)')
    parser.add_argument('--mcts-sims', type=int, default=200,
                       help='MCTS simulations per move (default: 200)')
    parser.add_argument('--progressive', action='store_true',
                       help='Run progressive benchmark (multiple ELO levels)')
    parser.add_argument('--stockfish-path', type=str, default=None,
                       help='Path to Stockfish binary (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Try multiple possible model paths
    possible_paths = [
        os.path.join(project_root, 'chess_model_best.pth'),
        os.path.join(project_root, 'models', 'chess_model.pth'),
        os.path.join(project_root, 'models', 'chess_model_best.pth'),
    ]
    default_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            default_model_path = path
            break
    if default_model_path is None:
        default_model_path = possible_paths[0]  # Fallback
    
    benchmark = StockfishBenchmark(
        stockfish_path=args.stockfish_path,
        model_path=default_model_path
    )
    
    if args.progressive:
        # Run progressive benchmark
        results = benchmark.progressive_benchmark(games_per_level=5)
    else:
        # Run single-level benchmark
        stats = benchmark.run_benchmark(
            stockfish_elo=args.elo,
            num_games=args.games,
            time_limit=1.0,
            mcts_sims=args.mcts_sims
        )
        
        if stats:
            # Estimate ELO
            estimated_elo = benchmark.estimate_elo(stats)
            print(f"\n📊 Estimated Bot ELO: ~{estimated_elo}")
            
            benchmark.save_results()


if __name__ == "__main__":
    main()

