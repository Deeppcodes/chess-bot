"""
Add tactical puzzle positions to training data.
Download from Lichess puzzle database and add to training.
"""

import requests
import chess
import chess.pgn
import numpy as np
from pathlib import Path


def download_lichess_puzzles(num_puzzles: int = 10000):
    """
    Download tactical puzzles from Lichess.
    These are positions where there's a clear best move (usually a tactic).
    """
    print(f"Downloading {num_puzzles} tactical puzzles from Lichess...")

    # Lichess puzzle database: https://database.lichess.org/#puzzles
    url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"

    # Download and extract
    print("Downloading puzzle database (this may take a while)...")
    response = requests.get(url, stream=True)

    import bz2
    import csv
    import io

    # Decompress
    decompressed = bz2.decompress(response.content)
    text = decompressed.decode('utf-8')

    # Parse CSV
    reader = csv.DictReader(io.StringIO(text))

    tactical_positions = []
    tactical_moves = []

    count = 0
    for row in reader:
        if count >= num_puzzles:
            break

        try:
            # Parse puzzle
            fen = row['FEN']
            moves = row['Moves'].split()
            rating = int(row['Rating'])

            # Only use puzzles rated 1500+ (decent tactics)
            if rating < 1500:
                continue

            # Set up position
            board = chess.Board(fen)

            # First move is the solution
            solution_move = chess.Move.from_uci(moves[0])

            # Store position and winning move
            from training.modal_train import board_to_tensor  # Import from your training code

            tactical_positions.append(board_to_tensor(board))
            tactical_moves.append(solution_move.uci())

            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} puzzles...")

        except Exception as e:
            continue

    print(f"\nExtracted {len(tactical_positions)} tactical puzzles")

    # Save to .npz file
    output_path = Path("data/tactical_puzzles.npz")
    output_path.parent.mkdir(exist_ok=True)

    np.savez_compressed(
        output_path,
        board_states=np.array(tactical_positions),
        moves=tactical_moves,
        outcomes=np.ones(len(tactical_positions))  # All puzzles are "winning" positions
    )

    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    download_lichess_puzzles(num_puzzles=10000)
