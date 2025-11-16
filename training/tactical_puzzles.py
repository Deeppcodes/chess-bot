"""
Phase 3: Tactical puzzle data preparation.
Downloads and processes Lichess puzzles for tactical training.
"""

import requests
import chess
import numpy as np
from pathlib import Path
import csv
import io
import time


def download_lichess_puzzles(
    num_puzzles: int = 50000,
    min_rating: int = 1500,
    max_rating: int = 2500,
    output_path: str = 'data/tactical_puzzles.npz',
    verbose: bool = True
):
    """
    Download tactical puzzles from Lichess database.

    Args:
        num_puzzles: Number of puzzles to download
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
        output_path: Where to save the processed puzzles
        verbose: Print progress

    Returns:
        Dictionary with puzzle data
    """
    if verbose:
        print("=" * 70)
        print("Downloading Lichess Tactical Puzzles")
        print("=" * 70)
        print(f"Target puzzles: {num_puzzles:,}")
        print(f"Rating range: {min_rating}-{max_rating}")
        print()

    # Lichess puzzle database URL
    url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"

    if verbose:
        print("Downloading puzzle database...")
        print("(This may take a few minutes - database is ~200MB compressed)")

    start_time = time.time()

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        if verbose:
            print("✓ Download complete, decompressing...")

        # Decompress bz2
        import bz2
        decompressed = bz2.decompress(response.content)
        text = decompressed.decode('utf-8')

        if verbose:
            download_time = time.time() - start_time
            print(f"✓ Decompressed in {download_time:.1f}s")
            print("\nProcessing puzzles...")

    except Exception as e:
        print(f"❌ Failed to download puzzles: {e}")
        print("Falling back to sample puzzles...")
        return generate_sample_puzzles(num_puzzles, output_path, verbose)

    # Parse CSV
    reader = csv.DictReader(io.StringIO(text))

    all_positions = []
    all_moves = []
    all_themes = []

    count = 0
    skipped = 0

    for row in reader:
        if count >= num_puzzles:
            break

        try:
            # Parse puzzle data
            puzzle_id = row.get('PuzzleId', '')
            fen = row.get('FEN', '')
            moves = row.get('Moves', '').split()
            rating = int(row.get('Rating', 0))
            themes = row.get('Themes', '').split()

            # Filter by rating
            if rating < min_rating or rating > max_rating:
                skipped += 1
                continue

            # Parse FEN position
            board = chess.Board(fen)

            # First move is opponent's move (set up the puzzle)
            # Second move is the solution
            if len(moves) < 2:
                skipped += 1
                continue

            # Apply first move (opponent blunders or sets up tactic)
            setup_move = chess.Move.from_uci(moves[0])
            board.push(setup_move)

            # Second move is our solution
            solution_move = chess.Move.from_uci(moves[1])

            # Encode position
            position = board_to_tensor(board)

            all_positions.append(position)
            all_moves.append(solution_move.uci())
            all_themes.append(themes)

            count += 1

            if verbose and count % 5000 == 0:
                print(f"  Processed: {count:,}/{num_puzzles:,} (skipped: {skipped:,})")

        except Exception as e:
            skipped += 1
            continue

    if verbose:
        total_time = time.time() - start_time
        print(f"\n✓ Processed {count:,} puzzles in {total_time:.1f}s")
        print(f"  Skipped: {skipped:,}")
        print(f"  Success rate: {count / (count + skipped) * 100:.1f}%")

    # Convert to numpy arrays
    positions_array = np.array(all_positions, dtype=np.float32)

    # Convert moves to policy targets (unified encoding)
    move_indices = []
    for move_str in all_moves:
        move = chess.Move.from_uci(move_str)
        idx = move.from_square * 64 + move.to_square
        move_indices.append(idx)

    move_indices_array = np.array(move_indices, dtype=np.int32)

    # All puzzles are "winning" positions (value = 1.0 for solver)
    values_array = np.ones(len(all_positions), dtype=np.float32)

    # Save data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        positions=positions_array,
        move_indices=move_indices_array,
        values=values_array,
        themes=all_themes,
        num_puzzles=count,
        rating_range=(min_rating, max_rating)
    )

    if verbose:
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"\n✓ Saved to: {output_path}")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Positions: {len(positions_array):,}")
        print("=" * 70)

    return {
        'positions': positions_array,
        'move_indices': move_indices_array,
        'values': values_array,
        'themes': all_themes
    }


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to tensor representation.

    Returns:
        numpy array of shape (8, 8, 12)
    """
    from chess import PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING

    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    piece_to_channel = {
        (PAWN, True): 0, (ROOK, True): 1, (KNIGHT, True): 2,
        (BISHOP, True): 3, (QUEEN, True): 4, (KING, True): 5,
        (PAWN, False): 6, (ROOK, False): 7, (KNIGHT, False): 8,
        (BISHOP, False): 9, (QUEEN, False): 10, (KING, False): 11,
    }

    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank = 7 - (square // 8)
            file = square % 8
            tensor[rank, file, channel] = 1.0

    return tensor


def generate_sample_puzzles(num_puzzles: int, output_path: str, verbose: bool):
    """Generate sample tactical puzzles if download fails."""
    if verbose:
        print("\nGenerating sample tactical puzzles...")

    # Common tactical patterns
    sample_fens = [
        # Back rank mate
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        # Fork
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        # Pin
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        # Skewer
        "r3k2r/ppp2ppp/2n1b3/3qp3/3P4/2PB1N2/PP3PPP/R2Q1RK1 b kq - 0 10",
    ]

    positions = []
    moves = []
    for _ in range(num_puzzles):
        fen = np.random.choice(sample_fens)
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            positions.append(board_to_tensor(board))
            moves.append(move.from_square * 64 + move.to_square)

    positions_array = np.array(positions, dtype=np.float32)
    moves_array = np.array(moves, dtype=np.int32)
    values_array = np.ones(len(positions), dtype=np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        positions=positions_array,
        move_indices=moves_array,
        values=values_array,
        num_puzzles=num_puzzles,
        sample=True
    )

    if verbose:
        print(f"✓ Generated {num_puzzles} sample puzzles")

    return {
        'positions': positions_array,
        'move_indices': moves_array,
        'values': values_array
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Lichess tactical puzzles")
    parser.add_argument('--num-puzzles', type=int, default=50000,
                       help='Number of puzzles to download')
    parser.add_argument('--min-rating', type=int, default=1500,
                       help='Minimum puzzle rating')
    parser.add_argument('--max-rating', type=int, default=2500,
                       help='Maximum puzzle rating')
    parser.add_argument('--output', type=str, default='data/tactical_puzzles.npz',
                       help='Output path')

    args = parser.parse_args()

    download_lichess_puzzles(
        num_puzzles=args.num_puzzles,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        output_path=args.output,
        verbose=True
    )
