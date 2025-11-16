"""
Inspect and verify the Lichess dataset.

Usage:
    python training/inspect_dataset.py data/lichess_data.npz
"""

import numpy as np
import chess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def tensor_to_board(tensor):
    """Convert 8x8x12 tensor back to chess board."""
    from chess import PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    
    board = chess.Board()
    board.clear()
    
    piece_from_channel = {
        0: (PAWN, True), 1: (ROOK, True), 2: (KNIGHT, True),
        3: (BISHOP, True), 4: (QUEEN, True), 5: (KING, True),
        6: (PAWN, False), 7: (ROOK, False), 8: (KNIGHT, False),
        9: (BISHOP, False), 10: (QUEEN, False), 11: (KING, False),
    }
    
    for rank in range(8):
        for file in range(8):
            for channel in range(12):
                if tensor[rank, file, channel] > 0.5:
                    piece_type, color = piece_from_channel[channel]
                    square = chess.square(file, 7 - rank)  # Convert back to chess square
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    break
    
    return board


def move_index_to_uci(move_idx):
    """Convert move index back to UCI (approximate)."""
    # This is a simplified conversion - the actual encoding is (from_square * 64 + to_square) % 4096
    # We can't perfectly reverse it, but we can show the index
    from_square = (move_idx // 64) % 64
    to_square = move_idx % 64
    try:
        move = chess.Move(from_square, to_square)
        return move.uci()
    except:
        return f"index_{move_idx}"


def inspect_dataset(file_path):
    """Inspect the dataset and show sample data."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Dataset file not found: {file_path}")
        return
    
    print("=" * 70)
    print(f"Inspecting dataset: {file_path}")
    print("=" * 70)
    
    # Load dataset
    data = np.load(file_path, allow_pickle=True)
    
    board_states = data['board_states']
    move_indices = data['move_indices']
    outcomes = data['outcomes']
    stats = data.get('stats', {})
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(board_states):,}")
    print(f"  Board shape: {board_states.shape}")
    print(f"  Wins (white): {np.sum(outcomes > 0):,}")
    print(f"  Losses (black): {np.sum(outcomes < 0):,}")
    print(f"  Win/Loss ratio: {np.sum(outcomes > 0) / np.sum(outcomes < 0):.3f}")
    
    if isinstance(stats, dict):
        print(f"\n📈 Collection Statistics:")
        print(f"  Games processed: {stats.get('games_processed', 'N/A')}")
        print(f"  Games filtered: {stats.get('games_filtered', 'N/A')}")
        print(f"  Avg positions/game: {stats.get('avg_positions_per_game', 'N/A'):.1f}")
        if 'filter_reasons' in stats:
            print(f"\n  Filter breakdown:")
            for reason, count in stats['filter_reasons'].items():
                if count > 0:
                    print(f"    - {reason}: {count}")
    
    # Show sample positions
    print(f"\n🔍 Sample Positions (first 5):")
    print("=" * 70)
    
    for i in range(min(5, len(board_states))):
        print(f"\nSample #{i+1}:")
        print(f"  Outcome: {'White wins' if outcomes[i] > 0 else 'Black wins'}")
        print(f"  Move index: {move_indices[i]}")
        
        # Convert tensor to board
        board = tensor_to_board(board_states[i])
        print(f"\n  Board position:")
        print(board)
        print(f"\n  FEN: {board.fen()}")
        
        # Show piece counts
        white_pieces = sum(1 for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).color)
        black_pieces = sum(1 for square in chess.SQUARES if board.piece_at(square) and not board.piece_at(square).color)
        print(f"  White pieces: {white_pieces}, Black pieces: {black_pieces}")
    
    # Show random samples
    print(f"\n🎲 Random Samples (5 random positions):")
    print("=" * 70)
    
    np.random.seed(42)
    random_indices = np.random.choice(len(board_states), size=min(5, len(board_states)), replace=False)
    
    for idx in random_indices:
        print(f"\nRandom Sample #{idx}:")
        print(f"  Outcome: {'White wins' if outcomes[idx] > 0 else 'Black wins'}")
        board = tensor_to_board(board_states[idx])
        print(f"  FEN: {board.fen()}")
    
    # Verify data integrity
    print(f"\n✅ Data Integrity Checks:")
    print("=" * 70)
    
    # Check 1: All outcomes are valid
    valid_outcomes = np.all((outcomes == 1.0) | (outcomes == -1.0))
    print(f"  ✓ Outcomes valid (1.0 or -1.0): {valid_outcomes}")
    
    # Check 2: Board values are in range
    valid_board_values = np.all((board_states >= 0) & (board_states <= 1))
    print(f"  ✓ Board values in range [0, 1]: {valid_board_values}")
    
    # Check 3: Each board has at least one piece (check if any square has a piece)
    # Shape is (N, 8, 8, 12) - sum across axes 1,2,3 to get total pieces per board
    pieces_per_board = np.sum(board_states, axis=(1, 2, 3))  # Sum all channels for all squares
    boards_with_pieces = np.all(pieces_per_board > 0)
    min_pieces = np.min(pieces_per_board)
    max_pieces = np.max(pieces_per_board)
    print(f"  ✓ All boards have pieces: {boards_with_pieces} (range: {min_pieces:.0f}-{max_pieces:.0f} pieces)")
    
    # Check 4: No boards are identical (check first 1000)
    sample_size = min(1000, len(board_states))
    unique_boards = len(np.unique(board_states[:sample_size].reshape(sample_size, -1), axis=0))
    print(f"  ✓ Unique positions in first {sample_size}: {unique_boards}/{sample_size}")
    
    # Check 5: Move indices are in valid range
    valid_move_indices = np.all((move_indices >= 0) & (move_indices < 4096))
    print(f"  ✓ Move indices in valid range [0, 4096): {valid_move_indices}")
    
    # Check 6: Board tensor sums (each square should have at most 1 piece across all channels)
    # Shape is (N, 8, 8, 12) - sum across axis 3 (channels) to get pieces per square
    pieces_per_square = np.sum(board_states, axis=3)  # Sum across all 12 channels for each square
    max_pieces_per_square = np.max(pieces_per_square)
    min_pieces_per_square = np.min(pieces_per_square)
    squares_with_multiple = np.sum(pieces_per_square > 1.0)
    print(f"  ✓ Pieces per square range: [{min_pieces_per_square:.1f}, {max_pieces_per_square:.1f}] (should be ≤ 1.0)")
    if squares_with_multiple > 0:
        print(f"    ⚠️ Warning: {squares_with_multiple} squares have > 1 piece (data encoding issue)")
    else:
        print(f"    ✓ All squares have ≤ 1 piece")
    
    print("\n" + "=" * 70)
    print("✅ Dataset inspection complete!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/inspect_dataset.py <dataset_file.npz>")
        print("Example: python training/inspect_dataset.py data/lichess_data.npz")
        sys.exit(1)
    
    inspect_dataset(sys.argv[1])

