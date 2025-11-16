"""
Prepare Lichess training data by downloading and filtering games.

This script downloads Lichess databases, filters for high-quality games,
and saves the processed data for training.

Usage:
    python training/prepare_lichess_data.py --num-games 1000 --output data/lichess_1000.npz
"""

import argparse
import requests
import zstandard as zstd
import io
import chess
import chess.pgn
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def board_to_tensor(board):
    """Convert chess board to 8x8x12 tensor representation.
    
    Channels: 6 piece types (pawn, rook, knight, bishop, queen, king) x 2 colors (white, black)
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
            rank = 7 - (square // 8)  # Convert to 0-7 rank (a8=0, a1=7)
            file = square % 8
            tensor[rank, file, channel] = 1.0
    
    return tensor


def parse_time_control(time_control_str):
    """Parse TimeControl header to extract initial time in seconds.
    
    Format: "600+0" means 600 seconds + 0 increment
    Returns: initial_time in seconds, or 0 if parsing fails
    """
    if not time_control_str:
        return 0
    
    try:
        if "+" in time_control_str:
            parts = time_control_str.split("+")
            initial_time = int(parts[0]) if parts[0].isdigit() else 0
        elif time_control_str.isdigit():
            initial_time = int(time_control_str)
        else:
            return 0
        return initial_time
    except (ValueError, AttributeError):
        return 0


def is_valid_game_type(time_control_str, game_types):
    """Check if game matches desired time control types.
    
    Args:
        time_control_str: TimeControl header value
        game_types: List of game types to accept (e.g., ["blitz", "rapid"])
    
    Returns:
        True if game matches one of the types
    """
    initial_time = parse_time_control(time_control_str)
    
    if "blitz" in game_types:
        if 180 <= initial_time <= 600:
            return True
    
    if "rapid" in game_types:
        if 600 < initial_time <= 1800:
            return True
    
    return False


def parse_rating(rating_str):
    """Parse rating from header value.
    
    Returns:
        Rating as int, or None if invalid/missing
    """
    if not rating_str:
        return None
    
    rating_str = str(rating_str).strip()
    if not rating_str or rating_str == "?":
        return None
    
    try:
        return int(rating_str)
    except ValueError:
        return None


def download_and_filter_lichess_games(
    num_games=1000,
    min_rating=1600,
    game_types=["blitz", "rapid"],
    skip_first_moves=10,
    skip_last_moves=5,
    sample_every_nth=2,
    verbose=True
):
    """
    Download and filter Lichess games from August 2025 database.
    
    Args:
        num_games: Target number of games to collect
        min_rating: Minimum ELO rating for both players
        game_types: List of game types to accept
        skip_first_moves: Number of opening moves to skip
        skip_last_moves: Number of endgame moves to skip
        sample_every_nth: Sample every Nth move
        verbose: Print progress information
    
    Returns:
        board_states: List of numpy arrays (8x8x12)
        moves: List of move UCI strings
        outcomes: List of outcomes (1.0 or -1.0)
        stats: Dictionary with statistics
    """
    all_board_states = []
    all_moves = []
    all_outcomes = []
    
    games_processed = 0
    games_filtered = 0
    games_checked = 0
    
    filter_reasons = {
        'no_rating': 0,
        'rating_too_low': 0,
        'game_type': 0,
        'draw': 0,
        'too_short': 0,
        'malformed': 0,
        'unknown_result': 0
    }
    
    # August 2025 database URL
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-08.pgn.zst"
    target_mb = 5000  # Stop after downloading ~5000 MB compressed (increased for more data)
    
    if verbose:
        print(f"\n📥 Downloading Lichess standard rated database (August 2025)...")
        print(f"  Target: {num_games} games")
        print(f"  Min rating: {min_rating}+ (both players)")
        print(f"  Game types: {', '.join(game_types)}")
        print(f"  Filters: Rated only, no draws, min 20 moves")
        print(f"  Sampling: Skip first {skip_first_moves}, last {skip_last_moves}, every {sample_every_nth}nd move")
        print(f"  Max download: {target_mb} MB compressed")
    
    # Setup decompression
    dctx = zstd.ZstdDecompressor()
    
    # Download with retries
    max_retries = 3
    for retry in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code != 200:
                if retry < max_retries - 1:
                    if verbose:
                        print(f"  ⚠️ Download failed (status {response.status_code}), retrying... ({retry + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Failed to download database after {max_retries} attempts (status {response.status_code})")
            
            # Get content length if available
            content_length = response.headers.get('Content-Length')
            total_size = int(content_length) if content_length else None
            
            if verbose:
                print(f"  ✓ Connected to database")
                if total_size:
                    print(f"  📦 File size: {total_size / 1024 / 1024:.1f} MB (compressed)")
            
            # Stream and decompress
            stream_reader = dctx.stream_reader(response.raw)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            
            # Use chess.pgn.read_game() directly on the stream - it handles game boundaries correctly
            # We need to track bytes read for progress, but chess.pgn.read_game() handles parsing
            total_read = 0
            max_read = target_mb * 1024 * 1024
            last_progress_update = 0
            last_progress_time = time.time()
            
            # Create a wrapper to track bytes read
            class ProgressTracker:
                def __init__(self, stream, callback):
                    self.stream = stream
                    self.callback = callback
                    self.pos = 0
                
                def read(self, size=-1):
                    data = self.stream.read(size)
                    self.pos += len(data) if data else 0
                    if self.callback:
                        self.callback(self.pos)
                    return data
                
                def __getattr__(self, name):
                    return getattr(self.stream, name)
            
            def update_progress(pos):
                nonlocal total_read, last_progress_update, last_progress_time
                total_read = pos
                current_time = time.time()
                mb_since_update = (total_read - last_progress_update) / 1024 / 1024
                time_since_update = current_time - last_progress_time
                
                if verbose and (mb_since_update >= 5.0 or time_since_update >= 5.0):
                    compressed_mb = total_read / 1024 / 1024
                    filter_summary = ", ".join([f"{k}:{v}" for k, v in filter_reasons.items() if v > 0])
                    
                    if total_size:
                        percent = (total_read / total_size) * 100
                        print(f"  📥 {compressed_mb:.1f} MB ({percent:.1f}%) | "
                              f"Games: {games_processed}/{num_games} | "
                              f"Filtered: {games_filtered} ({filter_summary}) | "
                              f"Positions: {len(all_board_states):,}")
                    else:
                        print(f"  📥 {compressed_mb:.1f} MB | "
                              f"Games: {games_processed}/{num_games} | "
                              f"Filtered: {games_filtered} ({filter_summary}) | "
                              f"Positions: {len(all_board_states):,}")
                    
                    last_progress_update = total_read
                    last_progress_time = current_time
            
            tracked_stream = ProgressTracker(text_stream, update_progress)
            
            # Parse games directly from stream - chess.pgn.read_game() handles boundaries
            while games_processed < num_games and total_read < max_read:
                try:
                    game = chess.pgn.read_game(tracked_stream)
                    
                    if game is None:
                        # End of stream or no more games
                        break
                    
                    games_checked += 1
                    show_debug = False  # Disabled detailed game output - only show progress updates
                    
                    # Check if we've exceeded download limit
                    if total_read >= max_read:
                        if verbose:
                            print(f"\n  ✓ Reached download limit: {target_mb} MB")
                        break
                    
                    # Process game
                    try:
                        # Extract headers
                        white_elo_str = game.headers.get("WhiteElo", "")
                        black_elo_str = game.headers.get("BlackElo", "")
                        time_control = game.headers.get("TimeControl", "")
                        result = game.headers.get("Result", "*")
                        
                        if show_debug:
                            print(f"\n  🔍 Game #{games_checked}:")
                            print(f"     Headers: {list(game.headers.keys())[:10]}...")
                            print(f"     WhiteElo: '{white_elo_str}'")
                            print(f"     BlackElo: '{black_elo_str}'")
                            print(f"     TimeControl: '{time_control}'")
                            print(f"     Result: '{result}'")
                        
                        # Check ratings exist and are valid
                        white_rating = parse_rating(white_elo_str)
                        black_rating = parse_rating(black_elo_str)
                        
                        if white_rating is None or black_rating is None:
                            if show_debug:
                                print(f"     ❌ Filtered: missing rating (white={white_rating}, black={black_rating})")
                            games_filtered += 1
                            filter_reasons['no_rating'] += 1
                            continue
                        
                        if show_debug:
                            print(f"     Parsed ratings: white={white_rating}, black={black_rating}")
                        
                        # Check minimum rating
                        if white_rating < min_rating or black_rating < min_rating:
                            if show_debug:
                                print(f"     ❌ Filtered: rating too low (min={min_rating})")
                            games_filtered += 1
                            filter_reasons['rating_too_low'] += 1
                            continue
                        
                        if show_debug:
                            print(f"     ✓ Ratings OK")
                        
                        # Check game type
                        if not is_valid_game_type(time_control, game_types):
                            if show_debug:
                                print(f"     ❌ Filtered: game type (TimeControl='{time_control}')")
                            games_filtered += 1
                            filter_reasons['game_type'] += 1
                            continue
                        
                        if show_debug:
                            print(f"     ✓ Game type OK")
                        
                        # Check result (no draws, no unfinished)
                        if result == "1/2-1/2":
                            if show_debug:
                                print(f"     ❌ Filtered: draw")
                            games_filtered += 1
                            filter_reasons['draw'] += 1
                            continue
                        
                        if result not in ["1-0", "0-1"]:
                            if show_debug:
                                print(f"     ❌ Filtered: unknown result (result='{result}')")
                            games_filtered += 1
                            filter_reasons['unknown_result'] += 1
                            continue
                        
                        # Determine outcome
                        outcome = 1.0 if result == "1-0" else -1.0
                        
                        if show_debug:
                            print(f"     ✓ Result OK: {result}")
                        
                        # Extract moves
                        board = game.board()
                        game_moves = list(game.mainline_moves())
                        
                        if show_debug:
                            print(f"     Moves: {len(game_moves)}")
                        
                        # Check minimum moves (at least 20 moves)
                        min_required_moves = max(20, skip_first_moves + skip_last_moves + 1)
                        if len(game_moves) < min_required_moves:
                            if show_debug:
                                print(f"     ❌ Filtered: too short ({len(game_moves)} < {min_required_moves})")
                            games_filtered += 1
                            filter_reasons['too_short'] += 1
                            continue
                        
                        # Extract positions with sampling
                        positions_extracted = 0
                        for move_idx, move in enumerate(game_moves):
                            # Skip opening moves
                            if move_idx < skip_first_moves:
                                board.push(move)
                                continue
                            
                            # Skip endgame moves
                            if move_idx >= len(game_moves) - skip_last_moves:
                                break
                            
                            # Sample every Nth move
                            if (move_idx - skip_first_moves) % sample_every_nth == 0:
                                all_board_states.append(board_to_tensor(board))
                                all_moves.append(move.uci())
                                all_outcomes.append(outcome)
                                positions_extracted += 1
                            
                            board.push(move)
                        
                        if positions_extracted > 0:
                            games_processed += 1
                            if show_debug:
                                print(f"     ✅ ACCEPTED! Positions extracted: {positions_extracted}")
                        else:
                            if show_debug:
                                print(f"     ❌ Filtered: no positions extracted")
                            games_filtered += 1
                            filter_reasons['too_short'] += 1
                    
                    except Exception as e:
                        games_filtered += 1
                        filter_reasons['malformed'] += 1
                        if verbose and games_checked <= 5:
                            print(f"     ⚠️ Error parsing game: {e}")
                        continue
                    
                    # Check if we've reached limits after processing game
                    if games_processed >= num_games:
                        break
                    if total_read >= max_read:
                        if verbose:
                            print(f"\n  ✓ Reached download limit: {target_mb} MB")
                        break
                
                except Exception as e:
                    # Error reading game from stream
                    games_filtered += 1
                    filter_reasons['malformed'] += 1
                    if verbose and games_checked <= 5:
                        print(f"     ⚠️ Error reading game: {e}")
                    continue
            
            # Successfully processed
            break
            
        except requests.exceptions.RequestException as e:
            if retry < max_retries - 1:
                if verbose:
                    print(f"  ⚠️ Download error: {e}, retrying... ({retry + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Failed to download database after {max_retries} attempts: {e}")
        except Exception as e:
            if retry < max_retries - 1:
                if verbose:
                    print(f"  ⚠️ Error: {e}, retrying... ({retry + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Failed to process database after {max_retries} attempts: {e}")
    
    if len(all_board_states) == 0:
        raise RuntimeError("No games found matching criteria!")
    
    # Convert to numpy arrays
    board_states = np.array(all_board_states)
    outcomes = np.array(all_outcomes)
    
    stats = {
        'games_processed': games_processed,
        'games_filtered': games_filtered,
        'games_checked': games_checked,
        'total_positions': len(board_states),
        'wins': int(np.sum(outcomes > 0)),
        'losses': int(np.sum(outcomes < 0)),
        'filter_reasons': filter_reasons,
        'avg_positions_per_game': len(board_states) / games_processed if games_processed > 0 else 0
    }
    
    if verbose:
        print(f"\n✓ Successfully processed {games_processed} games")
        print(f"  Checked: {games_checked} games")
        print(f"  Filtered out: {games_filtered} games")
        print(f"  Filter breakdown:")
        for reason, count in filter_reasons.items():
            if count > 0:
                print(f"    - {reason}: {count}")
        print(f"  Total positions: {len(board_states):,}")
        print(f"  Wins: {stats['wins']:,}")
        print(f"  Losses: {stats['losses']:,}")
        print(f"  Average positions per game: {stats['avg_positions_per_game']:.1f}")
    
    return board_states, all_moves, outcomes, stats


def save_dataset(board_states, moves, outcomes, output_path, stats):
    """Save dataset to compressed .npz file with metadata.
    
    Args:
        board_states: numpy array of shape (N, 8, 8, 12)
        moves: List of move UCI strings
        outcomes: numpy array of 1.0 or -1.0
        output_path: Path to save .npz file
        stats: Dictionary with collection statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert moves to indices for storage using unified encoding
    # Format: from_square * 64 + to_square (0-4095)
    move_indices = []
    for move_str in moves:
        move = chess.Move.from_uci(move_str)
        # Unified encoding: from_square * 64 + to_square
        target_idx = move.from_square * 64 + move.to_square
        move_indices.append(target_idx)

    move_indices = np.array(move_indices, dtype=np.int32)
    
    np.savez_compressed(
        output_path,
        board_states=board_states,
        move_indices=move_indices,
        outcomes=outcomes,
        stats=stats
    )
    
    print(f"\n✓ Saved dataset to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def verify_dataset(file_path):
    """Verify that a dataset file contains valid data.
    
    Args:
        file_path: Path to .npz file
    
    Returns:
        True if valid, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"\n🔍 Verifying dataset: {file_path}")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Check required keys
        required_keys = ['board_states', 'move_indices', 'outcomes']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        board_states = data['board_states']
        move_indices = data['move_indices']
        outcomes = data['outcomes']
        stats = data.get('stats', {})
        
        # Verify shapes
        num_samples = len(board_states)
        assert len(move_indices) == num_samples, "Mismatch: board_states and move_indices"
        assert len(outcomes) == num_samples, "Mismatch: board_states and outcomes"
        assert board_states.shape[1:] == (8, 8, 12), f"Invalid board shape: {board_states.shape}"
        
        # Verify data ranges
        assert np.all((outcomes == 1.0) | (outcomes == -1.0)), "Invalid outcomes (should be 1.0 or -1.0)"
        assert np.all((board_states >= 0) & (board_states <= 1)), "Invalid board values (should be 0-1)"
        
        print(f"  ✓ Dataset valid!")
        print(f"  ✓ Samples: {num_samples:,}")
        print(f"  ✓ Board shape: {board_states.shape}")
        print(f"  ✓ Wins: {np.sum(outcomes > 0):,}")
        print(f"  ✓ Losses: {np.sum(outcomes < 0):,}")
        
        if isinstance(stats, dict):
            print(f"  ✓ Games processed: {stats.get('games_processed', 'N/A')}")
            print(f"  ✓ Avg positions/game: {stats.get('avg_positions_per_game', 'N/A'):.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Lichess training data by downloading and filtering games',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--num-games', type=int, default=10000,
        help='Number of games to process (default: 10000)'
    )
    parser.add_argument(
        '--output', type=str, default='data/lichess_data.npz',
        help='Output file path (default: data/lichess_data.npz)'
    )
    parser.add_argument(
        '--min-rating', type=int, default=1600,
        help='Minimum player rating (default: 1600)'
    )
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify existing dataset, do not download'
    )
    
    args = parser.parse_args()

    # DEBUG: Print parsed arguments
    print(f"DEBUG: num_games={args.num_games}, min_rating={args.min_rating}, output={args.output}")

    if args.verify_only:
        verify_dataset(args.output)
        return
    
    print("=" * 70)
    print("Lichess Data Preparation")
    print("=" * 70)
    
    try:
        # Download and filter games
        board_states, moves, outcomes, stats = download_and_filter_lichess_games(
            num_games=args.num_games,
            min_rating=args.min_rating,
            game_types=["blitz", "rapid"],
            skip_first_moves=10,
            skip_last_moves=5,
            sample_every_nth=2,
            verbose=True
        )
        
        # Verify we got data
        if len(board_states) == 0:
            raise RuntimeError("No data extracted! Check filtering logic.")
        
        # Save dataset
        save_dataset(board_states, moves, outcomes, args.output, stats)
        
        # Verify saved dataset
        print("\n" + "=" * 70)
        verify_dataset(args.output)
        print("=" * 70)
        
        print("\n✅ Data preparation complete!")
        print(f"\nNext steps:")
        print(f"  1. Verify the dataset looks good")
        print(f"  2. Train with: modal run training/modal_train.py --data {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

