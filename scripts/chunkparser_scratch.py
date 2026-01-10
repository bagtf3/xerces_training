import os
import numpy as np
from xerces_training.chunkparser import ChunkParser

from chessbot.utils import show_board
from pyfastchess import Board
import chess 
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# done
#training-run1--20250209-1017
#training-run1--20240819-1917
#training-run1--20230505-0917
#training-run1--20230415-1317
#training-run1--20230505-0217
chunk_dir = "C:/Users/Bryan/Data/chessbot_data/training_data/lc0/training-run1--20250209-1017"
all_files = [f for f in os.listdir(chunk_dir) if f.endswith(".gz")]
random.shuffle(all_files)
chunks = [os.path.join(chunk_dir, c) for c in all_files[:30]]
batch_size = 512
# Make a parser that does NOT spawn worker processes or the chunk_reader
parser = ChunkParser(
    chunks,
    shuffle_size=1,
    sample=1,
    batch_size=batch_size,
    workers=0,                # <= IMPORTANT: disables chunk_reader / workers
)


def ep_possible(cb):
    """
    Return True if an en-passant capture *could* be legal based on pawn
    geometry and empty landing square. Heuristic only; input_format==1
    has no EP byte, so this helps detect possible EP cases.
    """
    stm_white = cb.turn  # True if white to move

    if stm_white:
        opp_rank = 4    # opponent pawn must be on rank5 (0-indexed 4)
        land_rank = 5   # EP capture would land on rank6 (0-indexed 5)
        our_color = chess.WHITE
        their_color = chess.BLACK
    else:
        opp_rank = 3    # opponent pawn must be on rank4 (0-indexed 3)
        land_rank = 2   # EP capture would land on rank3 (0-indexed 2)
        our_color = chess.BLACK
        their_color = chess.WHITE

    for file_idx in range(8):
        opp_sq = chess.square(file_idx, opp_rank)
        opp_piece = cb.piece_at(opp_sq)
        if opp_piece is None or opp_piece.piece_type != chess.PAWN \
           or opp_piece.color != their_color:
            continue

        # check for our pawn adjacent on same rank and empty landing square
        for df in (-1, 1):
            adj = file_idx + df
            if adj < 0 or adj > 7:
                continue
            our_sq = chess.square(adj, opp_rank)
            our_piece = cb.piece_at(our_sq)
            if our_piece is None or our_piece.piece_type != chess.PAWN \
               or our_piece.color != our_color:
                continue

            land_sq = chess.square(file_idx, land_rank)
            if cb.piece_at(land_sq) is None:
                return True

    return False


def should_ignore_chess960(cb):
    """
    Return True when FEN reports castling rights but king/rook are not on
    standard chess start squares (indicating Chess960/FRC or non-standard).
    Parses castling rights from the FEN string rather than using built-ins.
    """
    fen = cb.fen()
    parts = fen.split()
    if len(parts) < 3:
        return False

    castling_field = parts[2]  # e.g. "KQkq" or "-"
    if castling_field == '-' or castling_field == '':
        return False

    white_k = 'K' in castling_field
    white_q = 'Q' in castling_field
    black_k = 'k' in castling_field
    black_q = 'q' in castling_field

    # white checks
    if white_k or white_q:
        wk_sq = cb.king(chess.WHITE)
        # king must be on e1 for standard chess
        if wk_sq != chess.E1:
            return True
        if white_k:
            p = cb.piece_at(chess.H1)
            if p is None or p.piece_type != chess.ROOK or p.color != chess.WHITE:
                return True
        if white_q:
            p = cb.piece_at(chess.A1)
            if p is None or p.piece_type != chess.ROOK or p.color != chess.WHITE:
                return True

    # black checks
    if black_k or black_q:
        bk_sq = cb.king(chess.BLACK)
        if bk_sq != chess.E8:
            return True
        if black_k:
            p = cb.piece_at(chess.H8)
            if p is None or p.piece_type != chess.ROOK or p.color != chess.BLACK:
                return True
        if black_q:
            p = cb.piece_at(chess.A8)
            if p is None or p.piece_type != chess.ROOK or p.color != chess.BLACK:
                return True

    return False



def _flip_index(idx):
    """Vertical flip across ranks: a1(0)->a8(56)."""
    file = idx % 8
    rank = idx // 8
    return (7 - rank) * 8 + file

def xerces_idx_to_uci(idx, stm):
    """
    Reverse-engineer Xerces/Board::moves_to_indices mapping.
    idx: integer index (0..4287)
    stm: True/1 if white to move, False/0 if black to move
    Returns UCI string like 'e2e4' or 'e7e8n' for underpromotions.
    """
    idx = int(idx)
    stm_white = bool(stm)

    if idx < 0 or idx > 4287:
        raise ValueError("index out of supported range 0..4287")

    # under-promotion block
    if idx >= 4096:
        unique = idx - 4096
        underpromo_type = unique // 64  # 0=N,1=B,2=R
        rest = unique % 64
        from_file = rest % 8
        to_file = rest // 8

        # build STM coords for promotion move (from rank 6 -> rank 7)
        from_stm = from_file + 6 * 8
        to_stm   = to_file   + 7 * 8

        # convert back to raw board coords if stm was black at encode time
        if not stm_white:
            from_raw = _flip_index(from_stm)
            to_raw   = _flip_index(to_stm)
        else:
            from_raw = from_stm
            to_raw = to_stm

        promo_char = {0: 'n', 1: 'b', 2: 'r'}.get(underpromo_type, 'n')
        uci = chess.square_name(from_raw) + chess.square_name(to_raw) + promo_char
        return uci

    # standard move block 0..4095
    from_slot = idx // 64
    to_slot = idx % 64

    # these slots are STM coords in the encoder; flip back if needed
    if not stm_white:
        from_raw = _flip_index(from_slot)
        to_raw = _flip_index(to_slot)
    else:
        from_raw = from_slot
        to_raw = to_slot

    return chess.square_name(from_raw) + chess.square_name(to_raw)

#from chessbot.model import load_model
#model_file_init = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_9x296_vs_stockfish/conv_9x296_vs_stockfish_model.h5"
#model = load_model(model_file_init)

bad_sum_flag = False
bad_illegal_mass_flag = False
possible_ep_catch = 0
chess_960_ignored = 0
passing = 0
for bi, batch in enumerate(parser.sequential()):    
    print("batch", bi)
    epoch = bi
    Xstack = batch[0]
    Mstack = batch[1]
    Pstack = batch[2]
    Ystack = batch[3]
    #fens = batch[4]
    #infos = batch[5]
    #%%
    for i, f in enumerate(fens):
        b = Board(f)
        cb = chess.Board(f, chess960=True)
        stm, _, _, _, _, invariance_info = infos[i]
        
        token_check = b.encode_64_tokens()
        if not all(Xstack[i] == token_check):
            print("FAILED token check!!!")
            print("planes to tokens:")
            print(Xstack[i])
            print("tokens from fen:")
            print(token_check)
            mismatch = []
            for q, (t1, t2) in enumerate(zip(Xstack[i], token_check)):
                if t1 != t2:
                    mismatch.append([q, t1, t2])
            print(mismatch)
            show_board(cb)
            raise Exception
        
        if should_ignore_chess960(cb):
            chess_960_ignored += 1
            continue
        
        lm = b.legal_moves()
        indices = b.moves_to_indices(lm)
        not_indices = [i for i in range(len(Pstack[i])) if i not in indices]
        
        extra = 0.0
        if np.sum(Pstack[i][not_indices]) > 0:
            bad_mass = Pstack[i][not_indices]
            if (len(bad_mass[bad_mass > 0]) < 2) and ep_possible(cb):
                possible_ep_catch += 1
                extra = bad_mass[bad_mass > 0].sum()
                
            else:
                bad_illegal_mass_flag = True
                bad_indices = []
                for j, p in enumerate(Pstack[i]):
                    if p > 0:
                        if j not in indices:
                            bad_indices.append((j, p))
                            
        policy_sum = Pstack[i][indices].sum()
        if not np.isclose(policy_sum + extra, 1.0, 0.25):
            bad_sum_flag = True

            
        if bad_sum_flag:
            print("policy do not sum to 1.0")
        if bad_illegal_mass_flag:
            print("mass on not_indices")
        
        if bad_sum_flag or bad_illegal_mass_flag:
            show_board(cb)
            raise Exception
        else:
            bad_sum_flag = False
            bad_illegal_mass_flag = False
            passing += 1
            
print("possible eps allowed to pass", possible_ep_catch)
print("chess 960s ignored", chess_960_ignored)
print(passing, "passing records")


leela_dir = "C:/Users/Bryan/Data/chessbot_data/training_data/lc0/"
sub_dirs = [f for f in os.listdir(leela_dir) if not f.endswith(".tar")]
all_files = 0
for sd in sub_dirs:
    d = os.path.join(leela_dir, sd)
    all_files += len([f for f in os.listdir(d) if f.endswith(".gz")])
    
print(all_files, f"training games found across {len(sub_dirs)} folders")



