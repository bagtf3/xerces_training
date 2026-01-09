import numpy as np
from xerces_training.uci_to_idx import IDX_TO_UCI


def square_str_to_index(sq):
    """'a1' -> 0, 'h8' -> 63 (file a..h, rank 1..8)."""
    file_idx = ord(sq[0]) - ord("a")
    rank_idx = int(sq[1]) - 1
    return rank_idx * 8 + file_idx


def flip_index(idx):
    """Vertical flip across ranks: a1(0) -> a8(56), h1(7) -> h8(63)."""
    file = idx % 8
    rank = idx // 8
    return (7 - rank) * 8 + file


def uci_to_xerces_index(uci, stm_white=True):
    """
    Convert one UCI move string to Xerces 0..4287 index.
    stm_white: True if white to move (no flip); False flips squares.
    """
    if len(uci) < 4:
        raise ValueError("bad uci: " + repr(uci))
    from_raw = square_str_to_index(uci[0:2])
    to_raw = square_str_to_index(uci[2:4])

    # flip to STM perspective if black to move
    if not stm_white:
        from_slot = flip_index(from_raw)
        to_slot = flip_index(to_raw)
    else:
        from_slot = from_raw
        to_slot = to_raw

    # detect underpromotion: n->0, b->1, r->2 ; q or absent -> not underpromo
    underpromo_type = -1
    if len(uci) > 4:
        ch = uci[4].lower()
        if ch == "n":
            underpromo_type = 0
        elif ch == "b":
            underpromo_type = 1
        elif ch == "r":
            underpromo_type = 2

    if underpromo_type >= 0:
        # unique in 0..191: from_file + 8*to_file + 64*underpromo_type
        from_file = from_slot % 8
        to_file = to_slot % 8
        unique = from_file + 8 * to_file + 64 * underpromo_type
        flat = 4096 + unique
    else:
        flat = from_slot * 64 + to_slot

    return flat


def moves_to_xerces_indices(ucis, stm_white=True):
    """Map list of UCI strings to list of Xerces indices."""
    out = []
    for u in ucis:
        out.append(uci_to_xerces_index(u, stm_white))
    return out


def ep_from_history_planes(planes):
    p = np.asarray(planes)
    our_curr = p[0]
    their_curr = p[6]
    their_prev = p[19]
    candidates = set()
    
    # for EP, stm must have a pawn on the 5th rank
    if our_curr[4].sum() == 0:
        return []
    
    # opponent must also have a pawn on the 5th rank
    if their_curr[4].sum() == 0:
        return []
    
    # they also had to have a pawn on the 7th before
    if their_prev[6].sum() == 0:
        return []
    
    prev_occ = p[13:25].sum(axis=0)
    curr_occ = p[0:11].sum(axis=0)
    files = [f for f in range(8)]
    for f in files:
        # detect EP via their pawn moved 2 spaces last move
        if (their_curr[4, f] == 1) and (their_prev[6, f] == 1) and (their_curr[6, f]==0):
            # square must be empty now and last turn
            if (prev_occ[5, f] == 0) and (curr_occ[5, f] == 0):
                # check adjacent squares for our pawn
                to_check = [af for af in [f-1, f+1] if af in files]
                if our_curr[4, to_check].sum() >= 1:
                    candidates.add(8*5 + f)
    
    return sorted(candidates)


def planes_to_tokens(planes, us_oo, us_ooo, them_oo, them_ooo):
    BASE = 9
    KING_NO_CASTLE = 6
    KING_KS_ONLY = 7
    KING_QS_ONLY = 8
    KING_BOTH = 9
    EP_POSSIBLE = 19

    p = np.asarray(planes)
    
    if us_oo & us_ooo:
        us_king = KING_BOTH
    elif us_oo:
        us_king = KING_KS_ONLY
    elif us_ooo:
        us_king = KING_QS_ONLY
    else:
        us_king = KING_NO_CASTLE

    if them_oo & them_ooo:
        them_king = BASE + KING_BOTH
    elif them_oo:
        them_king = BASE + KING_KS_ONLY
    elif them_ooo:
        them_king = BASE + KING_QS_ONLY
    else:
        them_king = BASE + KING_NO_CASTLE
    
    plane_to_token = {
        0:1, 1:2, 2:3, 3:4, 4:5,
        6:10, 7:11, 8:12, 9:13, 10:14
    }

    out = np.zeros(64, dtype=np.int64)
    for r in range(8):
        for f in range(8):
            idx = r*8 + f
            for i in range(12):
                if p[i, r, f] > 0.5:
                    if i == 5:
                        out[idx] = us_king
                    elif i == 11:
                        out[idx] = them_king
                    else:
                        out[idx] = plane_to_token[i]
                        break

    ep_candidates = ep_from_history_planes(planes)
    if ep_candidates:
        for ep_square in ep_candidates:
            # should be 0
            assert out[ep_square] == 0
            # 19 indicates EP is possible for STM
            out[ep_square] = EP_POSSIBLE

    return out


def get_policy_vector(probs, stm, us_ooo, us_oo):
    '''Get all moves with training probabilities greater than 0, 
    sorted descending... returned as OrderedDict (uci => prob)'''
    
    # always STM-as-white
    idx_to_uci_idx = 2*stm + (us_ooo | us_oo)
    idx_to_uci = IDX_TO_UCI[idx_to_uci_idx]
    
    mask = probs >= 0
    idxs = np.nonzero(mask)[0]        
    
    moves = [idx_to_uci[idx] for idx in idxs]
    xerces_indices = moves_to_xerces_indices(moves, stm_white=(stm==0))
    xerces_policy = np.zeros(64 * 67, dtype=np.float32)
    for xi, idx in zip(xerces_indices, idxs):
        xerces_policy[xi] = probs[idx]

    mask = 1*(xerces_policy > 0)
    return xerces_policy, mask


def planes_to_fen(planes, stm_white, us_oo, us_ooo, them_oo, them_ooo, start_idx=0):
    white_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
    black_chars = ['p', 'n', 'b', 'r', 'q', 'k']
    ranks = [r for r in range(8)]

    if stm_white:
        chars = white_chars + black_chars
        white_oo = us_oo
        white_ooo = us_ooo
        black_oo = them_oo
        black_ooo = them_ooo
    else:
        ranks = ranks[::-1]
        chars = black_chars + white_chars
        white_oo = them_oo
        white_ooo = them_ooo
        black_oo = us_oo
        black_ooo = us_ooo

    board = [['' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for f in range(8):
            for i in range(12):
                if planes[start_idx + i, r, f] > 0.5:
                    board[ranks[r]][f] = chars[i]
                    break
    
    fen_rows = []
    for row in board[::-1]:
        empty = 0
        parts = []
        for ch in row:
            if not ch:
                empty += 1
            else:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append(ch)
        if empty:
            parts.append(str(empty))
        fen_rows.append(''.join(parts))
    placement = '/'.join(fen_rows)

    castling = ''
    if white_oo:
        castling += 'K'
    if white_ooo:
        castling += 'Q'
    if black_oo:
        castling += 'k'
    if black_ooo:
        castling += 'q'
    if castling == '':
        castling = '-'

    stm_char = 'w' if stm_white else 'b'
    # ep check for current position only (use ep_from_history_planes)
    ep = '-'
    candidates = ep_from_history_planes(planes)
    if candidates:
        if not stm_white:
            candidates = [flip_index(c) for c in candidates]
        idx = candidates[0]
        file = idx % 8
        rank = idx // 8
        ep = chr(ord('a') + file) + str(rank + 1)
    
    return f"{placement} {stm_char} {castling} {ep} 0 1"


def to_xerces_tuple(planes, probs, winner, best_q, extra_info):
    stm, us_oo, us_ooo, them_oo, them_ooo, inv = extra_info
    stm_is_white = stm == 0
    
    planes = np.reshape(np.frombuffer(planes, dtype=np.float32), (112, 8, 8))
    tokens = planes_to_tokens(planes, us_oo, us_ooo, them_oo, them_ooo)
    
    probs = np.frombuffer(probs, dtype=np.float32)
    policy, mask = get_policy_vector(probs, stm, us_oo, us_ooo)
    
    winner = np.frombuffer(winner, dtype=np.float32)
    Z = np.dot(winner, np.array([1, 0, -1]))
    best_q = np.frombuffer(best_q, dtype=np.float32)
    q = best_q[0] - best_q[2]
    y = 0.5 * Z + 0.5 * q

    fen = planes_to_fen(planes, stm_is_white, us_oo, us_ooo, them_oo, them_ooo)
    return tokens, mask, policy, y, fen
