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

def planes_to_tokens(planes, stm, us_oo, us_ooo, them_oo, them_ooo):
    BASE = 9
    KING_NO_CASTLE = 6
    KING_KS_ONLY = 7
    KING_QS_ONLY = 8
    KING_BOTH = 9

    p = np.asarray(planes)
    if p.shape != (112, 8, 8):
        raise ValueError("expected planes shape (112, 8, 8)")

    stm_is_white = bool(int(stm) == 1)

    def king_type(ks, qs):
        if ks and qs:
            return KING_BOTH
        if ks:
            return KING_KS_ONLY
        if qs:
            return KING_QS_ONLY
        return KING_NO_CASTLE

    # compute wht_king and blk_king in STM-as-white semantics
    if stm_is_white:
        wht_king = king_type(us_oo, us_ooo)
        blk_king = king_type(them_oo, them_ooo)
    else:
        wht_king = king_type(them_oo, them_ooo)
        blk_king = king_type(us_oo, us_ooo)

    out = np.zeros(64, dtype=np.int16)

    base = 0  # most recent block at planes[0..11]
    # piece order in planes: 0..5 = stm pieces P N B R Q K
    #                       6..11 = other-side P N B R Q K
    for rank in range(8):
        for file in range(8):
            sq = rank * 8 + file
            token = 0
            placed = False

            # same-as-stm pieces (planes 0..5)
            for i in range(6):
                if float(p[base + i, rank, file]) > 0.5:
                    # non-king pieces: 1..6
                    if i != 5:
                        token = i + 1
                    else:
                        # king: choose STM/OPP king-type later via base
                        token = i + 1  # placeholder; will adjust below
                    placed = True
                    same_as_stm = True
                    piece_index = i
                    break

            if not placed:
                # opponent pieces (planes 6..11)
                for i in range(6):
                    if float(p[base + 6 + i, rank, file]) > 0.5:
                        if i != 5:
                            token = BASE + (i + 1)
                        else:
                            token = BASE + (i + 1)
                        placed = True
                        same_as_stm = False
                        piece_index = i
                        break

            if not placed:
                continue

            # correct king tokens (special encoding)
            if piece_index == 5:  # king
                if same_as_stm:
                    # same-as-stm king uses wht_king when STM is white semantics
                    token = wht_king if same_as_stm else blk_king
                    # token currently is king_val; if piece is opponent we prefix BASE
                    if not same_as_stm:
                        token = BASE + token
                else:
                    # opponent king: use blk_king in STM-as-white semantics
                    token = BASE + blk_king if not same_as_stm else blk_king

            # ensure STM perspective: flip index if STM is black
            write_idx = sq if stm_is_white else (sq ^ 56)
            out[write_idx] = int(token)

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
    xerces_indices = moves_to_xerces_indices(moves, stm_white=stm)
    xerces_policy = np.zeros(64 * 67, dtype=np.float32)
    for xi, idx in zip(xerces_indices, idxs):
        xerces_policy[xi] = probs[idx]

    return xerces_policy, 1*mask


def planes_to_fen(planes, stm, us_oo, us_ooo, them_oo, them_ooo):
    """
    planes: numpy array shape (112,8,8) - LC0/Xerces bitplanes
    stm: 1 if side-to-move is white, else 0
    us_oo/us_ooo/them_oo/them_ooo: castling booleans for side-to-move (us)
      and opponent (them). 'oo' = kingside, 'ooo' = queenside.
    returns: FEN string
    """
    p = np.asarray(planes)
    if p.ndim != 3 or p.shape[1:] != (8, 8):
        raise ValueError("expected planes shape (112,8,8)")

    base = 0  # most recent block is at planes[0..12)
    # plane order base+0..5 = side-to-move pieces P N B R Q K
    # base+6..11 = opponent pieces P N B R Q K
    # If stm==1 the "side-to-move" planes are white pieces else they are black.
    if int(stm) == 1:
        ours_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
        theirs_chars = ['p', 'n', 'b', 'r', 'q', 'k']
    else:
        ours_chars = ['p', 'n', 'b', 'r', 'q', 'k']
        theirs_chars = ['P', 'N', 'B', 'R', 'Q', 'K']

    # build 8x8 board array (rank1..8 -> row 0..7, file a..h -> col 0..7)
    board = [['' for _ in range(8)] for _ in range(8)]
    # iterate squares; choose first plane >0.5 if multiple set
    for r in range(8):
        for f in range(8):
            placed = False
            for i in range(6):
                if float(p[base + i, r, f]) > 0.5:
                    board[r][f] = ours_chars[i]
                    placed = True
                    break
            if placed:
                continue
            for i in range(6):
                if float(p[base + 6 + i, r, f]) > 0.5:
                    board[r][f] = theirs_chars[i]
                    break

    # build FEN ranks from rank8->rank1 (board row 7 -> row 0)
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

    # castling mapping: if stm==1, us==white else us==black
    castling = ''
    if int(stm) == 1:
        if us_oo:
            castling += 'K'
        if us_ooo:
            castling += 'Q'
        if them_oo:
            castling += 'k'
        if them_ooo:
            castling += 'q'
    else:
        if us_oo:
            castling += 'k'
        if us_ooo:
            castling += 'q'
        if them_oo:
            castling += 'K'
        if them_ooo:
            castling += 'Q'
    if castling == '':
        castling = '-'

    stm_char = 'w' if int(stm) == 1 else 'b'
    ep = '-'  # no ep info provided
    return f"{placement} {stm_char} {castling} {ep} 0 1"


def to_xerces_tuple(planes, probs, winner, best_q, extra_info):
    stm, us_oo, us_ooo, them_oo, them_ooo = extra_info
    stm_is_white = not stm
    
    planes = np.reshape(np.frombuffer(planes, dtype=np.float32), (112, 8, 8))
    tokens = planes_to_tokens(planes, stm_is_white, us_oo, us_ooo, them_oo, them_ooo)
    
    probs = np.frombuffer(probs, dtype=np.float32)
    policy, mask = get_policy_vector(probs, stm, us_oo, us_ooo)
    
    winner = np.frombuffer(winner, dtype=np.float32)
    
    Z = np.dot(winner, np.array([1, 0, -1]))
    best_q = np.frombuffer(best_q, dtype=np.float32)
    q = best_q[0] - best_q[2]
    y = 0.5 * Z + 0.5 * q

    fen = planes_to_fen(planes, stm_is_white, us_oo, us_ooo, them_oo, them_ooo)
    return tokens, mask, policy, y, fen
