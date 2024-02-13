import math
from typing import Optional

import numpy as np
from beartype import beartype

from .constants import (
    MAX_TROOPS,
    Board,
    Coords,
    EndgameState,
    Evaluator,
    Move,
    MovesRecord,
)
from .game_state import GameState, Player, make_move, undo_move
from .neighbours_table import lookup_neighbours
from .transposition_table import NodeFlag, TranspositionTable, evaluate_entry
from .zobrist import compute_zobri_hash


def get_default_board() -> Board:
    board = np.zeros((5, 5), dtype=np.int8)
    board[0][4] = -10
    board[4][0] = 10
    return board


def get_endgame_state(state: Board) -> EndgameState:
    flattened_state = state.flatten()
    if np.all(flattened_state == 0):
        return EndgameState.DRAW
    elif np.all(flattened_state <= 0):
        return EndgameState.LOSS
    elif np.all(flattened_state >= 0):
        return EndgameState.WIN
    else:
        return EndgameState.ONGOING


def get_possible_moves(board: Board, player_to_move: Player) -> list[Move]:
    temp_board = board * player_to_move
    moves: list[Move] = []

    for index, value in np.ndenumerate(temp_board):
        if value <= 0:
            continue

        if value < MAX_TROOPS:
            moves.append((Coords(index), Coords(index)))

        for neighbor in lookup_neighbours(*index):
            troops_capacity = MAX_TROOPS - temp_board[neighbor]

            if troops_capacity == 0:
                continue

            troops_transfers_to_consider = (temp_board[index], temp_board[index] - 1, 1)

            for troops_to_transfer in troops_transfers_to_consider:
                troops = min(troops_capacity, troops_to_transfer) * player_to_move
                moves.append((Coords(index), neighbor, int(troops)))

    return moves


def order_moves(moves: list[Move], player_to_move: Player) -> list[Move]:
    scores = np.zeros(shape=(len(moves)), dtype=np.int8)

    target = 0 if player_to_move == Player.RED else 10
    for index, move in enumerate(moves):
        score = 0

        if len(move) == 2:
            score += 5

        if len(move) == 3:
            score += 3

        score_lost_due_to_position = target - move[1][0] - move[1][1]
        score += 5 - score_lost_due_to_position

        scores[index] = score

    return [x for _, x in sorted(zip(scores, moves), reverse=True)]


storage = TranspositionTable(1024)
SINGULAR_MOVE_MARGIN = 1.0
SINGULAR_EXTENSION_DEPTH_LIMIT = 3
SINGULAR_MOVE_EXTENSION = 1
MAX_DEPTH = 10


@beartype
def pvs(
    evaluator: Evaluator,
    state: GameState,
    depth: int,
    ply: int,
    alpha: float,
    beta: float,
    move_to_skip: Optional[Move],
    is_extended: bool,
) -> float:
    global best_move
    endgame_state = get_endgame_state(state.board)
    if depth <= 0 or endgame_state != EndgameState.ONGOING or ply >= MAX_DEPTH:
        return float(sum(evaluator(state.board.flatten() * state.player_to_move)))

    moves = get_possible_moves(state.board, state.player_to_move)
    moves = order_moves(moves, state.player_to_move)
    best_move = moves[0]
    best_score = -math.inf
    node_type = NodeFlag.UPPER_BOUND
    is_root = ply == 0
    is_pv_node = beta - alpha != 1
    do_full_search = True

    # =====================================================================#
    # TRANSPOSITION TABLE PROBING: Probe the transposition table to see if #
    # we have a useable matching entry for the current position. If we get #
    # a hit, return the score and stop searching.                          #
    # =====================================================================#

    entry = storage.get(state.hash)
    tt_score, tt_can_be_used, tt_move = evaluate_entry(entry, depth, alpha, beta)
    tt_hit = tt_move is not None

    if tt_hit:
        moves.insert(0, tt_move)
        best_move = tt_move

    if tt_can_be_used and not is_root and not move_to_skip == tt_move:
        return tt_score

    for move in moves:
        if move == move_to_skip:
            continue
        # =====================================================================#
        # LATE MOVE REDUCTION: Since our move ordering is good, the            #
        # first move is likely to be the best move in the position, which      #
        # means it's part of the principal variation. So instead of searching  #
        # every move equally, search the first move with full-depth and full-  #
        # window, and search every move after with a reduced-depth and null-   #
        # window to prove it'll fail low cheaply. If it raises alpha however,  #
        # we have to use a full-window, a full-depth, or both to get an        #
        # accurate score for the move.                                         #
        # =====================================================================#

        make_move(state, move)
        if do_full_search:
            next_depth = depth - 1
            # =====================================================================#
            # SINGULAR EXTENSIONS: If the best move in the position is from the    #
            # transposition table, check if it's "singular", i.e. substantially    #
            # better than any of the other moves in the position. We do this by    #
            # getting the move's score from the transposition table, and subtract- #
            # ing from it a certain margin. We then do a reduced depth search, in  #
            # the current position, with a window centered around the reduced      #
            # score, to see if any of the other moves can match or beat the        #
            # reduced score. If any of the moves can't even beat the reduced score #
            # we extend the search, since the best move seems particularly         #
            # important, and we don't want to miss something. We just need to be   #
            # careful not to let the search "explode", by repeadtly extending the  #
            # the depth, so once we've extended the search down a certain path,    #
            # make sure to stop extending any children nodes.                      #
            # =====================================================================#

            if (
                not is_extended
                and depth >= SINGULAR_EXTENSION_DEPTH_LIMIT
                and tt_move == move
                and is_pv_node
                and entry is not None
                and entry.flag in (NodeFlag.LOWER_BOUND, NodeFlag.EXACT)
            ):

                undo_move(state)
                score_to_beat = tt_score - SINGULAR_MOVE_MARGIN
                r = 3 + depth // 6
                next_best_score = pvs(
                    evaluator,
                    state,
                    depth - 1 - r,
                    ply + 1,
                    score_to_beat,
                    score_to_beat + 1,
                    move,
                    True,
                )

                # If the next best score is less than or equal to the score to beat,
                # we know the above search failed-low, a move wasn't found that could beat
                # the tt move score even with a margin, and so the tt move is singular. So
                # we should spend some extra time searching it.
                if next_best_score <= score_to_beat:
                    next_depth += SINGULAR_MOVE_EXTENSION

                make_move(state, move)

            score = -pvs(
                evaluator,
                state,
                next_depth,
                ply + 1,
                -beta,
                -alpha,
                move_to_skip,
                is_extended,
            )
        else:
            # Search with a null window.
            score = -pvs(
                evaluator,
                state,
                depth - 1,
                ply + 1,
                -alpha - 1,
                -alpha,
                move_to_skip,
                is_extended,
            )

            if alpha < score < beta:
                # If it failed high, do a full search.
                score = -pvs(
                    evaluator,
                    state,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    move_to_skip,
                    is_extended,
                )
        undo_move(state)

        if score > best_score:
            best_score = score
            best_move = move
            do_full_search = False

        # If we have a beta-cutoff (i.e this move gives us a score better than what
        # our opponent can already guarantee early in the tree), return beta and
        # the move that caused the cutoff as the best move.
        if score >= beta:
            node_type = NodeFlag.LOWER_BOUND
            break

        # If the score of this move is better than alpha (i.e better than the score
        # we can currently guarantee), set alpha to be the score and the best move
        # to be the move that raised alpha.
        if score > alpha:
            alpha = score
            node_type = NodeFlag.EXACT

    storage.add(state.hash, beta, best_move, depth, node_type)
    return best_score


best_move = None


def select_best_move(game_state: GameState, evaluator: Evaluator, depth: int) -> Move:
    global best_move
    score = pvs(evaluator, game_state, depth, 0, -math.inf, math.inf, None, False)
    assert best_move is not None
    return best_move


def play(
    player: Evaluator,
    opponent: Evaluator,
    rounds: int,
    depth: int,
) -> tuple[EndgameState, Board, MovesRecord]:
    board = get_default_board()
    player_to_move = Player.BLUE
    zobrist_key = compute_zobri_hash(board, player_to_move)
    game_state = GameState(board, zobrist_key, player_to_move)
    endgame_state = EndgameState.ONGOING

    for _ in range(rounds):
        evaluator = player if game_state.player_to_move == Player.BLUE else opponent
        move = select_best_move(game_state, evaluator, depth)
        make_move(game_state, move)
        # print(game_state.board)
        endgame_state = get_endgame_state(game_state.board)

        if endgame_state != EndgameState.ONGOING:
            break

    return endgame_state, board, game_state.history
