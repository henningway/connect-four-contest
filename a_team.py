from pedantic import overrides

from game import Player, Board
from lru_cache import LRUCache


class FarsBot(Player):
    @overrides(Player)
    async def next_move(self, board: Board) -> int:
        if board.is_empty:
            return 3   # MID

        return 4


def get_tt_entry(value, upper_bound: bool = False, lower_bound: bool = False) -> dict[str, int | bool]:
    return {'value': value, 'UB': upper_bound, 'LB': lower_bound}


def solve(board):
    transposition_table = LRUCache(4096)

    def recurse(alpha, beta):
        alpha_original = alpha

        # transposition table lookup
        if board.get_key() in transposition_table:
            entry = transposition_table[board.get_key()]
            if entry['LB']:
                alpha = max(alpha, entry['value'])  # lower bound stored in TT
            elif entry['UB']:
                beta = min(beta, entry['value'])    # upper bound stored in TT
            else:
                return entry['value']               # exact value stored in TT
            if alpha >= beta:
                return entry['value']               # cut-off (from TT)

        # negamax implementation
        if board.winning_board_state():
            return board.get_score()        # base case 1: winning alignment
        elif board.moves == board.w * board.h:
            return 0                        # base case 2: draw game
        value = -board.w * board.h
        for col in board.get_search_order():
            board.play(col)
            value = max(value, -recurse(-beta, -alpha))
            board.backtrack()
            alpha = max(alpha, value)
            if alpha >= beta:
                break                   # alpha cut-off

        # transposition table storage
        if value <= alpha_original:
            transposition_table[board.get_key()] = get_tt_entry(value, upper_bound=True)
        elif value >= beta:
            transposition_table[board.get_key()] = get_tt_entry(value, lower_bound=True)
        else:
            transposition_table[board.get_key()] = get_tt_entry(value)       # store exact in TT

        return value
    return recurse(-1e9, 1e9)