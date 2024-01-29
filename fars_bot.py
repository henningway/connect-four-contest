import copy
import math
import random

import numpy as np
from pedantic import overrides

from game import Player, Board, Color

LENGTH = 4
ROWS = 6
COLUMNS = 7
EMPTY_PIECE = 0


class FarsBot(Player):
    """
    https://github.com/oskarjonszon/Connect_4/tree/main
    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    """

    @overrides(Player)
    def next_move(self, board: Board, max_sec_per_step: float) -> int:
        board.print()

        if board.is_empty:
            return 3   # MID

        # no idea, best guess
        if max_sec_per_step > 30:
            depth = 7
        elif max_sec_per_step > 5:
            depth = 6
        elif max_sec_per_step > 1:
            depth = 5
        elif max_sec_per_step > 0.5:
            depth = 4
        else:
            depth = 3

        array = np.array(board.columns)
        array[array == None] = 0  # replace None with zeroes in memory
        array = np.rot90(array)  # rotate it
        array = array.astype(dtype=np.int32)
        column, value = self.minimax(
            board=array,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            maximize=True,
        )
        print(f'FarsBot choose column {column}')
        return column

    @property
    def ai_piece(self) -> int:
        return self.color.value

    @property
    def p_piece(self) -> int:
        if self.color == Color.RED:
            return Color.YELLOW.value

        return Color.RED.value

    @staticmethod
    def available_moves(board):
        return set(i for i in range(COLUMNS) if board[0][i] == 0)

    @staticmethod
    def drop_piece(board: np.array, action: int, player: int) -> None:
        for i in list(reversed(range(ROWS))):
            if np.any(board[i][action] == 0):
                board[i][action] = player
                return

    def evaluate(self, board):
        score = 0

        # Central Nodes
        center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
        center_count = center_array.count(self.ai_piece)
        score += center_count * 3

        # Horizontal Nodes
        for r in range(ROWS):
            row = [int(i) for i in list(board[r, :])]
            for c in range(COLUMNS - 3):
                interval = row[c: c + LENGTH]
                score += self.evaluate_interval(interval)

        # Vertical Nodes
        for c in range(COLUMNS):
            column = [int(i) for i in list(board[:, c])]
            for r in range(ROWS - 3):
                interval = column[r: r + LENGTH]
                score += self.evaluate_interval(interval)

        # Positive Diagonal
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                interval = [board[r + i][c + i] for i in range(LENGTH)]
                score += self.evaluate_interval(interval)

        # Negative Diagonal
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                interval = [board[r + 3 - i][c + i] for i in range(LENGTH)]
                score += self.evaluate_interval(interval)

        return score

    def evaluate_interval(self, interval):
        score = 0

        if interval.count(self.ai_piece) == 3 and interval.count(EMPTY_PIECE) == 1:
            score += 5

        if interval.count(self.ai_piece) == 2 and interval.count(EMPTY_PIECE) == 2:
            score += 2

        if interval.count(self.p_piece) == 3 and interval.count(EMPTY_PIECE) == 0:
            score -= 4

        return score

    def minimax(self, board: np.array, depth: int, alpha: float, beta: float, maximize: bool) -> tuple[int | None, int]:
        """ Returns columns, value """

        valid_moves = self.available_moves(board)
        
        if depth == 0:
            return None, self.evaluate(board)
        
        assert depth > 0, f'Depth: {depth}, invalid'

        if maximize:
            value = -math.inf
            column = random.choice(list(valid_moves))

            for action in valid_moves:
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, action, self.ai_piece)
                score = self.minimax(b_copy, depth - 1, alpha, beta, False)[1]

                if score > value:
                    value = score
                    column = action

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return column, value
        else:
            value = math.inf
            column = random.choice(list(valid_moves))

            for action in valid_moves:
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, action, self.p_piece)
                score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]

                if score < value:
                    value = score
                    column = action

                beta = min(beta, value)

                if alpha >= beta:
                    break

            return column, value
