from typing import List

from game import Player, Board, Dim, Color


class PlayerFaSKoe(Player):

    @staticmethod
    def sequences_by_color(board: Board) -> list[dict]:
        """Provides the longest sequence of one color in all rows and columns."""
        col_entries = list(
            map(
                lambda index: board.longest_sequence_specific(Dim.COL, index),
                range(0, board.col_count),
            )
        )
        row_entries = list(
            map(
                lambda index: board.longest_sequence_specific(Dim.ROW, index),
                range(0, board.row_count),
            )
        )
        diag_entries = list(
            map(
                lambda index: board.longest_sequence_specific(Dim.DIAG_CLOCKWISE, index),
                range(0, board.diag_count),
            )
        )
        diag_counter_entries = list(
            map(
                lambda index: board.longest_sequence_specific(
                    Dim.DIAG_COUNTER_CLOCKWISE, index
                ),
                range(0, board.diag_count),
            )
        )

        all_entries = col_entries + row_entries + diag_entries + diag_counter_entries
        all_entries = [x for x in all_entries if x is not None]
        return all_entries

    def resulting_row_if_inserted_in_col(self, board: Board, col_idx: int) -> int:
        col = board.col(index=col_idx)
        for idx in range(len(col)):
            if col[idx] is None:
                return idx
        return board.row_count + 1

    def opponent_color(self) -> Color:
        return Color((self.color.value + 1) % len(Color) + 1)

    def can_color_win(self, board: Board, color: Color) -> int | None:
        seq = PlayerFaSKoe.sequences_by_color(board)
        win_options = filter(lambda x: x["color"] == color and x["length"] == 3, seq)
        # TODO: Row {- x o x x - x}
        for option in win_options:
            match option["dim"]:
                case Dim.COL:
                    # Can this col be completed?
                    col: List[Color | None] = board.col(index=option["index"])
                    count = 0
                    for idx in range(board.row_count):
                        if idx >= len(col):
                            # this field is free
                            if count == 3:
                                return option["index"]
                            else:
                                break
                        if col[idx] == color:
                            # go on
                            count += 1
                        if col[idx] != color:
                            # ccc-combo breaker
                            count = 0
                    break
                case Dim.ROW:
                    # Can this row be completed?
                    row: List[Color | None] = board.row(index=option["index"])
                    count = 0
                    for idx, point in enumerate(row):
                        if count == 3:
                            if point is None:
                                return idx
                            elif idx > 4 and row[idx - 4] is None:
                                return idx - 4
                            else:
                                # SAD
                                break
                        if point is None:
                            count = 0
                        if point != color:
                            count = 0
                        if point == color:
                            count += 1
                    break
                case Dim.DIAG_CLOCKWISE:
                    curr_col_idx = option["index"]
                    curr_row_idx = 0 + (max(0, option["index"] - board.col_count))
                    count = 0
                    while curr_row_idx < option["index"]:
                        if board.columns[curr_col_idx][curr_row_idx] != color:
                            count = 0
                        if board.columns[curr_col_idx][curr_row_idx] == color:
                            count += 1
                        # idx up
                        curr_col_idx -= 1
                        curr_row_idx += 1
                        if curr_col_idx < 0 or curr_row_idx >= board.row_count:
                            break
                        if count == 3:
                            if board.columns[curr_col_idx][curr_row_idx] is None \
                                    and self.resulting_row_if_inserted_in_col(board=board, col_idx=curr_col_idx) == curr_row_idx:
                                return curr_col_idx
                            if (curr_col_idx + 4 < board.col_count) and (curr_row_idx - 4 >= 0) and board.columns[curr_col_idx + 4][curr_row_idx - 4] is None \
                                    and self.resulting_row_if_inserted_in_col(board=board, col_idx=curr_col_idx+4) == curr_row_idx - 4:
                                return curr_col_idx + 4
                    break
                case Dim.DIAG_COUNTER_CLOCKWISE:
                    curr_row_idx = option["index"]
                    curr_col_idx = 0 + (max(0, board.col_count - option["index"]))
                    count = 0
                    while curr_col_idx < option["index"]:
                        if board.columns[curr_col_idx][curr_row_idx] != color:
                            count = 0
                        if board.columns[curr_col_idx][curr_row_idx] == color:
                            count += 1
                        # idx up
                        curr_row_idx -= 1
                        curr_col_idx += 1
                        if curr_row_idx < 0 or curr_col_idx >= board.col_count:
                            break
                        if count == 3:
                            if board.columns[curr_col_idx][curr_row_idx] is None \
                                    and self.resulting_row_if_inserted_in_col(board=board, col_idx=curr_col_idx) == curr_row_idx:
                                return curr_col_idx
                            if (curr_row_idx + 4 < board.row_count) and (curr_col_idx - 4 >= 0) and board.columns[curr_col_idx - 4][curr_row_idx + 4] is None \
                                    and self.resulting_row_if_inserted_in_col(board=board, col_idx=curr_col_idx-4) == curr_row_idx + 4:
                                return curr_col_idx - 4


    def calculate_matrix(self, board: Board) -> List[int]:
        matrix = [0 for _ in range(board.col_count)]

        for col_idx in board.legal_moves():

            # columns upwards
            max_opponent_idx = -1
            current_own_streak = 0
            for idx, el in enumerate(board.col(col_idx)):
                if el == self.opponent_color():
                    max_opponent_idx = idx
                    current_own_streak = 0
                elif el == self.color:
                    current_own_streak += 1

            if max_opponent_idx <= board.row_count-4:
                matrix[col_idx] += 1 + current_own_streak

            # rows horizontally
            height = self.resulting_row_if_inserted_in_col(board=board, col_idx=col_idx)

            if height > board.row_count:
                continue

            streak = 0
            cur_col_idx = col_idx - 3 if col_idx - 3 > 0 else 0
            while cur_col_idx < col_idx + 3 and cur_col_idx < board.col_count:
                if self.resulting_row_if_inserted_in_col(board=board, col_idx=cur_col_idx) == height:
                    streak += 1
                elif board.row(height)[cur_col_idx] == self.color:
                    streak += 2
                else:
                    if streak >= 4:
                        matrix[col_idx] += 1 + streak - 4
                    streak = 0

                cur_col_idx += 1

        return matrix

    def next_move(self, board: Board, max_sec_per_step: float) -> int:
        # int -> 0..6 (column)
        # Let's play "What if?"
        if not any([x for x in [y for y in board.columns]]):
            # Board is empty
            next_move = int(board.col_count / 2)
        elif (next_move := self.can_color_win(board=board, color=self.color)) is not None:
            print("We can win")
        elif (next_move := self.can_color_win(board=board, color=self.opponent_color())) is not None:
            print("We must stop the opponent")
        else:
            # best effort
            matrix = sorted(enumerate(self.calculate_matrix(board=board)), key=lambda x: x[1], reverse=True)
            # print(matrix)
            next_move = matrix[0][0]
        return next_move