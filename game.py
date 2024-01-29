from abc import ABC
import asyncio
from enum import Enum
from itertools import groupby, repeat
from pedantic import in_subprocess
from random import choice
from time import sleep
from typing import Optional

DEFAULT_MOVE_TIMEOUT = 0.1
LAST_PRINT_LEN = 0


class Color(Enum):
    RED = 1
    YELLOW = 2


class Dim(Enum):
    COL = 1
    ROW = 2
    DIAG_CLOCKWISE = 3
    DIAG_COUNTER_CLOCKWISE = 4


def reprint(msg, finish=False):
    global LAST_PRINT_LEN

    print(" " * LAST_PRINT_LEN, end="\r")

    if finish:
        end = "\n"
        LAST_PRINT_LEN = 0
    else:
        end = "\r"
        LAST_PRINT_LEN = len(msg)

    print(msg, end=end)


def color_to_letter(color: Optional[Color], empty_char=".") -> str:
    if color is None:
        return empty_char
    if color is Color.RED:
        return "R"
    if color is Color.YELLOW:
        return "Y"


def letter_to_color(letter: str) -> Optional[Color]:
    if letter == " " or letter == ".":
        return None
    if letter == "R":
        return Color.RED
    if letter == "Y":
        return Color.YELLOW


def sequence_to_word(sequence: [Optional[Color]]) -> str:
    return "".join(map(lambda color: color_to_letter(color, " "), sequence))


def longest_repeat(word: [str]) -> [Optional[Color]]:
    """Provides the longest consecutive sequence of characters in given word."""
    if len(word) == 0:
        return []

    return sorted([list(g) for _, g in groupby(word)], key=len)[-1]


def longest_color_repeat(sequence: [Optional[Color]]) -> [Color]:
    """Provides the longest consecutive non-nil sequence of colors."""
    words = sequence_to_word(sequence).split()

    if len(words) == 0:
        return []

    return max(map(longest_repeat, words), key=len)


def transpose(matrix: [[any]]) -> [[any]]:
    width = len(matrix)
    height = len(matrix[0])

    result = []

    for i in range(height):
        result.append([matrix[j][i] for j in range(width)])

    return result


class Board:
    def __init__(self, cols=7, rows=6) -> None:
        self.col_count = cols
        self.row_count = rows
        self.diag_count = cols + rows - 1
        self.columns = [[None for _ in range(rows)] for _ in range(cols)]

    def rows(self) -> [[Optional[Color]]]:
        return [self.row(row) for row in range(self.row_count)]

    def diags_clockwise(self) -> [[Optional[Color]]]:
        """
        Provides a list representation of the diagonals produced by rotating the board clockwise 45 degrees.
        """
        skewed = [
            list(repeat(None, row_index))
            + row
            + list(repeat(None, self.row_count - row_index - 1))
            for row_index, row in enumerate(self.rows())
        ]
        return transpose(skewed)

    def diags_counter_clockwise(self) -> [[Optional[Color]]]:
        """
        Provides a matrix representation of the diagonals produced by rotating the board counter-clockwise 45 degrees.

        We achieve this by left-padding each row with a number of None's equal to the row's index, essentially skewing the board to the right. To make the matrix transposable we have to right-pad as well, in mirrored fashion to the left-padded None's.

        By transposing, we get the columns of the skewed matrix, providing the diagonals.
        """
        skewed = [
            list(repeat(None, self.row_count - row_index - 1))
            + row
            + list(repeat(None, row_index))
            for row_index, row in enumerate(self.rows())
        ]
        return transpose(skewed)

    def row(self, index: int) -> [Optional[Color]]:
        """Provides a list representation of the row at given index."""
        return [
            self.columns[col][index] if len(self.columns[col]) > index else None
            for col in range(self.col_count)
        ]

    def col(self, index: int) -> [Optional[Color]]:
        """Provides a list representation of the column at given index."""
        return self.columns[index]

    def diag_clockwise(self, index: int) -> [Optional[Color]]:
        """
        Provides a list representation of the diagonal (obtained by clockwise rotation) at given index. Note that the
        result has padded None-values, which limits the utility of this function.
        """
        return self.diags_clockwise()[index]

    def diag_counter_clockwise(self, index: int) -> [Optional[Color]]:
        """
        Provides a list representation of the diagonal (obtained by counter-clockwise rotation) at given index. Note
        that the result has padded None-values, which limits the utility of this function.
        """
        return self.diags_counter_clockwise()[index]

    def print(self):
        """Renders a representation of the board to stdout."""
        for i in range(self.row_count - 1, -1, -1):
            row = list(map(color_to_letter, self.row(i)))
            print(row)

    def legal_moves(self) -> [int]:
        """Provides a list of columns with open gaps (i.e. columns that are not full)."""
        return list(
            map(
                lambda value: value[0],
                filter(
                    lambda col: len(
                        list(filter(lambda value: value is not None, col[1]))
                    )
                    < self.row_count,
                    enumerate(self.columns),
                ),
            )
        )

    def is_legal_move(self, col: int) -> bool:
        """Tells whether given column index has open gaps."""
        return col in self.legal_moves()

    def register_move(self, color: Color, col: int):
        """Puts a token of the given color into given column index."""
        assert self.is_legal_move(col), "Not a legal move."
        column = self.columns[col]
        lowest_none_index: Optional[int] = next(
            filter(lambda x: x[1] is None, enumerate(column))
        )[0]
        self.columns[col][lowest_none_index] = color

    def is_full(self) -> bool:
        """Tells whether all the columns are filled."""
        return len(self.legal_moves()) == 0

    def longest_sequence_specific(self, dim: Dim, index: int) -> Optional[dict]:
        """Provides the longest consecutive sequence of one color in given row or column."""
        sequence: [Optional[Color]] = (
            self.col
            if dim == Dim.COL
            else (
                self.row
                if dim == Dim.ROW
                else (
                    self.diag_clockwise
                    if dim == Dim.DIAG_CLOCKWISE
                    else self.diag_counter_clockwise
                )
            )
        )(index)

        long_sequence: [str] = longest_color_repeat(sequence)

        return (
            {
                "color": letter_to_color(long_sequence[0]),
                "length": len(long_sequence),
                "dim": dim,
                "index": index,
            }
            if len(long_sequence) > 0
            else None
        )

    def longest_sequence(self) -> Optional[dict]:
        """Provides the longest sequence of one color in all rows and columns."""
        col_entries = list(
            map(
                lambda index: self.longest_sequence_specific(Dim.COL, index),
                range(0, self.col_count),
            )
        )
        row_entries = list(
            map(
                lambda index: self.longest_sequence_specific(Dim.ROW, index),
                range(0, self.row_count),
            )
        )
        diag_entries = list(
            map(
                lambda index: self.longest_sequence_specific(Dim.DIAG_CLOCKWISE, index),
                range(0, self.diag_count),
            )
        )
        diag_counter_entries = list(
            map(
                lambda index: self.longest_sequence_specific(
                    Dim.DIAG_COUNTER_CLOCKWISE, index
                ),
                range(0, self.diag_count),
            )
        )

        all_entries = col_entries + row_entries + diag_entries + diag_counter_entries

        max_entry = max(
            all_entries, key=(lambda entry: entry["length"] if entry is not None else 0)
        )

        if max_entry is None:
            return None

        return max_entry


class Player(ABC):
    def __init__(self, color: Color, with_timeout: bool = True) -> None:
        super().__init__()
        self.color = color
        self.with_timeout = with_timeout

    async def get_next_move(self, board: Board, max_sec_per_step: float) -> int:
        """
        Picks and runs the correct choice of the sync/async versions of next_move.
        """
        if self.with_timeout:
            return await self.subprocess_next_move(board, max_sec_per_step)
        else:
            return self.next_move(board, max_sec_per_step)

    def next_move(self, board: Board, max_sec_per_step: float) -> int:
        """
        Provides a column index to place a token of the player's color in. Should respect `max_sec_per_step` to not get
        timed out (other player wins).
        """
        pass

    @in_subprocess
    def subprocess_next_move(self, board: Board, max_sec_per_step: float) -> int:
        """
        Provides a column index to place a token of the player's color in. Should respect `max_sec_per_step` to not get
        timed out (other player wins).
        """
        return self.next_move(board, max_sec_per_step)


class MonkeyPlayer(Player):
    def next_move(self, board: Board, max_sec_per_step: float) -> int:
        # sleep(DEFAULT_MOVE_TIMEOUT) # uncomment this to simulate timeouts/game defaulting
        return choice(board.legal_moves())


class HumanPlayer(Player):
    def next_move(self, board: Board, max_sec_per_step: float) -> int:
        while True:
            try:
                col = int(input(f"Pick a column (1-{board.row_count+1}): "))
                if col in range(1, board.row_count + 2):
                    break
                else:
                    print(f"Given number is not in range.")
            except ValueError:
                print("That's not a valid number. Please try again.")

        return int(col) - 1


class Game:
    def __init__(self, p1: Player, p2: Player, max_sec_per_step: float) -> None:
        self.board = Board()
        self.p1 = p1
        self.p2 = p2
        self.active_player = self.p1
        self.default_winner = None
        self.max_sec_per_step = max_sec_per_step

    async def step(self):
        """Prompts the active player to decide on its next move and switches the active player."""
        self.board.print()
        next_move = await self.active_player.get_next_move(
            self.board, self.max_sec_per_step
        )
        self.board.register_move(self.active_player.color, next_move)
        print()
        self.active_player = self.p1 if self.active_player is self.p2 else self.p2

    def is_finished(self) -> bool:
        """Tells whether the game has ended (board is full or one player has connected four)."""
        return self.board.is_full() or self.winner() is not None

    def winner(self) -> Optional[Color]:
        """Provides the winner of the game, if there is any."""
        if self.default_winner is not None:
            return self.default_winner.color

        longest = self.board.longest_sequence()
        if longest is None:
            return None
        return longest["color"] if longest["length"] >= 4 else None

    def default(self):
        self.default_winner = self.p1 if self.active_player is self.p2 else self.p2


class Simulation:
    """
    Provides static methods for game simulation.

    Players have a limited amount of time for each move (`max_ms_per_step`) and get timed out when they take longer,
    defaulting the other player as winner of the game.
    """

    @staticmethod
    async def single(
        p1: Player, p2: Player, max_sec_per_step: float = DEFAULT_MOVE_TIMEOUT
    ):
        """Runs a single game and provides the final board and winner on stdout. p1 is the starting player."""
        game = Game(p1, p2, max_sec_per_step)

        try:
            while not game.is_finished():
                await asyncio.wait_for(game.step(), max_sec_per_step)
        except asyncio.exceptions.TimeoutError:
            print(game.active_player.color, "timed out!")
            game.default()

        game.board.print()

        print(
            "Winner:",
            game.winner(),
            "(won by default)" if game.default_winner is not None else "",
        )

    @staticmethod
    async def many(
        p1: Player,
        p2: Player,
        runs: int,
        max_sec_per_step: float = DEFAULT_MOVE_TIMEOUT,
    ):
        """
        Runs the given number of games of p1 against p2, with the starting player alternating each game. Provides the
        result statistic (games won by p1, games won by p2 and number of draws) on stdout.
        """
        wins = {"red": 0, "yellow": 0, "draw": 0, "red_timeout": 0, "yellow_timeout": 0}

        for i in range(0, runs):
            percentage = f"{round(i / runs * 100)}%"
            reprint(percentage)

            game = (
                Game(p1, p2, max_sec_per_step)
                if i % 2 == 0
                else Game(p2, p1, max_sec_per_step)
            )

            try:
                while not game.is_finished():
                    await asyncio.wait_for(game.step(), max_sec_per_step)

                winner = game.winner()

                match winner:
                    case Color.RED:
                        wins["red"] = wins["red"] + 1
                    case Color.YELLOW:
                        wins["yellow"] = wins["yellow"] + 1
                    case None:
                        wins["draw"] = wins["draw"] + 1
            except asyncio.exceptions.TimeoutError:
                game.default()

                winner = game.winner()

                match winner:
                    case Color.RED:
                        wins["red_timeout"] = wins["red_timeout"] + 1
                    case Color.YELLOW:
                        wins["yellow_timeout"] = wins["yellow_timeout"] + 1

        def get_length(entry: str) -> int:
            return wins[entry]

        print(
            f"Red won {get_length('red')} games, while yellow won {get_length('yellow')} games (draws: {get_length('draw')}, timeouts: {get_length('red_timeout')} for red, {get_length('yellow_timeout')} for yellow)."
        )


if __name__ == "__main__":
    from player_team_faskoe import PlayerFaSKoe

    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        Simulation.single(
            HumanPlayer(Color.RED, False), HumanPlayer(Color.YELLOW, False)
        )
        # Simulation.single(HumanPlayer(Color.RED, False), MonkeyPlayer(Color.YELLOW))
        # Simulation.single(PlayerFaSKoe(Color.YELLOW), MonkeyPlayer(Color.RED))
        # Simulation.many(PlayerFaSKoe(Color.RED), MonkeyPlayer(Color.YELLOW), 100)
    )
