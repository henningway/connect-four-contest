from abc import ABC
from enum import Enum
from functools import reduce
from itertools import groupby
from random import choice
from typing import Optional


class Color(Enum):
    RED = 1
    YELLOW = 2


class Dim(Enum):
    COL = 1
    ROW = 2


def color_to_letter(color: Optional[Color], empty_char=".") -> str:
    if color is None:
        return empty_char
    if color is Color.RED:
        return "R"
    if color is Color.YELLOW:
        return "Y"


def letter_to_color(letter: str) -> Color:
    if letter == " " or letter == ".":
        return None
    if letter == "R":
        return Color.RED
    if letter == "Y":
        return Color.YELLOW


def sequence_to_word(sequence: [Optional[Color]]) -> str:
    return "".join(map(lambda color: color_to_letter(color, " "), sequence))


def longest_repeat(word: [str]) -> [Optional[Color]]:
    "Provides the longest consecutive sequence of characters in given word."
    if len(word) == 0:
        return []

    return sorted([list(g) for _, g in groupby(word)], key=len)[-1]


def longest_color_repeat(sequence: [Optional[Color]]) -> [Color]:
    "Provides the longest consecuctive non-nil sequence of colors."
    words = sequence_to_word(sequence).split()

    if len(words) == 0:
        return []

    return max(map(longest_repeat, words))


class Board:
    def __init__(self) -> None:
        self.columns = [[], [], [], [], [], [], []]

    def rows(self) -> [[Optional[Color]]]:
        return [self.row(row) for row in range(6)]

    def row(self, row: int) -> [Optional[Color]]:
        """Provides a list representation of the row at given index."""
        return [
            self.columns[col][row] if len(self.columns[col]) > row else None
            for col in range(7)
        ]

    def col(self, col: int) -> [Optional[Color]]:
        """Provides a list representation of the column at given index."""
        return self.columns[col]

    def print(self):
        """Renders a representation of the board to stdout."""
        for i in range(5, -1, -1):
            row = list(map(color_to_letter, self.row(i)))
            print(row)

    def legal_moves(self) -> [int]:
        """Provides a list of columns with open gaps (i.e. columns that are not full)."""
        return list(
            map(
                lambda tuple: tuple[0],
                filter(lambda col: len(col[1]) < 6, enumerate(self.columns)),
            )
        )

    def is_legal_move(self, col: int) -> bool:
        """Tells whether given column index has open gaps."""
        return col in self.legal_moves()

    def register_move(self, color: Color, col: int):
        """Puts a token of the given color into given column index."""
        assert self.is_legal_move(col), "Not a legal move."
        self.columns[col].append(color)

    def is_full(self) -> bool:
        """Tells whether all of the columns are filled."""
        return len(self.legal_moves()) == 0

    def longest_sequence_specific(self, dim: Dim, index: int) -> Optional[dict]:
        """Provides the longest consecutive sequence of one color in given row or column."""
        sequence: [Optional[Color]] = (self.col if dim == Dim.COL else self.row)(index)

        long_sequence: [str] = longest_color_repeat(sequence)

        return (
            {
                "color": letter_to_color(long_sequence[0]),
                "length": len(long_sequence),
                "dim": dim,
            }
            if len(long_sequence) > 0
            else None
        )

    # TODO: CHECK DIAGONALS
    def longest_sequence(self) -> Optional[dict]:
        """Provides the longest sequence of one color in all rows and columns."""
        col_entries = list(
            map(
                lambda index: self.longest_sequence_specific(Dim.COL, index),
                range(0, 7),
            )
        )
        row_entries = list(
            map(
                lambda index: self.longest_sequence_specific(Dim.ROW, index),
                range(0, 6),
            )
        )

        all_entries = col_entries + row_entries

        max_entry = max(
            all_entries, key=(lambda entry: entry["length"] if entry is not None else 0)
        )

        if max_entry is None:
            return None

        max_entry["index"] = (
            col_entries if max_entry["dim"] == Dim.COL else row_entries
        ).index(max_entry)

        return max_entry


class Player(ABC):
    def __init__(self, color: Color) -> None:
        super().__init__()
        self.color = color

    def nextMove(self, board: Board) -> int:
        """Provides a column index to place a token of the player's color in."""
        pass


class MonkeyPlayer(Player):
    def nextMove(self, board: Board) -> int:
        return choice(board.legal_moves())


class Game:
    def __init__(self, p1: Player, p2: Player) -> None:
        self.board = Board()
        self.p1 = p1
        self.p2 = p2
        self.activePlayer = self.p1

    def step(self):
        """Prompts the active player to decide on its next move and switches the active player."""
        self.board.register_move(
            self.activePlayer.color, self.activePlayer.nextMove(self.board)
        )
        self.activePlayer = self.p1 if self.activePlayer is self.p2 else self.p2

    def is_finished(self) -> bool:
        """Tells whether the game has ended (board is full or one player has connected four)."""
        return self.board.is_full() or self.check_winner() is not None

    def check_winner(self) -> Optional[Color]:
        """Provides the winner of the game, if there is any."""
        longest_repeat = self.board.longest_sequence()
        if longest_repeat is None:
            return None
        return longest_repeat["color"] if longest_repeat["length"] >= 4 else None


if __name__ == "__main__":
    game = Game(MonkeyPlayer(Color.RED), MonkeyPlayer(Color.YELLOW))

    while not game.is_finished():
        game.step()

    game.board.print()
    print("\n")
    print("Winner:", game.check_winner())
