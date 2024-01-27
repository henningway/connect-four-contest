from abc import ABC
from enum import Enum
from random import choice
from typing import Optional


class Color(Enum):
    RED = 1
    YELLOW = 2


def color_to_letter(color: Optional[Color]) -> str:
    if color is None:
        return "."
    if color is Color.RED:
        return "R"
    if color is Color.YELLOW:
        return "Y"


class Board:
    def __init__(self) -> None:
        self.columns = [[], [], [], [], [], [], []]

    def row(self, n: int) -> [Optional[Color]]:
        """Provides a list representation of the row at given index."""
        return [
            self.columns[col][n] if len(self.columns[col]) > n else None
            for col in range(7)
        ]

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

    def is_legal_move(self, column: int) -> bool:
        """Tells whether given column index has open gaps."""
        return column in self.legal_moves()

    def register_move(self, color: Color, column: int):
        """Puts a token of the given color into given column index."""
        assert self.is_legal_move(column), "Not a legal move."
        self.columns[column].append(color)

    def is_full(self) -> bool:
        """Tells whether all of the columns are filled."""
        return len(self.legal_moves()) == 0


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
        # TODO: check for winner
        return self.board.is_full()


if __name__ == "__main__":
    game = Game(MonkeyPlayer(Color.RED), MonkeyPlayer(Color.YELLOW))

    while not game.is_finished():
        game.step()
        game.board.print()
        print("\n")
