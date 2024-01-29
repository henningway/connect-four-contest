import random
import time

import numpy as np
import math
import sys
import copy

# Constants
AI_PIECE = 1
P_PIECE = -1
EMPTY_PIECE = 0
ROWS = 6
COLUMNS = 7
LENGTH = 4


def player_move(board):

    action = int(input("Enter your move!\n"))
    while not valid_action(board, action):
        action = int(input("Your move was not valid, please enter a valid move!\n"))
    return action


def valid_action(board, action):
    return 0 <= action <= COLUMNS and board[0][action] == 0


def evaluate_interval(interval):

    score = 0

    if interval.count(AI_PIECE) == 3 and interval.count(EMPTY_PIECE) == 1:
        score += 5

    if interval.count(AI_PIECE) == 2 and interval.count(EMPTY_PIECE) == 2:
        score += 2

    if interval.count(P_PIECE) == 3 and interval.count(EMPTY_PIECE) == 0:
        score -= 4

    return score


def evaluate(board):

    score = 0

    # Central Nodes
    center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
    center_count = center_array.count(AI_PIECE)
    score += center_count * 3

    # Horizontal Nodes
    for r in range (ROWS):
        row = [int(i) for i in list(board[r, :])]
        for c in range(COLUMNS - 3):
            interval = row[c: c + LENGTH]
            score += evaluate_interval(interval)

    # Vertical Nodes
    for c in range (COLUMNS):
        column = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            interval = column[r: r + LENGTH]
            score += evaluate_interval(interval)

    # Positive Diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            interval = [board[r + i][c + i] for i in range(LENGTH)]
            score += evaluate_interval(interval)

    # Negative Diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            interval = [board[r + 3 - i][c + i] for i in range(LENGTH)]
            score += evaluate_interval(interval)

    return score


def winning_move(board, piece):

    # Horizontal Nodes
    for r in range(ROWS):
        row = [int(i) for i in list(board[r, :])]
        for c in range(COLUMNS - 3):
            interval = row[c: c + LENGTH]
            if interval.count(piece) == 4:
                return True

    # Vertical Nodes
    for c in range(COLUMNS):
        column = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            interval = column[r: r + LENGTH]
            if interval.count(piece) == 4:
                return True

    # Positive Diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            interval = [board[r + i][c + i] for i in range(LENGTH)]
            if interval.count(piece) == 4:
                return True

    # Negative Diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            interval = [board[r + 3 - i][c + i] for i in range(LENGTH)]
            if interval.count(piece) == 4:
                return True


def terminal_state(board, valid_moves):
    return winning_move(board, AI_PIECE) or winning_move(board, P_PIECE) or len(valid_moves) == 0


def available_moves(board):
    return set(i for i in range(COLUMNS) if board[0][i] == 0)


def drop_piece(board, action, player):
    for i in list(reversed(range(ROWS))):
        if np.any(board[i][action] == 0):
            board[i][action] = player
            return


def minimax(board, depth, alpha, beta, maximize: bool) -> tuple[int | None, int]:
    """ Returns columns, value """

    valid_moves = available_moves(board)
    terminal = terminal_state(board, valid_moves)

    if depth == 0 or terminal:

        if terminal:
            if winning_move(board, AI_PIECE):
                return None, 10000000000000
            elif winning_move(board, P_PIECE):
                return None, (-1000000000000)
        else:
            return None, evaluate(board)

    if maximize:

        value = -math.inf
        column = random.choice(list(valid_moves))

        for action in valid_moves:
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, action, AI_PIECE)
            score = minimax(b_copy, depth - 1, alpha, beta, False)[1]

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
            drop_piece(b_copy, action, P_PIECE)
            score = minimax(b_copy, depth - 1, alpha, beta, True)[1]

            if score < value:
                value = score
                column = action

            beta = min(beta, value)

            if alpha >= beta:
                break

        return column, value


def get_result(board, valid_moves):
    if winning_move(board, P_PIECE):
        return 1
    elif winning_move(board, AI_PIECE):
        return -1
    elif len(valid_moves) == 0:
        return 0


def play_game():

    board = np.zeros((6, 7), dtype=int)
    player = random.choice([True, False])

    if player:
        print('You start!')
        print()
    else:
        print('AI starts!')
        print()

    done = False
    while not done:

        valid_moves = available_moves(board)
        terminal = terminal_state(board, valid_moves)

        if terminal:
            return get_result(board, valid_moves)

        if player:
            action = player_move(board)
            drop_piece(board, action, P_PIECE)
            print("Your Move:\n", board)
            print()
            player = False
        else:
            depth = 5
            start = time.time()
            action, value = minimax(board, depth, -math.inf, math.inf, True)
            diff = time.time() - start
            print(f'AI turn with depth {depth} took {diff} seconds.')
            drop_piece(board, action, AI_PIECE)
            print("AI Move:\n", board)
            print()
            player = True


def main():
    result = play_game()
    print(result)


if __name__ == "__main__":
    main()
