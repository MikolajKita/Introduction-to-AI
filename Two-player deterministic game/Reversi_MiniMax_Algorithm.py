import numpy as np
import random


def board(row = 8, column = 8):
    board_list = [['+']*row for _ in range(column)]
    board_array = np.array(board_list, ndmin = 2)
    #board_array = np.zeros((row, column))
    return board_list


def in_board_boundaries(x_cor, x_direction, y_cor, y_direction):
    if x_cor + x_direction < 8 and x_cor + x_direction > -1 and y_cor + y_direction < 8 and y_cor + y_direction > -1:
        return True
    else:
        return False

def valid_move(board_list, player, row_number, column_number):
    number_to_flip_list = []
    array_to_return = [0,0,0,0,0]

    if board_list[row_number][column_number] != '+':
        #print('Znajduje sie tutaj figura')
        return array_to_return
    if player == 1:
        seek_piece = -1
    else:
        seek_piece = 1
    for x_direction, y_direction in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:

        x_cor = row_number
        y_cor = column_number
        number_of_pieces = 0
        if(x_cor + x_direction < 8 and x_cor + x_direction > -1):
            x_cor = x_cor + x_direction
        if (y_cor + y_direction < 8 and y_cor + y_direction > -1):
            y_cor = y_cor + y_direction
        while board_list[x_cor][y_cor] == seek_piece and in_board_boundaries(x_cor, x_direction, y_cor, y_direction):
                x_cor = x_cor + x_direction
                y_cor = y_cor + y_direction
                number_of_pieces = number_of_pieces + 1
        if board_list[x_cor][y_cor] == seek_piece*(-1) and number_of_pieces > 0:
            array_to_return = [number_of_pieces, x_cor, y_cor, x_direction, y_direction]
            return array_to_return

    return array_to_return



def all_valid_moves(board_list, player, row=8, column=8):
    valid_move_table = [[0]*row for _ in range(column)]
    for i in range (0, row-1):
        for j in range(0, column-1):
            array = valid_move(board_list, player, i, j)
            if array[0]:
                valid_move_table[i][j] = array[0]


    board_array = np.array(valid_move_table, ndmin=2)
    #print(board_array)
    return board_array

def move(row_number, column_number, board_list, player):
    if player == 1:
        value = 1
    else:
        value = -1
    helparray = valid_move(board_list, player, row_number, column_number)
    x_end = helparray[1]
    y_end = helparray[2]
    x_direct = helparray[3]
    y_direct = helparray[4]
    #print(helparray)
    if (x_direct == 0 and y_direct == 0):
        new_board_array = np.array(board_list, ndmin=2)
        return new_board_array

    else:
        board_list[row_number][column_number] = value
        while not (row_number == x_end and y_end == column_number):
            row_number = row_number + x_direct
            column_number = column_number + y_direct
            #print(x_direct, y_direc)
            board_list[row_number][column_number] = value

    new_board_array = np.array(board_list, ndmin = 2)
    return new_board_array


def print_board(board_list):
    new_board_array = np.array(board_list, ndmin=2)
    print(new_board_array)
    print()


def starting_moves(board_list):
    board_list[3][3] = 1
    board_list[3][4] = -1
    board_list[4][3] = -1
    board_list[4][4] = 1

def random_move(available_moves, board_list, player):

    x = random.randint(0,7)
    y = random.randint(0, 7)
    while available_moves[x][y] == 0:
        x = random.randint(0, 7)
        y = random.randint(0, 7)

    move(x, y, board_list, player)

def score(board_list, row, column):
    player_one = 0
    player_two = 0
    for i in range (0, row-1):
        for j in range(0, column-1):
            if(board_list[i][j]!='+'):
                if(board_list[i][j]==1):
                    player_one = player_one + 1
                else:
                    player_two = player_two - 1

    print(player_one, player_two*(-1))


new_board_list = board()

starting_moves(new_board_list)
print_board(new_board_list)
print(all_valid_moves(new_board_list, 1))
i = 0
while i < 5:

    valid_moves = all_valid_moves(new_board_list,1)
    random_move(valid_moves, new_board_list,1)
    print_board(new_board_list)

    valid_moves = all_valid_moves(new_board_list, 2)
    random_move(valid_moves, new_board_list,2)
    print_board(new_board_list)
    i = i + 1

score(new_board_list, 8,8)