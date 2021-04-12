import numpy as np



def board(row = 8, column = 8):
    board_list = [['+']*row for _ in range(column)]
    board_array = np.array(board_list, ndmin = 2)
    #board_array = np.zeros((row, column))
    return board_list


def move(row_number, column_number, board_list, player):
    if player == 1:
        value = 1
    else:
        value = -1
    board_list[row_number][column_number] = value
    new_board_array = np.array(board_list, ndmin = 2)
    return new_board_array


def print_board(board_list):
    new_board_array = np.array(board_list, ndmin=2)
    print(new_board_array)

def starting_moves(board_list):
    move(3, 3, new_board_list, 1)
    move(3, 4, new_board_list, 2)
    move(4, 3, new_board_list, 2)
    move(4, 4, new_board_list, 1)

def in_board_boundaries(x_cor, x_direction, y_cor, y_direction):
    if x_cor + x_direction < 8 and x_cor + x_direction > -1 and y_cor + y_direction < 8 and y_cor + y_direction > -1:
        return True
    else:
        return False

def valid_move(board_list, player, row_number, column_number):
    number_to_flip_list = []

    if board_list[row_number][column_number] != '+':
        print('Znajduje sie tutaj figura')
        return False
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
            move(row_number, column_number, board_list, player)
            print(x_cor, y_cor)


    #while(board_list[x_cor][y_cor]==seek_piece):



new_board_list = board()
print()
starting_moves(new_board_list)
print()
valid_move(new_board_list, 1, 2,4)
valid_move(new_board_list, 1, 3,5)
valid_move(new_board_list, 1, 4,2)
valid_move(new_board_list, 1, 5,3)
print_board(new_board_list)

