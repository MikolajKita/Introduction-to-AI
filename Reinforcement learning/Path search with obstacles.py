import numpy as np
import pandas as pd
import random
import networkx as nx
import pprint
import matplotlib.pyplot as plt
from leafy.graph import Graph

random.seed(21)


def generate_board(starting_vertex, ending_vertex, side_length=8, obstacle_probability=0.2):

    obstacle = -1
    clear_path = 1
    path_available = 0
    while path_available == 0:

        x = 0
        #random.seed(side_length + x * side_length)

        board_list = [[0 for i in range(side_length)] for j in range(side_length)]
        board_graph_help = [[0 for i in range(side_length)] for j in range(side_length)]
        for i in range(0, side_length):
            for j in range(0, side_length):
                board_list[i][j] = random.choices((obstacle, clear_path),
                                                  weights=[obstacle_probability, 1 - obstacle_probability])
                board_graph_help[i][j] = x
                x = x + 1

        board = np.array(board_list).reshape(side_length, side_length)
        graph_list = [[0 for i in range(side_length * side_length)] for j in range(side_length * side_length)]
        for i in range(0, side_length):
            for j in range(0, side_length):
                number = board_graph_help[i][j]

                if board[i][j] == 1:
                    graph_list[number][number] = 1
                    if 0 <= i - 1 < side_length:
                        if board[i - 1][j] == 1:
                            graph_list[board_graph_help[i][j]][board_graph_help[i - 1][j]] = 1
                            graph_list[board_graph_help[i - 1][j]][board_graph_help[i][j]] = 1

                    if 0 <= i + 1 < side_length:
                        if board[i + 1][j] == 1:
                            graph_list[board_graph_help[i][j]][board_graph_help[i + 1][j]] = 1
                            graph_list[board_graph_help[i + 1][j]][board_graph_help[i][j]] = 1

                    if 0 <= j - 1 < side_length:
                        if board[i][j - 1] == 1:
                            graph_list[board_graph_help[i][j]][board_graph_help[i][j - 1]] = 1
                            graph_list[board_graph_help[i][j - 1]][board_graph_help[i][j]] = 1

                    if 0 <= j + 1 < side_length:
                        if board[i][j + 1] == 1:
                            graph_list[board_graph_help[i][j]][board_graph_help[i][j + 1]] = 1
                            graph_list[board_graph_help[i][j + 1]][board_graph_help[i][j]] = 1

        graph = np.array(graph_list).reshape(side_length * side_length, side_length * side_length)
        board_graph = nx.from_numpy_array(graph)
        for path in nx.all_simple_paths(board_graph, source=starting_vertex, target=ending_vertex):
            print("path")
            if path:
                path_available = 1
                break


    color_map = []
    for node in board_graph:
        if graph[node][node] == 0:
            color_map.append('red')
        elif node == starting_vertex:
            color_map.append('blue')
        elif node == ending_vertex:
            color_map.append('purple')
        else:
            color_map.append('green')

    nx.draw(board_graph, pos=nx.spring_layout(board_graph), node_color=color_map, with_labels=True)

    plt.savefig(
        'Graf planszy',
        bbox_inches='tight')
    return board


a = generate_board(0, 63, 8, 0.3)

