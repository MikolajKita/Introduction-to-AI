import random
import numpy as np


def sym_graph(number_of_vertices):
    graph = np.random.randint(1,3,size=(number_of_vertices,number_of_vertices))
    graph_symm = (graph + graph.T)%2
    return graph_symm

