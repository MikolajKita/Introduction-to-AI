import random
import numpy as np


def sym_graph(number_of_vertices):
    graph = np.random.randint(1,3,size=(number_of_vertices,number_of_vertices))
    graph_symm = (graph + graph.T)%2
    return graph_symm

def mark_result(graph_symm, points):

    result = graph_symm*points
    result_two = (result | result.T)
    result_three = (result | result.T)
    print(graph_symm)
    print()
    print(result_two)
    print(np.sum(result_two)/2)


#result(sym_graph(5), )


#array = np.array([1,1,1,1,1])
#result(sym_graph(5), array)
graph = sym_graph(6)
array = np.array([1,0,1,0,1,1])
mark_result(graph, array)
array = np.array([0,1,1,1,0,1])
mark_result(graph, array)
#result(np.ones((5,5)), array)