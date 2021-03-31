import random
import numpy as np


def sym_graph(number_of_vertices):
    graph = np.random.randint(1,3,size=(number_of_vertices,number_of_vertices))
    graph_symm = (graph + graph.T)%2
    return graph_symm


def evaluation_method(graph_symm, points):

    chosen_vertex_graph = graph_symm*points
    chosen_vertex_graph_symm = (chosen_vertex_graph | chosen_vertex_graph.T) ## OR allows me to capture all edges, makes it easy to just divide by two and get result
    result = np.sum(chosen_vertex_graph_symm/2)
    return result


#result(sym_graph(5), )


#array = np.array([1,1,1,1,1])
#result(sym_graph(5), array)
graph = sym_graph(6)
array = np.array([1,0,1,0,1,1])
evaluation_method(graph, array)
#result(np.ones((5,5)), array)