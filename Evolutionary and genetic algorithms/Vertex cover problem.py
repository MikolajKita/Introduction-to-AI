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

def list_to_evaluation_array(list, max_population):
    k = 0
    return_list = []
    for i in range (0,max_population):
        if i == list[k]:
            return_list.append(1)
            k = k + 1
            if k >= len(list):
                k = k - 1
        else:
            return_list.append(0)
        i = i + 1

    print(return_list)
    return return_list

def vertex_to_evaluation(vertex, max_population):

    return_list = []
    for i in range (0, max_population):
        if i == vertex:
            return_list.append(1)
        else:
            return_list.append(0)
    print(return_list)
    return return_list

def tournament (graph, starting_population, tournament_size, evaluation, max_population):

    tour_list = np.sort(np.random.choice(starting_population, tournament_size, replace=False))
    max_result = 0
    for i in tour_list:
        if max_result < evaluation(graph, vertex_to_evaluation(i, max_population)):
            max_result = evaluation(graph, vertex_to_evaluation(i, max_population))
            tournament_winner = i

    print(tournament_winner)
    return tournament_winner


def evolution_algorithm(graph, evaluation, starting_population, max_population, mutation_prob, max_iter, all_population, current_population):
    i = 0
    max_result = evaluation(graph, starting_population)

    while i < max_iter:
        #vertex_to_evaluation(tourney[0], max_population)
        #vertex_to_evaluation(tourney[1], max_population)
        tournament(graph, list_to_evaluation_array(current_population, max_population), 2, evaluation, max_population)
        print()
        tournament(graph, all_pop, 3, evaluation, max_population)



        curr_population_array = list_to_evaluation_array(current_population)


        if max_result < evaluation(graph, curr_population_array):
            max_result = evaluation(graph, curr_population_array)
        #list_to_evaluation_array(tourney, max_population)
        i = i + 1

    print (max_result)
    return max_result



#array = np.array([1,1,1,1,1])
#result(sym_graph(5), array)

#result(np.ones((5,5)), array)


max_population = 6
list = np.arange(0, max_population)
all_pop = np.arange(0,max_population)
curr_pop = [5,6,7]
list_two = np.array(curr_pop)
print(all_pop - list_two)
#list_to_evaluation_array(list, 6)


evolution_algorithm(sym_graph(6), evaluation_method, list, max_population, 8, 10, all_pop, curr_pop)
