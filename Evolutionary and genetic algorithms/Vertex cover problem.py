import random
import numpy as np


def sym_graph(number_of_vertices):
    graph = np.random.randint(0,2,size=(number_of_vertices,number_of_vertices))
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

    #print(return_list)
    return return_list

def vertex_to_evaluation(vertex, max_population):

    return_list = []
    for i in range (0, max_population):
        if i == vertex:
            return_list.append(1)
        else:
            return_list.append(0)

    return return_list

def array_to_list(array):
    x = 0
    r_list = []
    for i in array:
        if(array[x]==1):
            r_list.append(x)
        x = x + 1
        i = i + 1

    return r_list

def tournament (graph, starting_population, tournament_size, evaluation, max_population):

    tour_list = np.sort(np.random.choice(starting_population, tournament_size, replace=False))
    max_result = 0
    for i in tour_list:

        if max_result < evaluation(graph, vertex_to_evaluation(i, max_population)):
            max_result = evaluation(graph, vertex_to_evaluation(i, max_population))
            tournament_winner = i

    return tournament_winner


def evolution_algorithm(graph, evaluation, starting_population, max_population, mutation_prob, max_iter, all_population, current_population):
    i = 0

    curr_population_array = list_to_evaluation_array(current_population, max_population)
    max_result = evaluation(graph, curr_population_array)
    start_result = max_result
    print(start_result)
    print()
    while i < max_iter:
        #vertex_to_evaluation(tourney[0], max_population)
        #vertex_to_evaluation(tourney[1], max_population)
        #tournament(graph, current_population, 3, evaluation, max_population)
        print(current_population)
        removed_one = tournament(graph, current_population, 3, evaluation, max_population)
        chosen_one = tournament(graph, array_to_list(all_population-curr_population_array), 4, evaluation, max_population)
        current_population.remove(removed_one)
        current_population.append(chosen_one)
        current_population.sort()
        print(current_population)
        curr_population_array = list_to_evaluation_array(current_population, max_population)
        print(evaluation(graph, curr_population_array))

        if max_result < evaluation(graph, curr_population_array):
            max_result = evaluation(graph, curr_population_array)
        #list_to_evaluation_array(tourney, max_population)
        i = i + 1

    print (max_result, start_result)
    return max_result

max_population = 25


list = np.random.randint(0,2,size=(max_population))
all_pop = np.ones(max_population)
curr_pop = [2,4,6,10,12]
graph_mine = np.array([[0,1,0,0,1,0,0,0,1,0]
,[1,0,0,1,0,1,0,1,0,1]
,[0,0,0,0,1,1,0,1,0,1]
,[0,1,0,0,0,0,1,1,1,1]
,[1,0,1,0,0,0,1,0,1,0]
,[0,1,1,0,0,0,0,1,0,0]
,[0,0,0,1,1,0,0,0,0,0]
,[0,1,1,1,0,1,0,0,0,0]
,[1,0,0,1,1,0,0,0,0,0]
,[0,1,1,1,0,0,0,0,0,0]])
print(graph_mine)
tab = []
chosen_graph = sym_graph(max_population)
full_graph = np.ones((max_population,max_population))
print(full_graph)



#tab.append(evolution_algorithm(chosen_graph, evaluation_method, list, max_population, 8, 10, all_pop, curr_pop))
#tab.append(evolution_algorithm(chosen_graph, evaluation_method, list, max_population, 8, 100, all_pop, curr_pop))
#tab.append(evolution_algorithm(chosen_graph, evaluation_method, list, max_population, 8, 1000, all_pop, curr_pop))

print(tab)

