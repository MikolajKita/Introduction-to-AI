import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
import datetime

def sym_graph(number_of_vertices):
    graph = np.random.randint(0, 2, size=(number_of_vertices, number_of_vertices))
    graph_symm = (graph + graph.T) % 2
    return graph_symm


def evaluation_method(graph_symm, points):
    chosen_vertex_graph = graph_symm * points
    chosen_vertex_graph_symm = (chosen_vertex_graph | chosen_vertex_graph.T)
    # binary OR allows me to capture all edges, makes it easy to just divide by two and get result
    result = np.sum(chosen_vertex_graph_symm / 2)
    return result

def policemen_graph(graph_symm, points):
    chosen_vertex_graph = graph_symm * points
    chosen_vertex_graph_symm = (chosen_vertex_graph | chosen_vertex_graph.T)
    return chosen_vertex_graph_symm

def list_to_evaluation_array(list, max_population):
    k = 0
    return_list = []
    for i in range(0, max_population):
        if i == list[k]:
            return_list.append(1)
            k = k + 1
            if k >= len(list):
                k = k - 1
        else:
            return_list.append(0)
        i = i + 1

    # print(return_list)
    return return_list


def vertex_to_evaluation(vertex, max_population):
    return_list = []
    for i in range(0, max_population):
        if i == vertex:
            return_list.append(1)
        else:
            return_list.append(0)

    return return_list


def array_to_list(array):
    x = 0
    r_list = []
    for i in array:
        if (array[x] == 1):
            r_list.append(x)
        x = x + 1
        i = i + 1

    return r_list


def tournament(graph, population, tournament_size, evaluation, max_population):
    tour_list = np.sort(np.random.choice(population, tournament_size, replace=False))
    max_result = 0
    tournament_winner = tour_list[0]
    for i in tour_list:

        if max_result < evaluation(graph, vertex_to_evaluation(i, max_population)):
            max_result = evaluation(graph, vertex_to_evaluation(i, max_population))
            tournament_winner = i

    return tournament_winner


def losing_tournament(graph, population, tournament_size, evaluation, max_population):
    tour_list = np.sort(np.random.choice(population, tournament_size, replace=False))
    min_result = 0
    tournament_loser = tour_list[0]
    for i in tour_list:

        if min_result > evaluation(graph, vertex_to_evaluation(i, max_population)):
            min_result = evaluation(graph, vertex_to_evaluation(i, max_population))
            tournament_loser = i

    return tournament_loser


def tournament_selection(graph, current_population, tournament_size, evaluation, max_population, all_population, curr_population_array):
    removed_one = losing_tournament(graph, current_population, tournament_size, evaluation, max_population)
    chosen_one = tournament(graph, array_to_list(all_population - curr_population_array), tournament_size, evaluation, max_population)
    current_population.remove(removed_one)
    current_population.append(chosen_one)
    current_population.sort()
    return current_population


def search_element(mutated_element, current_population):
    for i in current_population:
        if i == mutated_element:
            return True
    return False


def mutation(current_population, probability, max_population):
    mutated_element = random.choice(current_population)
    x = random.random()
    if x < probability:
        current_population.remove(mutated_element)
        mutated_element = mutated_element + random.randint(-2, 2)
        while (search_element(mutated_element, current_population)) or mutated_element >= max_population:
            mutated_element = mutated_element + random.randint(-2,2)

        current_population.append(mutated_element)


def evolution_algorithm(graph, evaluation, starting_population, max_population, mutation_prob, max_iter, all_population, size):
    start = datetime.datetime.now()
    i = 0
    curr_population_array = list_to_evaluation_array(starting_population, max_population)
    max_result = evaluation(graph, curr_population_array)
    start_result = max_result
    best_vertex_placement = curr_population_array
    current_population = copy.deepcopy(starting_population)
    while i < max_iter:

        tournament_selection(graph, current_population, size, evaluation, max_population, all_population, curr_population_array)
        mutation(current_population, mutation_prob, max_population)

        curr_population_array = list_to_evaluation_array(current_population, max_population)

        if max_result < evaluation(graph, curr_population_array):
            max_result = evaluation(graph, curr_population_array)
            best_vertex_placement= curr_population_array
        i = i + 1

    best_vertex_placement_graph = policemen_graph(graph, best_vertex_placement)
    end = datetime.datetime.now()
    time = (end - start)
    print(max_result, start_result, time)
    return max_result, best_vertex_placement_graph, best_vertex_placement


def plot_graph(chosen_graph, evaluation_method, population_start, population, mutation_probability, max_iter, all_pop, tournament_size):
    array = evolution_algorithm(chosen_graph, evaluation_method, population_start, population, mutation_probability,
                                100, all_pop, tournament_size)
    optimal_policemen_position = array[2]
    score = array[0]
    color_map = []
    for node in (optimal_policemen_position):
        if node == 1:
            color_map.append('red')
        else:
            color_map.append('green')
    G = nx.from_numpy_array(array[1])
    nx.draw(G, pos=nx.spring_layout(G), node_color=color_map, with_labels=True)
    plt.title(
        'Graf o {} wierzchołkach przy maksymalnej liczbie policjantów {} \n z policjantami w wierzcholkach {} pokrywającymi {} ulic'.
        format(population, len(population_start), array_to_list(optimal_policemen_position), int(score)))
    plt.savefig(
        'Graf o {} wierzchołkach przy maksymalnej liczbie policjantów {} z policjantami w wierzcholkach {} pokrywającymi {} ulic.png'.
        format(population, len(population_start), array_to_list(optimal_policemen_position), int(score)),
        bbox_inches='tight')


population = 5
all_pop = np.ones(population, dtype=int)
number_of_policemen = 3
tournament_size = 2
np.random.seed(2021)
population_start = np.sort(np.random.choice(array_to_list(all_pop), number_of_policemen, replace=False))
population_start = population_start.tolist()
max_iter = 100
mutation_probability = 0.5


chosen_graph = np.array([[0, 1, 1, 1, 1],
[1, 0, 1, 1, 1],
[1, 1, 0, 1, 1],
[1, 1, 1, 0, 1],
[1, 1, 1, 1, 0]])

plot_graph(chosen_graph, evaluation_method, population_start, population, mutation_probability, max_iter, all_pop, tournament_size)







#.append(evolution_algorithm(chosen_graph, evaluation_method, population_start, population, 0.8, 10, all_pop, 2))
#tab.append(evolution_algorithm(chosen_graph, evaluation_method, population_start, population, 0.8, 100, all_pop, 2))
#tab.append(evolution_algorithm(chosen_graph, evaluation_method, population_start, population, 0.8, 1000, all_pop, 2))

#print(tab)
