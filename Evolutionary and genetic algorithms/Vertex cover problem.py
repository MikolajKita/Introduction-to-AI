# Mikołaj Kita
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import statistics


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


def tournament_selection(graph, current_population, tournament_size, evaluation, max_population, all_population,
                         curr_population_array):
    removed_one = losing_tournament(graph, current_population, tournament_size, evaluation, max_population)
    chosen_one = tournament(graph, array_to_list(all_population - curr_population_array), tournament_size, evaluation,
                            max_population)
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
    iter = 0
    x = copy.deepcopy(mutated_element)
    if random.random() < probability:
        current_population.remove(mutated_element)
        mutated_element = mutated_element + random.randint(-2, 2)
        while (search_element(mutated_element, current_population)) or mutated_element >= max_population:
            mutated_element = mutated_element + random.randint(-2, 2)
            iter = iter + 1
            if iter > 100:
                mutated_element = mutated_element % max_population + random.randint(0, iter) % max_population
                if mutated_element < 0:
                    print("TUTAJ")

        current_population.append(mutated_element)


def evolution_algorithm(graph, evaluation, starting_population, max_population, mutation_prob, max_iter, all_population,
                        size):
    start = datetime.datetime.now()
    i = 0
    curr_population_array = list_to_evaluation_array(starting_population, max_population)
    max_result = evaluation(graph, curr_population_array)
    start_result = max_result
    best_vertex_placement = curr_population_array
    current_population = copy.deepcopy(starting_population)
    while i < max_iter:

        tournament_selection(graph, current_population, size, evaluation, max_population, all_population,
                             curr_population_array)
        mutation(current_population, mutation_prob, max_population)

        curr_population_array = list_to_evaluation_array(current_population, max_population)

        if max_result < evaluation(graph, curr_population_array):
            max_result = evaluation(graph, curr_population_array)
            best_vertex_placement = curr_population_array
        i = i + 1

    best_vertex_placement_graph = policemen_graph(graph, best_vertex_placement)
    end = datetime.datetime.now()
    time = (end - start)
    print(max_result, start_result, time, array_to_list(best_vertex_placement))
    return max_result, best_vertex_placement_graph, best_vertex_placement, start_result, time


def plot_graph(graph, evaluation, starting_population, population_size, mutation_prob, max_steps, all_population,
               tournament_size):
    array = evolution_algorithm(graph, evaluation, starting_population, population_size, mutation_prob,
                                max_steps, all_population, tournament_size)
    starting_score = evaluation(graph, list_to_evaluation_array(starting_population, population_size))
    G = nx.from_numpy_array(graph)
    color_map = []
    for node in list_to_evaluation_array(starting_population, population_size):
        if node == 1:
            color_map.append('red')
        else:
            color_map.append('green')
    nx.draw(G, pos=nx.spring_layout(G), node_color=color_map, with_labels=True)
    plt.savefig(
        'Wylosowany graf o {} wierzchołkach przy maksymalnej liczbie policjantów {} z policjantami w wierzcholkach '
        'poczatkowych {} z poczatkowym wynikiem {}.png'.format(population_size, len(starting_population), starting_population, starting_score),
        bbox_inches='tight')
    G.clear()
    plt.clf()
    optimal_policemen_position = array[2]
    score = array[0]
    color_map.clear()
    for node in optimal_policemen_position:
        if node == 1:
            color_map.append('red')
        else:
            color_map.append('green')
    G = nx.from_numpy_array(array[1])
    nx.draw(G, pos=nx.spring_layout(G), node_color=color_map, with_labels=True)
    plt.title(
        'Graf o {} wierzchołkach przy maksymalnej liczbie policjantów {} \n z policjantami w wierzcholkach {} pokrywającymi {} ulic'.
            format(population_size, len(starting_population), array_to_list(optimal_policemen_position), int(score)))
    plt.savefig(
        'Graf o {} wierzchołkach przy maksymalnej liczbie policjantów {} z policjantami w wierzcholkach {} pokrywającymi {} ulic.png'.
            format(population_size, len(starting_population), array_to_list(optimal_policemen_position), int(score)),
        bbox_inches='tight')
    G.clear()
    plt.clf()


population_parameter = 5
all_pop = np.ones(population_parameter, dtype=int)
number_of_policemen = 3
tournament_size_parameter = 2
np.random.seed(2021)
population_start = np.sort(np.random.choice(array_to_list(all_pop), number_of_policemen, replace=False))
population_start = population_start.tolist()
max_iter_parameter = 100
mutation_probability = 0.5

chosen_graph = np.array([[0, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1],
                         [1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 0]])

# plot_graph(chosen_graph, evaluation_method, population_start, population_parameter, mutation_probability, max_iter_parameter, all_pop, tournament_size_parameter) #graf pelny


np.random.seed(100)
chosen_graph = sym_graph(population_parameter)
# plot_graph(chosen_graph, evaluation_method, population_start, population_parameter, mutation_probability, max_iter_parameter, all_pop, tournament_size_parameter) #graf losowy


population_parameter = 6
all_pop = np.ones(population_parameter, dtype=int)
population_start = np.sort(np.random.choice(array_to_list(all_pop), number_of_policemen, replace=False))
population_start = population_start.tolist()

chosen_graph = np.array(
    [[0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1],
     [1, 1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0, 0]])
# plot_graph(chosen_graph, evaluation_method, population_start, population_parameter, mutation_probability, max_iter_parameter, all_pop, tournament_size_parameter) #graf dwudzielny


population_parameter = 25
all_pop = np.ones(population_parameter, dtype=int)
number_of_policemen = 7
tournament_size_parameter = 3
mutation_probability = 0.5
chosen_graph = sym_graph(population_parameter)
population_start = np.sort(np.random.choice(array_to_list(all_pop), number_of_policemen, replace=False))
population_start = population_start.tolist()

iter_10 = []
iter_100 = []
iter_1000 = []
max_range = 0
# max_range = 100
for j in range(0, max_range):
    iter_10.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 10,
                            all_pop, tournament_size_parameter)[0])
    iter_100.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 100,
                            all_pop, tournament_size_parameter)[0])
    iter_1000.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 500,
                            all_pop, tournament_size_parameter)[0])
    print(j)

# print(statistics.mean(iter_10), min(iter_10), max(iter_10), statistics.stdev(iter_10))
# print(statistics.mean(iter_100), min(iter_100), max(iter_100), statistics.stdev(iter_100))
# print(statistics.mean(iter_1000), min(iter_1000), max(iter_1000), statistics.stdev(iter_1000))

iter_10.clear()
iter_100.clear()
iter_1000.clear()
max_range = 0

for j in range(0, max_range):
    iter_10.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 250,
                            all_pop, 2)[0])
    iter_100.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 250,
                            all_pop, 3)[0])
    iter_1000.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter,
                            mutation_probability, 250,
                            all_pop, 5)[0])
    print(j)

# print(statistics.mean(iter_10), min(iter_10), max(iter_10), statistics.stdev(iter_10))
# print(statistics.mean(iter_100), min(iter_100), max(iter_100), statistics.stdev(iter_100))
# print(statistics.mean(iter_1000), min(iter_1000), max(iter_1000), statistics.stdev(iter_1000))

# max_range = 100
iter_10.clear()
iter_100.clear()
iter_1000.clear()
max_range = 0

for j in range(0, max_range):
    iter_10.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter, 0.05, 250, all_pop,
                            2)[0])
    iter_100.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter, 0.4, 250, all_pop,
                            2)[0])
    iter_1000.append(
        evolution_algorithm(chosen_graph, evaluation_method, population_start, population_parameter, 0.8, 250, all_pop,
                            2)[0])
    print(j)

# print(statistics.mean(iter_10), min(iter_10), max(iter_10), statistics.stdev(iter_10))
# print(statistics.mean(iter_100), min(iter_100), max(iter_100), statistics.stdev(iter_100))
# print(statistics.mean(iter_1000), min(iter_1000), max(iter_1000), statistics.stdev(iter_1000))

plot_graph(chosen_graph, evaluation_method, population_start, population_parameter, mutation_probability,
           max_iter_parameter, all_pop,
           tournament_size_parameter)  # graf dwudzielny
