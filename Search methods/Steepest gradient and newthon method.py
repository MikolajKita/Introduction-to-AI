import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def function(x, y):
    z = np.array([(x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2], dtype='int64')
    if sys.maxsize <= z:
        z = np.array([sys.maxsize])

    return z


def gradient(x, y):
    return np.array(
        [[4 * x ** 3 + 4 * x * y - 42 * x + 2 * y ** 2 - 14], [4 * y ** 3 + 2 * x ** 2 + 4 * x * y - 26 * y - 22]])


def hessian(x, y):
    hessian = np.array([[12 * x ** 2 + 4 * y - 42, 4 * x + 4 * y], [4 * x + 4 * y, 12 * y ** 2 + 4 * x - 26]])
    return hessian


def inv_hessian(x, y):
    hessian = np.array([[12 * x ** 2 + 4 * y - 42, 4 * x + 4 * y], [4 * x + 4 * y, 12 * y ** 2 + 4 * x - 26]])
    return np.linalg.inv(hessian)


def steepest_gradient_desc(function, gradient, start_point, learning_rate=0.01, steps=10 ** 3, stop_value=10 ** (-12)):
    i = 0
    overflow_flag = 1
    x = start_point[0, 0]
    y = start_point[1, 0]
    x_coordinates = []
    y_coordinates = []
    value_array = []
    x0 = x
    y0 = y
    min_value = function(x, y)

    while i < steps and min_value > stop_value and overflow_flag:
        x = start_point[0, 0]
        y = start_point[1, 0]
        curr_value = function(x, y)
        x_coordinates.append(x)
        y_coordinates.append(y)

        if curr_value == sys.maxsize:
            overflow_flag = 0

        value_array.append(curr_value)
        start_point = start_point - learning_rate * gradient(x, y)

        if curr_value < min_value:
            min_value = curr_value
            x0 = x
            y0 = y

        i = i + 1

    return i, min_value, x_coordinates, y_coordinates, value_array, x0, y0


def newthon_method(function, gradient, inv_hessian, start_point, learning_rate=0.05, steps=10 ** 3,
                   stop_value=10 ** (-12)):
    i = 0
    overflow_flag = 1
    x = start_point[0, 0]
    y = start_point[1, 0]
    x_coordinates = []
    y_coordinates = []
    value_array = []
    x0 = x
    y0 = y
    min_value = function(x, y)
    if (min_value < 10 ** (-12)):
        value_array.append(min_value)
        x_coordinates.append(x)
        y_coordinates.append(y)

    while i < steps and min_value > stop_value and overflow_flag:
        x = start_point[0, 0]
        y = start_point[1, 0]
        x_coordinates.append(x)
        y_coordinates.append(y)
        curr_value = function(x, y)

        if curr_value == sys.maxsize:
            overflow_flag = 0
        value_array.append(curr_value)
        start_point = start_point - learning_rate * inv_hessian(x, y) * gradient(x, y)

        if curr_value < min_value:
            min_value = curr_value
            x0 = x
            y0 = y

        i = i + 1

    return i, min_value, x_coordinates, y_coordinates, value_array, x0, y0


###steepest_gradient_desc(function, gradient, np.array([[-5], [3]]), 0.01, 10 ** 4)
###newthon_method(function, gradient, inv_hessian, np.array([[-5], [3]]), 0.01, 10 ** 4)


def plot_surface_and_path_both_methods(start_point, random_learning_rate=np.random.uniform(0.05, 0.01),
                                       random_max_steps=np.random.randint(10**2, 10**4)):
    result_grad = steepest_gradient_desc(function, gradient, start_point, random_learning_rate, random_max_steps)
    result_newthon = newthon_method(function, gradient, inv_hessian, start_point, random_learning_rate,
                                    random_max_steps)
    X, Y = np.meshgrid(result_grad[2], result_grad[3])
    X_val = np.array(result_grad[2])
    Y_val = np.array(result_grad[3])
    Z = np.array(result_grad[4])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel("Współrzędna  X")
    ax.set_ylabel("Współrzędna  Y")
    ax.set_zlabel("Wartość funkcji w punkcie [X,Y]")
    plt.plot([start_point[0, 0]], [start_point[1, 0]], [function(start_point[0, 0], start_point[1, 0])[0]],
             markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10)
    plt.plot([result_grad[5]], [result_grad[6]], [result_grad[1][0]], markerfacecolor='m', markeredgecolor='m',
             marker='p', markersize=8)
    plt.title(
        "Wykres powierzchni metodą SGD dla\n punktu startowego [{} , {}] do punktu [{} , {}] przy \n współczynniku beta {} i maksymalnej liczbie kroków = {} \n Wartość funkcji w tym punkcie to: {} po {} krokach"
        .format(start_point[0, 0], start_point[1, 0], np.around(result_grad[5], 2), np.around(result_grad[6], 2),
                np.around(random_learning_rate, 3), np.around(random_max_steps), np.around(result_grad[1][0], 2),
                np.around(result_grad[0]), fontsize=8))
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot(X_val, Y_val, Z.flatten())
    plt.plot([start_point[0, 0]], [start_point[1, 0]], [function(start_point[0, 0], start_point[1, 0])[0]],
             markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10)
    plt.plot([result_grad[5]], [result_grad[6]], [result_grad[1][0]], markerfacecolor='m', markeredgecolor='m',
             marker='p', markersize=8)
    ax.set_xlabel("Współrzędna  X")
    ax.set_ylabel("Współrzędna  Y")
    ax.set_zlabel("Wartość funkcji w punkcie [X,Y]")
    plt.title(
        "Wykres drogi metodą SGD dla\n punktu startowego [{} , {}] do punktu [{} , {}] przy \n współczynniku beta {} i maksymalnej liczbie kroków = {} \n Wartość funkcji w tym punkcie to: {} po {} krokach"
            .format(start_point[0, 0], start_point[1, 0], np.around(result_grad[5], 2), np.around(result_grad[6], 2),
                    np.around(random_learning_rate, 3), np.around(random_max_steps), np.around(result_grad[1][0], 2),
                    np.around(result_grad[0]), fontsize=8))
    plt.show()

    X, Y = np.meshgrid(result_newthon[2], result_newthon[3])
    X_val = np.array(result_newthon[2])
    Y_val = np.array(result_newthon[3])
    Z = np.array(result_newthon[4])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel("Współrzędna  X")
    ax.set_ylabel("Współrzędna  Y")
    ax.set_zlabel("Wartość funkcji w punkcie [X,Y]")
    plt.plot([start_point[0, 0]], [start_point[1, 0]], [function(start_point[0, 0], start_point[1, 0])[0]],
             markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10)
    plt.plot([result_newthon[5]], [result_newthon[6]], [result_newthon[1][0]], markerfacecolor='m', markeredgecolor='m',
             marker='p', markersize=8)
    plt.title(
        "Wykres powierzchni metodą Newthona dla\n punktu startowego [{} , {}] do punktu [{} , {}] przy \n współczynniku beta {} i maksymalnej liczbie kroków = {} \n Wartość funkcji w tym punkcie to: {} po {} krokach"
            .format(start_point[0, 0], start_point[1, 0], np.around(result_newthon[5], 2),
                    np.around(result_newthon[6], 2),
                    np.around(random_learning_rate, 3), np.around(random_max_steps), np.around(result_newthon[1][0], 2),
                    np.around(result_newthon[0]), fontsize=8))
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot(X_val, Y_val, Z.flatten())
    plt.plot([start_point[0, 0]], [start_point[1, 0]], [function(start_point[0, 0], start_point[1, 0])[0]],
             markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10)
    plt.plot([result_newthon[5]], [result_newthon[6]], [result_newthon[1][0]], markerfacecolor='m', markeredgecolor='m',
             marker='p', markersize=8)
    ax.set_xlabel("Współrzędna  X")
    ax.set_ylabel("Współrzędna  Y")
    ax.set_zlabel("Wartość funkcji w punkcie [X,Y]")
    plt.title(
        "Wykres drogi metodą Newthona dla\n punktu startowego [{} , {}] do punktu [{} , {}] przy \n współczynniku beta {} i maksymalnej liczbie kroków = {} \n Wartość funkcji w tym punkcie to: {} po {} krokach"
            .format(start_point[0, 0], start_point[1, 0], np.around(result_newthon[5], 2),
                    np.around(result_newthon[6], 2),
                    np.around(random_learning_rate, 3), np.around(random_max_steps), np.around(result_newthon[1][0], 2),
                    np.around(result_newthon[0]), fontsize=8))
    plt.show()


#plot_surface_and_path_both_methods(start_point=np.array([[5], [2.4]]))
plot_surface_and_path_both_methods(start_point=np.array([[5], [3]]), random_learning_rate=np.around(0.033,4))
#plot_surface_and_path_both_methods(start_point=np.array([[-5], [5]]))
newthon_method(function, gradient, inv_hessian, np.array([[5], [3]]), np.around(0.033), 10)