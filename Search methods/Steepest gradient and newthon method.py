import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    z = np.array([(x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2], dtype='int64')
    if sys.maxsize <= z:
        z = sys.maxsize
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


def steepest_gradient_desc(function, gradient, start_point, learning_rate=0.01, steps=10 ** 4):
    i = 0
    overflow_flag = 1
    x = start_point[0, 0]
    y = start_point[1, 0]
    x0 = x
    y0 = y
    min_value = function(x, y)

    while i < steps and min_value > 10 ** (-12) and overflow_flag:
        x = start_point[0, 0]
        y = start_point[1, 0]
        curr_value = function(x, y)

        if curr_value == sys.maxsize:
            overflow_flag = 0

        start_point = start_point - learning_rate * gradient(x, y)
        if curr_value < min_value:
            min_value = curr_value
            x0 = x
            y0 = y

        i = i + 1
    print(i)
    print(min_value)
    
    return i, min_value


def newthon_method(function, gradient, inv_hessian, start_point, learning_rate=0.05, steps=10 ** 4):
    i = 0
    x = start_point[0, 0]
    y = start_point[1, 0]
    x_coordinates = []
    y_coordinates = []
    value_array = []
    x0 = x
    y0 = y
    min_value = function(x, y)
    if(min_value < 10 ** (-12)):
        value_array.append(min_value)
        x_coordinates.append(x)
        y_coordinates.append(y)

    while i < steps and min_value > 10 ** (-12):
        x = start_point[0, 0]
        y = start_point[1, 0]
        x_coordinates.append(x)
        y_coordinates.append(y)
        curr_value = function(x, y)
        value_array.append(curr_value)
        start_point = start_point - learning_rate * inv_hessian(x, y) * gradient(x, y)
    
        if curr_value < min_value:
            min_value = curr_value
            x0 = x
            y0 = y

        i = i + 1
    print(i)
    print(min_value)
    
    return i, min_value, x_coordinates, y_coordinates, value_array, x0, y0


###steepest_gradient_desc(function, gradient, np.array([[-5], [3]]), 0.01, 10 ** 4)
###newthon_method(function, gradient, inv_hessian, np.array([[-5], [3]]), 0.01, 10 ** 4)


def plot_random_scatter(first_function, second_function, start_point, number_of_points = 100):
    step_list = []
    value_list = []
    step_list_newthon = []
    value_list_newthon = []
    number = 0
    random_learning_rate = np.random.uniform(10 ** (-2), 10 ** (-12))
    random_max_steps = np.random.uniform(10**2, 10**4)
    while number < number_of_points:
        result_grad = steepest_gradient_desc(function, gradient, start_point, random_learning_rate, random_max_steps)
        step_list.append(result_grad[0])
        value_list.append(result_grad[1])
        result_newthon = newthon_method(function, gradient, inv_hessian, start_point, random_learning_rate, random_max_steps)
        step_list_newthon.append(result_newthon[0])
        value_list_newthon.append(result_newthon[1])
        number = number + 1
    steps = np.array(step_list)
    value = np.array(value_list)
    steps_newthon = np.array(step_list_newthon)
    value_newthon = np.array(value_list_newthon)
    plt.scatter(steps, value, label = "Gradient", c = "red", alpha = 0.5, s = 4)
    plt.scatter(steps_newthon, value_newthon, label = "Newthon method", c = "blue", alpha = 0.5, s = 4)
    plt.xlabel("Number of steps")
    plt.ylabel("Minimal value found")
    plt.title("Wykres obu metod dla punktu {} przy parametrach: learning_rate = {} i maksymalnej ilości kroków = {}".format(start_point, random_learning_rate, random_max_steps))
    plt.legend()
    plt.show()


###plot_random_scatter(steepest_gradient_desc, newthon_method, np.array([[-5], [3]]), 10)
start = -100
stop = 100
n_values = 5
start_point = np.array([[-5], [3]])

result = newthon_method(function, gradient, inv_hessian, start_point)
x_vals = np.array(result[2])
y_vals = np.array(result[3])
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array(result[4])

min_coordinates = np.array([result[5],result[6]])
print(hessian(-5,5))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, Z, linewidth=0, cmap = cm.coolwarm, antialiased=False)
x = start_point[0,0]
print()
print(x)
y = start_point[1,0]
print(y)
z = function(x,y)
min = function(min_coordinates[0], min_coordinates[1])
ax.plot([x], [y], [z[0]],markerfacecolor='k', markeredgecolor='k', marker='o', markersize=15)
ax.plot([min_coordinates[0]], [min_coordinates[1]], min[0],markerfacecolor='m', markeredgecolor='m', marker='p', markersize=10)
plt.show()




