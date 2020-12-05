import matplotlib.pyplot as plt
import pandas as pd
from random import randint
import random
import numpy as np
from sklearn import model_selection
import seaborn as sns
from array import array
from matplotlib import cm
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
import os


# returns the sigmoid of an input
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# the derivative of the sigmoid function
def derivative_sigmoid(z):
    return z * (1 - z)


# given two different weights and a bias, returns x coordinates and y coordinates
# so we can plot a line using those coordinates
def plot_decision_boundary(weight_one, weight_two, bias):
    line_y_coords = []
    line_x_coords = [1, 2, 3, 4, 5, 6, 7, 8]
    slope = -weight_one / weight_two
    y_intercept = -bias / weight_two

    for i in range(8):
        line_y_coords.append(slope * line_x_coords[i] + y_intercept)

    result_coords = [line_x_coords, line_y_coords]
    return result_coords


# computes the mean squared error and does gradient descent. Dataset is the values of petal
# length and petal width. Patterns is the values of the class of each datapoint
def mean_squared_error(dataset, patterns, weight_one, weight_two, bias, learning_rate):
    weight_one_update = 0
    weight_two_update = 0
    bias_update = 0
    total_squared_error = 0.0

    size = len(dataset)
    for i in range(size):
        z = (dataset[i][0] * weight_one) + (dataset[i][1] * weight_two) + bias
        sigmoid_result = sigmoid(z)
        derivative_result = derivative_sigmoid(sigmoid_result)
        if patterns[i] == 'versicolor':
            actual_class = 0
        else:
            actual_class = 1

        err_result = sigmoid_result - actual_class
        squared_error_result = err_result ** 2
        bias_update += (err_result * derivative_result)
        weight_one_update += (err_result * derivative_result) * dataset[i][0]
        weight_two_update += (err_result * derivative_result) * dataset[i][1]
        total_squared_error += squared_error_result

    # computes new weights based on the updated weights computed in the above loop
    new_weight_one = weight_one - learning_rate * (weight_one_update * 2 / size)
    new_weight_two = weight_two - learning_rate * (weight_two_update * 2 / size)
    new_bias = bias - learning_rate * (bias_update * 2 / size)

    # computes the mean squared error
    squared_error_mean = (total_squared_error / size)
    # returns all of the above computed values
    return_array = [squared_error_mean, new_weight_one, new_weight_two, new_bias]
    return return_array


# plots a scatter plot with the iris petal length and petal width data
# takes in a problem letter to know what to title the plot
# if a decision boundary needs to be plotted, also plots that
def problem_1_a(problem_letter, x_coords_line, y_coords_line):
    colors = ['red', 'blue']
    species = ['setosa', 'virginica', 'versicolor']
    for i in range(1, 3):
        species_df = data[data['species'] == species[i]]
        plt.scatter(species_df['petal_length'], species_df['petal_width'], color=colors[i - 1])

    plt.xlim(0, 10)
    plt.ylim(0, 6)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Petal Width vs Petal Length - Problem #1(A)')
    if problem_letter == 'c':
        plt.plot(x_coords_line, y_coords_line)
        plt.title('Petal Width vs Petal Length - Problem #1(C)')
    elif problem_letter == '2_b_small':
        plt.plot(x_coords_line, y_coords_line)
        plt.title('Petal Width vs Petal Length - Problem #2(B) with small error')
    elif problem_letter == '2_b_large':
        plt.plot(x_coords_line, y_coords_line)
        plt.title("Petal Width vs Petal Length - Problem #2(B) with large error")
    elif problem_letter == '2_e_beginning':
        plt.plot(x_coords_line, y_coords_line)
        plt.title('Petal Width vs Petal Length - Problem #2(E) before the small step')
    elif problem_letter == '2_e_end':
        plt.plot(x_coords_line, y_coords_line)
        plt.title("Petal Width vs Petal Length - Problem #2(E) after the small step")
    elif problem_letter == '4_c_5':
        plt.plot(x_coords_line, y_coords_line)
        plt.title("Problem #3(C) Initial Location of the Decision Boundary")
    elif problem_letter == '4_c_10000':
        plt.plot(x_coords_line, y_coords_line)
        plt.title("Problem #3(C) Middle Location of the Decision Boundary")
    elif problem_letter == '4_c_20000':
        plt.plot(x_coords_line, y_coords_line)
        plt.title("Problem #3(C) Final Location of the Decision Boundary")
    plt.show()


# computes a linear decision boundary by creating a training and test data
# takes in a problem letter to know which situation we are looking at
# depending on the problem letter, may look at datapoints near or far away from the decision boundary
# also computes the number of the test data set that were correct based on the computed decision boundary
def problem_1_b(problem_letter):
    # petal length and width
    dataset = data.iloc[50:, [2, 3]].values
    # correspond class
    patterns = data.iloc[50:, 4].values
    # split the dataset to get 60 test trial and we are not training the weights in 1b but later
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset, patterns, test_size=0.6)

    # initialize the two weights and bias - found by trial and error in problem 2_b
    weight_one = 0.32
    weight_two = 3.70
    bias = -7.61

    if problem_letter == 'e':
        x_test = [[5.1, 2], [5, 1.9], [5.1, 1.9], [4.8, 1.8], [5.1, 1.8]]
        y_test = ['virginica', 'viginica', 'versicolor', 'versicolor', 'virginica']

    if problem_letter == 'ee':
        x_test = [[6.6, 2.1], [6.3, 1.8], [6.1, 2.5], [6.7, 2.2], [4, 1.3], [4.6, 1.5], [3.3, 1], [3.9, 1.4]]
        y_test = ['virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'versicolor',
                  'versicolor', 'versicolor']
    size = len(x_test)
    num_wrong = 0
    num_right = 0
    for i in range(size):
        if y_test[i] == 'versicolor':
            actual_class = 0

        else:
            actual_class = 1

        z = x_test[i][0] * weight_one + x_test[i][1] * weight_two + bias
        sigmoid_result = sigmoid(z)
        if sigmoid_result < 0.5:
            predicted_class = 0
        else:
            predicted_class = 1

        print("actual class: ", actual_class)
        print("predicted class: ", predicted_class)
        if actual_class != predicted_class:
            num_wrong += 1
        else:
            num_right += 1
        print("--------")
    percent_correct = num_right / (num_right + num_wrong)
    if problem_letter == 'e':
        print("part 1e correctness when looking near the decision boundary: ", percent_correct)
        print('')
        print('')
    elif problem_letter == 'ee':
        print("part 1e correctness when looking far from the decision boundary: ", percent_correct)
        print('')
        print('')
    else:
        print("part 1b correctness: ", percent_correct)
        print('')
        print('')

        problem_1_c(weight_one, weight_two, bias)
        problem_1_d(dataset, patterns, weight_one, weight_two, bias)


# plots a scatter plot with a decision boundary based on two weights and a bias
def problem_1_c(weight_one, weight_two, bias):
    line_coords = plot_decision_boundary(weight_one, weight_two, bias)
    line_x_coords = line_coords[0]
    line_y_coords = line_coords[1]

    problem_1_a('c', line_x_coords, line_y_coords)


# helper function for 1_d, using the same weight coefficients to generate z
def z_function(x, y):
    z_result = 0.32 * x + 3.70 * y - 7.61
    return sigmoid(z_result)


# creates a surface plot of petal length and petal width on the x and y axes.
# sigmoid value is on the z axis. Kind of looks sigmoidish a little bit but is kind of ugly
def problem_1_d(dataset, patterns, weight_one, weight_two, bias):
    ax = plt.axes(projection='3d')
    x = np.linspace(0, 8, 100)
    y = np.linspace(0, 3, 100)
    x1, y1 = np.meshgrid(x, y)
    z1 = z_function(x1, y1)
    ax.plot_surface(x1, y1, z1)
    ax.set_xlabel('petal length - cm')
    ax.set_ylabel('petal width - cm')
    ax.set_zlabel('sigmoid value')
    ax.set_title('3D plot of neural network output')
    plt.show()


# shows the output of simple classifier by calling problem_1_b()
# shows output for datapoints near and far from the decision boundary
def problem_1_e():
    problem_1_b('e')
    problem_1_b('ee')


# shows output for the mean squared error for good weights and bias
# as well as mean squared error for bad weights and bias
# the values of "good" and "bad" parameters were found by trial and error
def problem_2_b():
    dataset = data.iloc[50:, [2, 3]].values
    patterns = data.iloc[50:, 4].values
    good_parameters = [.48, .99, -3.9]
    bad_parameters = [.99, .98, -3.2]
    good_mean_squared = mean_squared_error(dataset, patterns, good_parameters[0], good_parameters[1], good_parameters[2]
                                           , .001)
    bad_mean_squared = mean_squared_error(dataset, patterns, bad_parameters[0], bad_parameters[1], bad_parameters[2],
                                          .001)

    print("Good parameters mean squared error: ", good_mean_squared[0])
    print("Bad parameters mean squared error: ", bad_mean_squared[0])

    # plot small error
    line_coords = plot_decision_boundary(good_parameters[0], good_parameters[1], good_parameters[2])
    line_x_coords = line_coords[0]
    line_y_coords = line_coords[1]
    problem_1_a('2_b_small', line_x_coords, line_y_coords)
    # plot large error
    line_coords1 = plot_decision_boundary(bad_parameters[0], bad_parameters[1], bad_parameters[2])
    line_x_coords1 = line_coords1[0]
    line_y_coords1 = line_coords1[1]
    problem_1_a('2_b_large', line_x_coords1, line_y_coords1)


# shows how the decision boundary can shift after a very small amount of using
# gradient descent
def problem_2_e():
    colors = ['red', 'blue']
    species = ['setosa', 'virginica', 'versicolor']
    for i in range(1, 3):
        species_df = data[data['species'] == species[i]]
        plt.scatter(species_df['petal_length'], species_df['petal_width'], color=colors[i - 1])

    plt.xlim(0, 10)
    plt.ylim(0, 6)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Problem #2(E) - changes after one small step')

    dataset = data.iloc[50:, [2, 3]].values
    patterns = data.iloc[50:, 4].values
    weight_one = 0.7
    weight_two = 1
    bias = -3.2
    line_coords = plot_decision_boundary(weight_one, weight_two, bias)
    line_x_coords = line_coords[0]
    line_y_coords = line_coords[1]

    update_results = mean_squared_error(dataset, patterns, weight_one, weight_two, bias, .01)
    w1 = [weight_one, weight_two, bias]
    print('-----------Problem_2e----------------')
    print('The MSE before the step: ', update_results[0])
    print('The weights for the step:', w1)
    line_coords1 = plot_decision_boundary(update_results[1], update_results[2], update_results[3])
    line_x_coords1 = line_coords1[0]
    line_y_coords1 = line_coords1[1]
    w2 = [update_results[1], update_results[2], update_results[3]]
    update_results = mean_squared_error(dataset, patterns, update_results[1], update_results[2], update_results[3], .01)

    print('The MSE after the step: ', update_results[0])
    print('The weights after the step :', w2)
    print('-----------------------------------')
    plt.plot(line_x_coords, line_y_coords, color='green')
    plt.plot(line_x_coords1, line_y_coords1, color='red')
    plt.show()


# implements gradient descent with random weights and bias
# pauses in the beginning, middle, and end to show:
# 1) number trials vs mean squared error
# 2) scatter plot of data and the updated decision boundary
def problem_3_a_b_c():
    dataset = data.iloc[50:, [2, 3]].values
    patterns = data.iloc[50:, 4].values
    weight_one = random.random()
    weight_two = random.random()
    bias = -random.random()
    bias_multiplier = randint(1, 4)
    bias = bias * bias_multiplier

    mean_squared_error_list = []
    x_value_list = []
    for i in range(20000):
        model_result = mean_squared_error(dataset, patterns, weight_one, weight_two, bias, 0.01)
        weight_one = model_result[1]
        weight_two = model_result[2]
        bias = model_result[3]
        mean_squared_error_list.append(model_result[0])
        x_value_list.append(i)

        if model_result[0] < .1:
            i = 19999
        if i == 5 or i == 10000 or i == 19999:
            plt.plot(x_value_list, mean_squared_error_list)
            plt.xlabel("Number of trials")
            plt.ylabel("Mean Squared Errors")

            if i == 5:
                plt.title("Mean Squared Error After 5 trials")
            elif i == 10000:
                plt.title("Mean Squared Error After 10000 trials")
            else:
                plt.title("Mean Squared Error After reach final boundary")

            plt.show()

            line_coords = plot_decision_boundary(weight_one, weight_two, bias)
            line_x_coords = line_coords[0]
            line_y_coords = line_coords[1]
            if i == 5:
                problem_1_a("4_c_5", line_x_coords, line_y_coords)
            elif i == 10000:
                problem_1_a("4_c_10000", line_x_coords, line_y_coords)
            else:
                problem_1_a("4_c_20000", line_x_coords, line_y_coords)
                arr = [weight_one, weight_two, bias]
                print(arr)
                break


def problem_4():
    iris_data = pd.read_csv('irisdata.csv')

    x = iris_data.iloc[:, 0:4].values
    y = iris_data['species'].astype('category').cat.codes
    y = keras.utils.to_categorical(y, num_classes=None)
    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    # define keras model
    model = keras.Sequential()
    # layers
    model.add(keras.layers.Dense(4, input_dim=4, activation='tanh'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(keras.optimizers.Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=0)
    accuracy = model.evaluate(X_test, y_test)[1]
    error = model.evaluate(X_test, y_test)[0]
    print('Accuracy: {}'.format(accuracy))
    print('Error: {}'.format(error))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred > 0.5))

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)
    accuracy = knn.score(X_test, y_test)
    error = model.evaluate(X_test, y_test)[0]
    print('Accuracy: {}'.format(accuracy))

    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred > 0.5))


if __name__ == "__main__":
    data = pd.read_csv('irisdata.csv', sep=',')
    problem_1_a('a', [], [])
    problem_1_b('b')
    problem_1_e()
    problem_2_b()
    problem_2_e()
    problem_3_a_b_c()
    problem_4()
