import numpy as np
import random


# Part I

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification parameters
    Args:
    :param feature_vector: A numpy array describing the given data point.
    :param label: A real valued number, the correct classification of the data point
    :param theta: A numpy array describing the linear classifier
    :param theta_0: A real valued number representing the offset parameter
    :return: A real number representing the hinge loss associated with the given datapoint and parameters
    """
    y = np.dot(theta, feature_vector) + theta_0
    loss = max(0.0, 1 - y * label)
    return loss


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification parameters

    Args:
    :param feature_matrix: A numpy matrix describing the given data, Each row represents a single data point
    :param labels: A numpy array where Kth element of the array is the correct classification of the Kth row
                  of the feature matrix.
    :param theta: A numpy array describing the linear classification
    :param theta_0: A real valued number representing the offset parameter
    :return: A real number representing the hinge loss associated with the given dataset and parameters.
             This number should be the average hinge loss across all of the points in the feature matrix
    """
    loss = 0
    for idx in range(len(feature_matrix)):
        loss += hinge_loss_single(feature_matrix[idx], labels[idx], theta, theta_0)
    return loss / len(labels)


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Property updates the classification parameter, theta and theta_0 on a single step of the perceptron
    algorithm.

    Argss:
    :param feature_vector: A numpy array describing a single data point.
    :param label: The correct classification of the feature vector
    :param current_theta: The current theta being used by the perceptron algorithm before this update
    :param current_theta_0: The current theta_0 being used by the perceptron algorithm before this update
    :return: A tuple where the first element is a numpy array with value of the theta after the current
             update has completed and the second element is a real valued number iwth the value of theta_0 after
             the current update has completed.
    """
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
        return current_theta + label * feature_vector, current_theta_0 + label
    return current_theta, current_theta_0


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    samples, features = feature_matrix.shape
    theta = np.zeros(features)
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(samples):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    (samples, features) = feature_matrix.shape
    theta = np.zeros(features)
    theta_sum = np.zeros(features)
    theta_0 = 0.0
    theta_0_sum = 0.0
    for t in range(T):
        for i in get_order(samples):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return theta_sum / (samples * T), theta_0_sum / (samples * T)


def pegasos_single_step_update(feature_vector, label, l, eta, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    multi = 1 - (eta * l)
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return (multi * current_theta) + (eta * label * feature_vector), current_theta_0 + (eta * label)
    return multi * current_theta, current_theta_0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).


    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    samples, features = feature_matrix.shape
    theta = np.zeros(features)
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(samples):
            count += 1
            eta = 1.0 / np.sqrt(count)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return theta, theta_0


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    raise NotImplementedError
