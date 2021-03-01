import sentiment_analysis.algorithms as ai
import logging
import numpy as np
import os
import sys
import traceback
import time


def equals(x, y):
    if type(y) == np.ndarray:
        return (x == y).all
    return x == y


def check_real(ex_name, f, exp_result, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        logging.error('%s: not implemented', ex_name)
        return True
    if not np.isreal(res):
        logging.error('%s: does not return a real number, type: %s', ex_name, type(res))
        return True
    if res != exp_result:
        logging.error('%s: incorrect answer. Expected %s, got %s', ex_name, exp_result, res)
        return True


def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        logging.error('%s: not implemented', ex_name)
        return True
    if not type(res) == tuple:
        logging.error('%s: does not return a tuple, type: %s', ex_name, type(res))
        return True
    if not len(res) == len(exp_res):
        logging.error('%s: expected a tuple of size %d, but got tuple of size %d', ex_name, len(exp_res), len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        logging.error('%s: incorrect answer, expected %s, but got %s', ex_name, exp_res, res)
        return True


def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except FileNotFoundError:
        logging.error("%s: Not implemented", ex_name)
        return True
    if not type(res) == list:
        logging.error('%s: does not return a list, type %s', ex_name, type(res))
        return True
    if not len(res) == len(exp_res):
        logging.error('%s: expected a list of size %d but got the list of size %d', ex_name, len(exp_res), len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        logging.error('%s: incorrect answer. Expected, %s, got: %s', ex_name, exp_res, res)
        return True


def check_get_order():
    ex_name = "get_order"
    if check_list(ex_name, ai.get_order, [0], 1):
        logging.info("You should revert `get_order` to its original implementation for this test to pass")
        return
    if check_list(ex_name, ai.get_order, [1, 0], 2):
        logging.info("You should revert 'get_order' to its original implementation for this test to pass")
        return
    logging.info("%s: PASS", ex_name)


def check_hinge_loss_single():
    do_check_hinge_loss_single(1 - 0.8, np.array([1, 2]), 1, np.array([-1, 1]), -0.2)
    do_check_hinge_loss_single(0.0, np.array([0.92454549, 0.80196337, 0.38027544, 0.69273305,
                                              0.01614677, 0.35642963, 0.83956723, 0.83481115,
                                              0.66612153, 0.96900118]), 1.0,
                               np.array([0.05408063, 0.06234699, 0.13148364, 0.07217788, 3.09659492,
                                         0.14028014, 0.05955449, 0.05989379, 0.07506138, 0.05159952]), 0.5)


def check_hinge_loss_full():
    ex_name = 'hinge_loss_full'
    feature_matrix = np.array([[1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
    exp_result = 1 - 0.8
    if check_real(ex_name, ai.hinge_loss_full, exp_result, feature_matrix, label, theta, theta_0):
        return
    logging.info("%s: PASS", ex_name)


def do_check_hinge_loss_single(ex_result, feature_vector, label, theta, theta_0):
    ex_name = 'hinge_loss_single'
    if check_real(ex_name, ai.hinge_loss_single, ex_result, feature_vector, label, theta, theta_0):
        return
    logging.info("%s: PASS", ex_name)


def check_perceptron_single_step_update():
    ex_name = 'perceptron_single_step_update'
    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    exp_result = (np.array([0, 3]), -0.5)
    if check_tuple(ex_name, ai.perceptron_single_step_update, exp_result, feature_vector, label, theta, theta_0):
        return
    logging.info("%s: PASS", ex_name)


def check_perceptron():
    ex_name = "perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    t = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(ex_name, ai.perceptron, exp_res, feature_matrix, labels, t):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    t = 1
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(ex_name, ai.perceptron, exp_res, feature_matrix, labels, t):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    t = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(ex_name, ai.perceptron, exp_res, feature_matrix, labels, t):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    t = 2
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(ex_name, ai.perceptron, exp_res, feature_matrix, labels, t):
        return

    logging.info('%s: PASS', ex_name)


def check_average_perceptron():
    ex_name = "average_perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(ex_name, ai.average_perceptron, exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([-0.5, 1]), 1.5)
    if check_tuple(ex_name, ai.average_perceptron, exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(ex_name, ai.average_perceptron, exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([-0.25, 1.5]), 1.75)
    if check_tuple(ex_name, ai.average_perceptron, exp_res, feature_matrix, labels, T):
        return

    logging.info('%s: PASS', ex_name)


def check_pegasos_single_update():
    ex_name = "pegasos_single_update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.4)
    if check_tuple(ex_name, ai.pegasos_single_step_update, exp_res, feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 1])
    label, theta, theta_0 = 1, np.array([-1, 1]), 1
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.08]), 1.1)
    if check_tuple(ex_name + " (boundary case)", ai.pegasos_single_step_update, exp_res, feature_vector, label, L, eta,
                   theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -2
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.9)
    if check_tuple(ex_name, ai.pegasos_single_step_update, exp_res, feature_vector, label, L, eta, theta, theta_0):
        return

    logging.info('%s: PASS', ex_name)


def check_pegasos():
    ex_name = "pegasos"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    L = 0.2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(ex_name, ai.pegasos, exp_res, feature_matrix, labels, T, L):
        return

    feature_matrix = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 1])
    T = 1
    L = 1
    exp_res = (np.array([1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)]), 1)
    if check_tuple(ex_name, ai.pegasos, exp_res, feature_matrix, labels, T, L):
        return

    logging.info('%s: PASS', ex_name)


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', level=logging.DEBUG, )
    logging.info("Import algorithms")
    check_get_order()
    check_hinge_loss_single()
    check_hinge_loss_full()
    check_perceptron_single_step_update()
    check_perceptron()
    check_average_perceptron()
    check_pegasos_single_update()
    check_pegasos()


if __name__ == "__main__":
    main()
