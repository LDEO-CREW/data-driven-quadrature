import random
import numpy as np
import cvxpy as cp
import sys
from copy import deepcopy as copy

def error_message(info):
    """Prints an error message to standard error

    :param info: string containing the error message to print
    """
    print(info, sys.stderr)

def verbose_print(info, verbose, threshold):
    """Prints optimization messages given a verbosity level

    :param info: string containing message to print
    :param verbose: user-defined verbosity level to print
    :param threshold: message threshold to determine whether to print the message
    """
    if verbose >= threshold:
        print(info)

def check_params(x, y_ref, C, M, params, x_sup=None):
    """Checks validity of user-provided data (x and y_ref), cost function (C), map function (M), and parameters (params)
    
    :param x: xarray dataset containing integration axes
    :param y_ref: numpy multi-dimensional array containing reference/target values
    :param C: cost function
    :param M: mapping function
    :param params: dictionary of user-defined parameters for optimization loop and integration axes
    :param x_sup: optional parameter passed to map function
    :returns: integer representing success (0) or error (<0) 
    """

    # check if the cost function returns a valid cost value
    try:
        self_cost = C(y_ref, y_ref).value
        # cost function must return an integer or a float
        if not isinstance(self_cost, (int, float)):
            error_message("Cost Function Error: Invalid return type (" + type(self_cost + ") from cost function. Return type must be integer or float."))
            return -1
        # cost function evaluated on two identical reference matrices must return 0
        elif self_cost != 0:
            error_message("Cost Function Error: Invalid cost function. Cost function evaluated on two instances of reference data must equal zero (C(y_ref, y_ref) == 0).")
            return -1
    # cost function could not be run for some reason
    except:
        error_message("Cost Function Error: Cost function cannot be evaluated on reference output.")
        return -1
    
    # check if all user-defined parameters are valid
    try:
        # n_points must be a positive integer
        if 'n_points' in params and params['n_points'] < 1 or not isinstance(params['n_points'], int):
            error_message("Parameter Error: 'n_points' must be a positive, non-zero integer.")
            return -2
        # epochs must be a positive integer
        if 'epochs' in params and params['epochs'] < 1 or not isinstance(params['epochs'], int):
            error_message("Parameter Error: 'epochs' must be a positive, non-zero integer.")
            return -2
        # block_size must be a positive integer
        if 'block_size' in params and params['block_size'] < 1 or not isinstance(params['block_size'], int):
            error_message("Parameter Error: 'block_size' must be a positive, non-zero integer.")
            return -2
        # success must be a positive integer less than or equal to than block_size
        if 'success' in params and params['success'] < 1 or not isinstance(params['success'], int) or params['success'] > params['block_size']:
            error_message("Parameter Error: 'success' must be a positive, non-zero integer, less than or equal to block_size.")
            return -2
        if 'random_seed' in params:
            # random_seed must be an integer
            if not isinstance(params['random_seed'], int):
                error_message("Parameter Error: 'random_seed' must be an integer.")
                return -2
            # if valid, set the random seed for the program
            else:
                random.seed(params['random_seed'])
        if 'verbose' in params and params['verbose'] not in [0, 1, 2, 3]:
            # verbose must be an integer between 0 and 3
            error_message("Parameter Error: 'verbose' must be one of [0, 1, 2, 3].")
            return -2
        elif 'verbose' not in params:
            params['verbose'] = 1
        # solver must be an available and CVXPY-compatible solver
        if 'solver' in params and params['solver'] not in cp.installed_solvers():
            error_message("Solver Error: User chosen solver (" + params['solver'] + ") not a valid/installed solver.")
            return -2
        # check integration axes with data
        axes_list = params['integration_list']
        # at least one integration axis must be provided
        if len(axes_list) < 1:
            error_message("Parameter Error: 'integration_list' is empty.")
            return -3
        # all integration axes must be defined in the xarray dataset (x)
        for axis_name in axes_list:
            if axis_name not in x.coords:
                error_message("Parameter Error: '" + axis_name + "' not in data coordinates.")
                return -3
    except:
        error_message("Parameter Error: One or more of ['n_points', 'epochs', 'block_size', 'success', 'integration_list'] not included in parameter dictionary.")
        return -2

    # check mapping function for proper output shape
    try:
        # select a random pointset and calling the mapping function on it\\
        sized_integration_axes_list = [(axis, len(x[axis])) for axis in params['integration_list']]
        test_points = select_point_set(sized_integration_axes_list, params['n_points'])
        v = M(x, test_points, x_sup)
        # check proper mapped vector shape in the single-dimensional case
        if len(v.shape) == 2 and v.shape != (y_ref.shape[0], params['n_points']):
            error_message("Map Function Error: Map function returned an object of shape", v.shape, "while", str((y_ref.shape[0], params['n_points'])), "expected.")
            return -4
        # check proper mapped vector shape in the multi-dimensional case
        elif len(v.shape) == 3 and v.shape != (len(sized_integration_axes_list), y_ref.shape[0], params['n_points']):
            error_message("Map Function Error: Map function returned an object of shape", v.shape, "while", str((len(sized_integration_axes_list), y_ref.shape[0], params['n_points'])), "expected.")
            return -4
    except:
        error_message("Map Function Error: Map function cannot be evaluated on point set.")
        return -4
    
    return 0


def flatten_history(history):
    """Flattens history object returned by the optimization loop

    :param history: history dictionary returned from an optimization loop
    :returns: history dictionary with all multi-dimensional lists fully flattened
    """
    new_history = {}
    flatten_keys = ['cost', 'point_sets', 'weight_sets']
    for key in flatten_keys:
        full_data = history[key]
        flat_data = []
        for block in full_data:
            for value in block:
                flat_data.append(value)
        new_history[key] = flat_data
    new_history['temperature_history'] = history['temperature_history']
    new_history['best'] = history['best']
    return new_history

def prob_move(current_cost, last_cost, T):
    """Calculates the probability of a move given the previous cost, current cost, and temperature

    :param current_cost: cost of the new point set
    :param last_cost: cost of the previous point set
    :param T: annealing temperature
    :returns: float between 0 and 1 representing the probability of accepting the new state
    """
    return np.exp((last_cost - current_cost)/T)

def select_point(sized_integration_axes_list):
    """Selects an integration point from a list of integration axes and corresponding axes lengths

    :param sized_integration_axes_list: list of tuples representing the integration axis name and length of integration axis
    :returns: list representing a point with the order of indices matching the given order of integration axes
    """
    point = []
    # TODO: Add user-defined limit selection
    for axis, size in sized_integration_axes_list:
        point.append(random.randrange(size))
    return point

def neighbor(point_set, sized_integration_axes_list):
    """"Returns a neighbor state of a given point set (one point changed)
    
    :param point_set: a list of points, each matching the defined integration axes list
    :param sized_integration_axes_list: list of tuples representing the integration axis name and length of integration axis
    :returns: a new object containing a point set that is a neighbor of the input point set
    """
    point_set_copy = copy(point_set)
    replace_index = random.randrange(len(point_set))
    point_set_copy[replace_index] = select_point(sized_integration_axes_list)
    return point_set_copy

def select_point_set(sized_integration_axes_list, n_points):
    """"Selects an initial point set given integration axes list and number of points to select
    
    :param sized_integration_axes_list: list of tuples representing the integration axis name and length of integration axis
    :param n_points: number of points to select
    :returns: a point set of size n_points
    """
    integration_points = []
    while len(integration_points) < n_points:
        new_point = select_point(sized_integration_axes_list)
        if new_point not in integration_points:
            integration_points.append(new_point)
    return integration_points

def find_normalization_vector(x, integration_axes):
    """Returns a normalization vector for all given integration axes
    
    :param x: xarray dataframe containing integration axes
    :param integration_axes: list of integration axes in x
    :returns: list of normalization tuples (scale, offset) to normalize integration axes to [0, 1]
    """
    return [(abs(x[axis].values[-1] - x[axis].values[0]), min(x[axis].values[-1], x[axis].values[0])) for axis in integration_axes]

def find_weights(v, y_ref, C, solver):
    """
    
    :param v:
    :param y_ref:
    :param C:
    :param solver: solver to use for optimization
    :returns:
    """
    weights = cp.Variable(v.shape[-1], nonneg=True)
    y_hat = weights @ v.T
    cost = C(y_hat, y_ref)
    constraint_list = []
    constraint_list = [cp.sum(weights) == 1.0]
    objective_fnc = cp.Minimize(cost)
    prob = cp.Problem(objective=objective_fnc, constraints=constraint_list)
    prob.solve(max_iters = 1000, solver=solver)
    if prob.status != 'optimal':
        weights = np.zeros(len(v[0]))
        cost = 10000000 # change this value later
    else:
        weights = np.array(weights.value)
        cost = objective_fnc.value

    return weights, cost


def anneal_loop(x, y_ref, C, M, params, x_sup=None):
    """Main annealing loop to find the optimal point set and corresponding weights
    
    :param x: xarray dataset containing integration axes
    :param y_ref: numpy multi-dimensional array containing reference/target values
    :param C: cost function
    :param M: mapping function
    :param params: dictionary of user-defined parameters for optimization loop and integration axes
    :param x_sup: optional parameter passed to map function
    :returns: history object containing point set, weight, cost, and temperature history of optimization
    """
    cost_history = []
    point_set_history = []
    weight_set_history = []
    temperature_history = []  
    solver = 'ECOS_BB' if 'solver' not in params else params['solver']

    n_epochs = params['epochs'] if 'epochs' in params.keys() else 100
    n_success = params['success'] if 'success' in params.keys() else 25
    block_size = params['block_size'] if 'block_size' in params.keys() else 50
    best_cost = np.infty
    T_fact = 0.9

    sized_integration_axes_list = [(axis, len(x[axis])) for axis in params['integration_list']]
    point_set = select_point_set(sized_integration_axes_list, params['n_points'])

    # initial block run to determine starting temperature (Buehler et al., 2010)
    block_cost_history = []
    block_point_history = []
    block_weight_history = []
    for i in range(block_size):
        print_str = "INITIAL BLOCK: iteration " + str(i)
        verbose_print(print_str, params['verbose'], 3)
        v = M(x, point_set, x_sup)
        w, c = find_weights(v, y_ref, C, solver)
        block_cost_history.append(c)
        block_point_history.append(copy(point_set))
        block_weight_history.append(w)
        if c <= best_cost:
            best_cost = c
            best_index = (0, i)
        point_set = neighbor(point_set, sized_integration_axes_list)

    # take this initial block as the first block of optimization
    cost_history.append(copy(block_cost_history))
    point_set_history.append(copy(block_point_history))
    weight_set_history.append(copy(block_weight_history))

    # choose initial temperature s.t. 99% of moves in the initial block would be accepted
    T = -np.mean(np.abs(np.diff(block_cost_history)))/np.log(0.99)

    # set initial cost, points, and weights
    v = M(x, point_set, x_sup)
    current_weights, current_cost = find_weights(v, y_ref, C, solver)
    last_cost = current_cost

    # primary optimization loop
    for epoch in range(1, n_epochs):
        block_successes = 0
        block_cost_history = []
        block_point_history = []
        block_weight_history = []
        for block_idx in range(block_size):
            # Mapping function here allows for a very general use-case with non-linear transforms
            new_point_set = neighbor(point_set, sized_integration_axes_list)
            v = M(x, new_point_set, x_sup)
            current_weights, current_cost = find_weights(v, y_ref, C, solver)
            
            #save cost, point set, and weight to history
            block_cost_history.append(current_cost)
            block_point_history.append(copy(new_point_set))
            block_weight_history.append(copy(current_weights))
            print_str = "EPOCH: " + str(epoch) + "\tBLOCK: " + str(block_idx) + "\tCOST: " + "{:.3e}".format(current_cost)
            verbose_print(print_str, params['verbose'], 3)

            # update best index if necessary
            if current_cost <= best_cost:
                best_cost = current_cost
                best_index = (epoch, block_idx)
                block_successes += 1
                point_set = new_point_set
            # compute probability and accept move if applicable 
            elif random.random() < prob_move(current_cost, last_cost, T):
                block_successes += 1
                point_set = new_point_set

            last_cost = current_cost

            if block_successes >= n_success:
                break

        print_str = "EPOCH: " + str(epoch) + "\tAVG COST: " + "{:.3e}".format(np.mean(block_cost_history)) + "\tBEST COST: " + "{:.3e}".format(best_cost)
        verbose_print(print_str, params['verbose'], 2)

        temperature_history.append(T)
        cost_history.append(copy(block_cost_history))
        point_set_history.append(copy(block_point_history))
        weight_set_history.append(copy(block_weight_history))


        # decrease temperature if this block had an lower mean cost than the previous block
        if (np.mean(cost_history[-1]) <= np.mean(cost_history[-2])):
            T *= T_fact

        if block_successes == 0:
            break

    history = {
        'cost': cost_history,
        'point_sets': point_set_history,
        'weight_sets': weight_set_history,
        'temperature_history': temperature_history,
        'best': best_index
    }
    print_str = "OPTIMIZATION COMPLETE\n"
    print_str += "\tBEST COST LOCATION\tEPOCH: " + str(best_index[0]) + "\tBLOCK: " + str(best_index[1]) + "\n"
    print_str += "\tBEST COST: " + "{:.3e}".format(cost_history[best_index[0]][best_index[1]])
    verbose_print(print_str, params['verbose'], 1)
    return history

def optimize(x, y_ref, C, M, params, x_sup=None, verbose=False):
    """User-called function to check validity of inputs and determine optimal point set

    :param x: xarray dataset containing integration axes
    :param y_ref: numpy multi-dimensional array containing reference/target values
    :param C: cost function
    :param M: mapping function
    :param params: dictionary of user-defined parameters for optimization loop and integration axes
    :param x_sup: optional parameter passed to map function
    :param verbose:
    :returns: history object containing point set, weight, cost, and temperature history of optimization
    """
    if (check_val := check_params(x, y_ref, C, M, params, x_sup)) < 0: return check_val
    return anneal_loop(x, y_ref, C, M, params, x_sup=x_sup)