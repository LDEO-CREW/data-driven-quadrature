import random
import numpy as np
import cvxpy as cp
import sys
from copy import deepcopy as copy

# TODO: provide benchmarking for mapping, cost fnc, etc.
# TODO: allow random seeding

def error_message(info):
    print(info, sys.stderr)

def check_params(x, y_ref, C, M, params, x_sup=None):
    # check if the cost function returns a valid cost value
    try:
        self_cost = C(y_ref, y_ref)
        print(type(self_cost))
        if not isinstance(self_cost, (int, float)):
            error_message("Cost Function Error: Invalid return type ("+type(self_cost+") from cost function. Return type must be integer or float."))
            return -1
        elif self_cost != 0:
            error_message("Cost Function Error: Invalid cost function. Cost function evaluated on two instances of reference data must equal zero (C(y_ref, y_ref) == 0).")
            return -1
    except:
        error_message("Cost Function Error: Cost function cannot be evaluated on reference output.")
        return -1
    try:
        # check if model parameters are defined properly with allowable values
        if params['n_points'] < 1 or not isinstance(params['n_points'], int):
            error_message("Parameter Error: 'n_points' must be a positive, non-zero integer.")
            return -2
        if params['epochs'] < 1 or not isinstance(params['epochs'], int):
            error_message("Parameter Error: 'epochs' must be a positive, non-zero integer.")
            return -2
        if params['block_size'] < 1 or not isinstance(params['block_size'], int):
            error_message("Parameter Error: 'block_size' must be a positive, non-zero integer.")
            return -2
        if params['success'] < 1 or not isinstance(params['success'], int) or params['success'] > params['block_size']:
            error_message("Parameter Error: 'success' must be a positive, non-zero integer, less than or equal to block_size.")
            return -2
        # check integration axes with data
        axes_list = params['integration_list']
        if len(axes_list) < 1:
            error_message("Parameter Error: 'integration_list' is empty.")
            return -3

        for axis_name in axes_list:
            if axis_name not in x.coords:
                error_message("Parameter Error: '" + axis_name + "' not in data coordinates.")
                return -3
            # TODO: Do we need to check if all integration axes lengths need to be > the number of integration points?
    except:
        error_message("Parameter Error: One or more of ['n_points', 'epochs', 'block_size', 'success', 'integration_list'] not included in parameter dictionary.")
        return -2
    
    # check mapping function for proper output shape


    return 0


def flatten_history(history):
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
    return np.exp((last_cost - current_cost)/T)

# Verify cost function on same data evaluates to 0 and cost can be run on y_ref
def verify_cost_fnc(C, y_ref):
    try:
        zero_cost = C(y_ref, y_ref)
        if zero_cost != 0:
            return -1 # standard error
    except:
        return -1

def select_point(sized_integration_axes_list):
    point = []
    # TODO: for schemes with multiple variables, do we want to select points without replacement?
    for axis, size in sized_integration_axes_list:
        point.append(random.randrange(size))
    return point

def neighbor(point_set, sized_integration_axes_list):
    point_set_copy = copy(point_set)
    replace_index = random.randrange(len(point_set))
    point_set_copy[replace_index] = select_point(sized_integration_axes_list)
    return point_set_copy

def select_point_set(x, sized_integration_axes_list, n_points):
    integration_points = []
    while len(integration_points) < n_points:
        new_point = select_point(sized_integration_axes_list)
        if new_point not in integration_points:
            integration_points.append(new_point)
    return integration_points

def find_normalization_vector(x, integration_axes):
    return [(abs(x[axis].values[-1] - x[axis].values[0]), min(x[axis].values[-1], x[axis].values[0])) for axis in integration_axes]

def find_weights(v, y_ref, C, norm_vector):
    # TODO: Currently asking for y_ref and v to both be numpy arrays
    weights = cp.Variable(v.shape[-1], nonneg=True)
    # y_hat = np.zeros(y_ref.shape)
    # y_hat = weights @ ((v.T - norm_vector[0][0]) / norm_vector[0][1]) if len(norm_vector) == 1 else sum([weights @ ((v[i].T - norm_vector[i][1]) / norm_vector[i][0])] for i in range(len(norm_vector)))
    y_hat = weights @ v.T
    # TODO: Fix cost functions
    cost = C(y_hat, y_ref)
    # cost = cp.norm(y_ref - y_hat)
    constraint_list = []
    constraint_list = [cp.sum(weights) == 1.0]
    objective_fnc = cp.Minimize(cost)
    prob = cp.Problem(objective=objective_fnc, constraints=constraint_list)
    # TODO: Add solver choice
    prob.solve(max_iters = 1000, solver='ECOS_BB')
    if prob.status != 'optimal':
        weights = np.zeros(len(v[0]))
        cost = 10000000 # change this value later
    else:
        weights = np.array(weights.value)
        cost = objective_fnc.value

    return weights, cost


def anneal_loop(x, y_ref, C, M, params, point_set, sized_integration_axes_list, x_sup=None, verbose=False):
    cost_history = []
    point_set_history = []
    weight_set_history = []
    temperature_history = []
    norm_vector = find_normalization_vector(x, params['integration_list'])    

    n_epochs = params['epochs'] if 'epochs' in params.keys() else 100
    n_success = params['success'] if 'success' in params.keys() else 50
    block_size = params['block_size'] if 'block_size' in params.keys() else 100
    best_index = (0, 0)
    best_cost = np.infty
    T_fact = 0.9

    optimization_passes = 1

    # initial block run to determine starting temperature (Buehler et al., 2010)
    block_cost_history = []
    block_point_history = []
    block_weight_history = []
    for i in range(block_size):
        print("INITIAL BLOCK: iteration", i)
        v = M(x, point_set, x_sup)
        w, c = find_weights(v, y_ref, C, norm_vector)
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
    v = M(x, point_set, x_sup) # TODO: add req for flattened shape of v
    current_weights, current_cost = find_weights(v, y_ref, C, norm_vector)
    # point_set_history.append(copy(point_set))
    # weight_set_history.append(copy(current_weights))
    # cost_history.append(current_cost)
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
            v = M(x, new_point_set, x_sup) # TODO: add req for flattened shape of v
            current_weights, current_cost = find_weights(v, y_ref, C, norm_vector)
            block_cost_history.append(current_cost)
            block_point_history.append(copy(new_point_set))
            block_weight_history.append(copy(current_weights))
            print(epoch, block_idx, point_set, current_cost)

            # update best index if necessary
            if current_cost <= best_cost:
                best_cost = current_cost
                best_index = (epoch, block_idx)
                block_successes += 1
                point_set = new_point_set
            elif random.random() < prob_move(current_cost, last_cost, T):
                block_successes += 1
                point_set = new_point_set

            last_cost = current_cost

            optimization_passes += 1
            if block_successes >= n_success:
                break

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
    return history

def optimize(x, y_ref, C, M, params, x_sup=None, verbose=False):
    if (check_val := check_params(x, y_ref, C, M, params, x_sup)) < 0: return check_val
    sized_integration_axes_list = [(axis, len(x[axis])) for axis in params['integration_list']]
    point_set = select_point_set(x, sized_integration_axes_list, params['n_points'])
    return anneal_loop(x, y_ref, C, M, params, point_set, sized_integration_axes_list, x_sup=x_sup)