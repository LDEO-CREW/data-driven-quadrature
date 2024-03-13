import random
import numpy as np
import cvxpy as cp
from copy import deepcopy as copy

# TODO: rename all functions with specific prefixes to avoid naming conflicts
# TODO: provide benchmarking for mapping, cost fnc, etc.
# TODO: allow random seeding
# TODO: require xarray inputs 
# TODO: for testing, use n=16 points
# TODO: make toy dataset for testing
# TODO: sum weights normalize to one

# TODO: standardized error messages and trace for incorrect input
# def error_message(info):

# # Returns a deep copy of a 1D array
# def copy(arr):
#     return copy.deepcopy(arr)

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
    
def evaluate_cost(x, y_ref, M, C):
    return C(M(x), y_ref)

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

def find_weights(v, y_ref, C):
    # TODO: Currently asking for y_ref and v to both be numpy arrays
    weights = cp.Variable(v.shape[-1], nonneg=True)
    # y_hat = np.zeros(y_ref.shape)
    y_hat = weights @ v.T
    # TODO: Fix cost functions
    # cost = C(y_hat, y_ref)
    cost = cp.norm(y_ref - y_hat)
    constraint_list = []
    # TODO: Impose the weight sum constraint(s)
    # TODO: Maybe add in support for custom constraints here
    objective_fnc = cp.Minimize(cost)
    prob = cp.Problem(objective=objective_fnc, constraints=constraint_list)
    # TODO: Allow max_iters to be chosen by user using params
    prob.solve(max_iters = 1000)
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
    block_lengths = [1]
    n_epochs = params['epochs'] if 'epochs' in params.keys() else 5000
    n_success = params['success'] if 'success' in params.keys() else 200
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
        v = M(x, point_set, x_sup)
        w, c = find_weights(v, y_ref, C)
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
    current_weights, current_cost = find_weights(v, y_ref, C)
    point_set_history.append(copy(point_set))
    weight_set_history.append(copy(current_weights))
    cost_history.append(current_cost)
    last_cost = current_cost

    # primary optimization loop
    for epoch in range(1, n_epochs):
        block_successes = 0
        block_cost_history = []
        block_point_history = []
        block_weight_history = []
        for block_idx in range(block_size):
            print(epoch, block_idx, point_set)
            # Mapping function here allows for a very general use-case with non-linear transforms
            new_point_set = neighbor(point_set, sized_integration_axes_list)
            v = M(x, new_point_set, x_sup) # TODO: add req for flattened shape of v
            current_weights, current_cost = find_weights(v, y_ref, C)
            block_cost_history.append(current_cost)
            block_point_history.append(copy(new_point_set))
            block_weight_history.append(copy(current_weights))

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
    # TODO: do all the initial checks to make sure the input is valid
    integration_axes_list = params['integration_list']
    # TODO: check to make sure all axes are in x
    sized_integration_axes_list = [(axis, len(x[axis])) for axis in integration_axes_list]
    n_points = params['n_points']
    # TODO: check to make sure n_points > 0 
    point_set = select_point_set(x, sized_integration_axes_list, n_points)
    return anneal_loop(x, y_ref, C, M, params, point_set, sized_integration_axes_list, x_sup=x_sup)