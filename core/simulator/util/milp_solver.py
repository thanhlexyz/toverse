import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import warnings
import torch
import time
import sys
import os


MULTIPROCESS_MODEL = None

def has_relu_var(model):
    names = [v.VarName for v in model.getVars()]
    if not len(names):
        return False
    relus = [v for v in names if v.startswith('aReLU')]
    return len(relus) > 0


def _mip_solver_worker(candidate):

    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9:
            bound = bound_type(grb_model.objbound, reference)
            refined = abs(bound - reference) >= eps
        elif grb_model.status == 2:
            bound = grb_model.objbound
            refined = abs(bound - reference) >= eps
        elif grb_model.status == 15:
            bound = bound_type(1., -1.) * eps
            refined = True
        else:
            bound = reference
        return bound, refined, grb_model.status

    def solve_ub(model, v, out_ub, eps=1e-5):
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.optimize()
        vub, refined, status_ub = get_grb_solution(model, out_ub, min, eps=eps)
        return vub, refined, status_ub

    def solve_lb(model, v, out_lb, eps=1e-5):
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.optimize()
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max, eps=eps)
        return vlb, refined, status_lb

    refine_time = time.time()
    model = MULTIPROCESS_MODEL.copy()
    var_name = candidate
    v = model.getVarByName(var_name)
    out_lb, out_ub = v.LB, v.UB
    eps = 1e-5
    v.LB, v.UB = -np.inf, np.inf
    model.update()

    neuron_refined = False
    vlb, refined, status_lb = solve_lb(model, v, out_lb, eps=eps)
    neuron_refined = neuron_refined or refined
    vub, refined, status_ub = solve_ub(model, v, out_ub, eps=eps)
    neuron_refined = neuron_refined or refined


    return var_name, vlb, vub, neuron_refined

def _build_solver_input(model, input_lower, input_upper):
    layer_vars = []
    if input_lower.ndim == 4:
        assert input_lower.size(0) == input_upper.size(0) == 1
        input_lower = input_lower.squeeze(0)
        input_upper = input_upper.squeeze(0)

    this_layer_shape = input_lower.shape
    for dim, (lb, ub) in enumerate(zip(input_lower.flatten(), input_upper.flatten())):
        v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{dim}')
        layer_vars.append(v)
    layer_vars = np.array(layer_vars).reshape(this_layer_shape).tolist()
    model.update()
    return layer_vars, (input_lower, input_upper)

def _build_solver_linear(model, layer, layer_name, layer_bounds, prev_vars, c, refine, max_value=None, min_value=None, max_power=None):
    global MULTIPROCESS_MODEL
    layer_vars = []

    weight = layer.weight.clone()
    bias = layer.bias.clone()
    W_plus = torch.clamp(weight, min=0.)
    W_minus = torch.clamp(weight, max=0.)

    assert torch.all(layer_bounds[0] <= layer_bounds[1])
    upper = W_plus @ layer_bounds[1] + W_minus @ layer_bounds[0] + bias
    lower = W_plus @ layer_bounds[0] + W_minus @ layer_bounds[1] + bias
    assert torch.all(lower <= upper)

    if c is not None:
        weight = c.squeeze(0).mm(weight)
        bias = c.squeeze(0).mm(bias.unsqueeze(-1)).view(-1)

    for neuron_idx in range(weight.size(0)):
        v = model.addVar(
            lb=lower[neuron_idx].item(),
            ub=upper[neuron_idx].item(),
            obj=0,
            vtype=grb.GRB.CONTINUOUS,
            name=f'lay{layer_name}_{neuron_idx}'
        )
        lin_expr = grb.LinExpr(weight[neuron_idx, :], prev_vars) + bias[neuron_idx].item()
        model.addConstr(lin_expr == v)
        layer_vars.append(v)
    model.update()

    if (c is None) and has_relu_var(model) and refine:
        candidates = [v_.VarName for v_ in layer_vars]
        if len(candidates):
            MULTIPROCESS_MODEL = model
            max_worker = min(len(candidates), os.cpu_count() // 2)
            with multiprocessing.Pool(max_worker) as pool:
                solver_results = pool.map(_mip_solver_worker, candidates, chunksize=1)
            MULTIPROCESS_MODEL = None

            for (var_name, var_lb, var_ub, refined) in solver_results:
                v_ = model.getVarByName(var_name)
                v_.lb = max(v_.lb, var_lb)
                v_.ub = min(v_.ub, var_ub)
            model.update()

    if max_value is not None:
        lower = torch.tensor([v.lb for v in layer_vars])
        upper = torch.tensor([v.ub for v in layer_vars])
        print(f'[before] max range = {(upper - lower).max().item():.4f}, min range = {(upper - lower).min().item():.4f}, avg range = {(upper - lower).mean().item():.4f}')
        if max_power is not None:
            model.addConstr(grb.quicksum(v*v for v in layer_vars) <= max_power, name="max_power_constraint")
            print(f'added max_power constraint')
        for i, v in enumerate(layer_vars):
            model.addConstr(v <= max_value[i], name=f"max_value_constraint_{i}")
            model.addConstr(v >= min_value[i], name=f"min_value_constraint_{i}")
        print(f'added max_value and min_value constraints')

        model.update()
        candidates = [v_.VarName for v_ in layer_vars]
        if len(candidates):
            MULTIPROCESS_MODEL = model
            max_worker = min(len(candidates), os.cpu_count() // 2)
            with multiprocessing.Pool(max_worker) as pool:
                solver_results = pool.map(_mip_solver_worker, candidates, chunksize=1)
            MULTIPROCESS_MODEL = None

            for (var_name, var_lb, var_ub, refined) in solver_results:
                v_ = model.getVarByName(var_name)
                v_.lb = max(v_.lb, var_lb)
                v_.ub = min(v_.ub, var_ub)
            model.update()

        lower = torch.tensor([v.lb for v in layer_vars])
        upper = torch.tensor([v.ub for v in layer_vars])
        print(f'[after]  max range = {(upper - lower).max().item():.4f}, min range = {(upper - lower).min().item():.4f}, avg range = {(upper - lower).mean().item():.4f}')

    lower = torch.tensor([v.lb for v in layer_vars])
    upper = torch.tensor([v.ub for v in layer_vars])
    assert torch.all(lower <= upper)
    return layer_vars, (lower, upper)

def _build_solver_relu(model, layer, layer_name, prev_vars):
    layer_vars = []

    zero_var = model.getVarByName('zero')

    vars_array = np.array(prev_vars)
    this_layer_shape = vars_array.shape

    count_a_vars = 0
    for neuron_idx, pre_var in enumerate(vars_array.reshape(-1)):
        if pre_var.lb >= 0:
            v = pre_var
        elif pre_var.ub <= 0:
            v = zero_var
        else:
            v = model.addVar(ub=pre_var.ub, lb=0, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'ReLU{layer_name}_{neuron_idx}')

            a = model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_name}_{neuron_idx}')

            model.addConstr(pre_var - pre_var.lb * (1 - a) >= v, name=f'ReLU{layer_name}_{neuron_idx}_a_0')
            model.addConstr(v >= pre_var, name=f'ReLU{layer_name}_{neuron_idx}_a_1')
            model.addConstr(pre_var.ub * a >= v, name=f'ReLU{layer_name}_{neuron_idx}_a_2')
            model.addConstr(v >= 0, name=f'ReLU{layer_name}_{neuron_idx}_a_3')

            count_a_vars += 1

        layer_vars.append(v)
    model.update()


    lower = torch.tensor([v.lb for v in layer_vars])
    upper = torch.tensor([v.ub for v in layer_vars])

    layer_vars = np.array(layer_vars).reshape(this_layer_shape).tolist()
    lower = lower.view(this_layer_shape)
    upper = upper.view(this_layer_shape)
    assert torch.all(lower <= upper)
    assert torch.all(lower >= 0.)

    return layer_vars, (lower, upper)

def _build_solver_flatten(model, layer, layer_name, prev_vars):
    layer_vars = np.array(prev_vars).reshape(-1).tolist()
    lower = torch.tensor([v.lb for v in layer_vars])
    upper = torch.tensor([v.ub for v in layer_vars])
    assert torch.all(lower <= upper)
    return layer_vars, (lower, upper)

def _build_solver_conv2d(model, layer, layer_name, layer_bounds, prev_vars, refine):
    global MULTIPROCESS_MODEL
    layer_vars = []
    prev_vars_array = np.array(prev_vars)
    prev_layer_shape = (1,) + prev_vars_array.shape
    assert len(prev_layer_shape) == 4, f'{len(prev_layer_shape)=}'

    weight = layer.weight.clone().detach().numpy()
    bias = layer.bias.clone().detach().numpy()

    conv_plus = nn.Conv2d(
        layer.in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,
        stride= layer.stride,
        padding=layer.padding)
    conv_minus = nn.Conv2d(
        layer.in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,
        stride= layer.stride,
        padding=layer.padding
    )
    conv_plus._parameters['weight'] = layer.weight.detach().clone().clamp(min=0)
    conv_minus._parameters['weight'] = layer.weight.detach().clone().clamp(max=0)
    conv_plus._parameters['bias'] = layer.bias.detach().clone() / 2.0
    conv_minus._parameters['bias'] = layer.bias.detach().clone() /2.0

    upper = conv_plus(layer_bounds[1]) + conv_minus(layer_bounds[0])
    lower = conv_plus(layer_bounds[0]) + conv_minus(layer_bounds[1])
    assert torch.all(lower <= upper)
    this_layer_shape = (1,) + upper.shape

    in_row_idx_mins = np.arange(this_layer_shape[2]) * layer.stride[0] - layer.padding[0]
    in_row_idx_maxs = in_row_idx_mins + weight.shape[2] - 1
    ker_row_mins = np.zeros(this_layer_shape[2], dtype=int)
    ker_row_maxs = np.ones(this_layer_shape[2], dtype=int) * weight.shape[2]
    ker_row_mins[in_row_idx_mins < 0] = -in_row_idx_mins[in_row_idx_mins < 0]
    ker_row_maxs[in_row_idx_maxs >= prev_layer_shape[2]] = ker_row_maxs[in_row_idx_maxs >= prev_layer_shape[2]] - in_row_idx_maxs[in_row_idx_maxs >= prev_layer_shape[2]] + prev_layer_shape[2] - 1
    in_row_idx_mins = np.maximum(in_row_idx_mins, 0)
    in_row_idx_maxs = np.minimum(in_row_idx_maxs, prev_layer_shape[2] - 1)

    in_col_idx_mins = np.arange(this_layer_shape[3]) * layer.stride[1] - layer.padding[1]
    in_col_idx_maxs = in_col_idx_mins + weight.shape[3] - 1
    ker_col_mins = np.zeros(this_layer_shape[3], dtype=int)
    ker_col_maxs = np.ones(this_layer_shape[3], dtype=int) * weight.shape[3]
    ker_col_mins[in_col_idx_mins < 0] = -in_col_idx_mins[in_col_idx_mins < 0]
    ker_col_maxs[in_col_idx_maxs >= prev_layer_shape[3]] = ker_col_maxs[in_col_idx_maxs >= prev_layer_shape[3]] - in_col_idx_maxs[in_col_idx_maxs >= prev_layer_shape[3]] + prev_layer_shape[3] - 1
    in_col_idx_mins = np.maximum(in_col_idx_mins, 0)
    in_col_idx_maxs = np.minimum(in_col_idx_maxs, prev_layer_shape[3] - 1)

    neuron_idx = 0
    for out_chan_idx in range(this_layer_shape[1]):
        out_chan_vars = []
        for out_row_idx in range(this_layer_shape[2]):
            out_row_vars = []
            ker_row_min, ker_row_max = ker_row_mins[out_row_idx], ker_row_maxs[out_row_idx]
            in_row_idx_min, in_row_idx_max = in_row_idx_mins[out_row_idx], in_row_idx_maxs[out_row_idx]
            for out_col_idx in range(this_layer_shape[3]):
                ker_col_min, ker_col_max = ker_col_mins[out_col_idx], ker_col_maxs[out_col_idx]
                in_col_idx_min, in_col_idx_max = in_col_idx_mins[out_col_idx], in_col_idx_maxs[out_col_idx]
                lin_expr = bias[out_chan_idx]
                for in_chan_idx in range(weight.shape[1]):
                    coeffs = weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                    gvars = prev_vars_array[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                    lin_expr += grb.LinExpr(coeffs, gvars)

                var = model.addVar(
                    lb=lower[out_chan_idx, out_row_idx, out_col_idx].item(),
                    ub=upper[out_chan_idx, out_row_idx, out_col_idx].item(),
                    obj=0,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f'lay{layer_name}_{neuron_idx}'
                )
                model.addConstr(lin_expr == var, name=f'lay{layer_name}_{neuron_idx}_eq')
                neuron_idx += 1

                out_row_vars.append(var)
            out_chan_vars.append(out_row_vars)
        layer_vars.append(out_chan_vars)

    assert np.array(layer_vars).shape == this_layer_shape[1:]
    model.update()

    if has_relu_var(model) and refine:
        candidates = [v_.VarName for v_ in np.array(layer_vars).reshape(-1) if v_.lb * v_.ub < 0]
        if len(candidates):
            MULTIPROCESS_MODEL = model
            max_worker = min(len(candidates), os.cpu_count() // 2)
            with multiprocessing.Pool(max_worker) as pool:
                solver_results = pool.map(_mip_solver_worker, candidates, chunksize=1)
            MULTIPROCESS_MODEL = None

            for (var_name, var_lb, var_ub, refined) in solver_results:
                if refined:
                    v_ = model.getVarByName(var_name)
                    v_.lb = max(v_.lb, var_lb)
                    v_.ub = min(v_.ub, var_ub)
            model.update()


    lower = torch.tensor([v.lb for v in np.array(layer_vars).reshape(-1)]).view(this_layer_shape)
    upper = torch.tensor([v.ub for v in np.array(layer_vars).reshape(-1)]).view(this_layer_shape)
    assert torch.all(lower <= upper)
    return layer_vars, (lower, upper)

def build_milp_solver(net,
                      input_lower, input_upper,
                      timeout=10.0, c=None, name='',
                      refine=True, return_bounds=False,
                      max_power=None,
                      max_value=None,
                      min_value=None):
    solver = grb.Model(name)
    solver.setParam('OutputFlag', False)
    solver.setParam('Threads', 1)
    solver.setParam("FeasibilityTol", 1e-5)
    solver.setParam('MIPGap', 1e-2)
    solver.setParam('MIPGapAbs', 1e-2)
    solver.setParam('TimeLimit', timeout)

    gurobi_vars = []

    solver.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')

    input_vars, layer_bounds = _build_solver_input(
        model=solver,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    gurobi_vars.append(input_vars)

    pre_relu_names = []
    relu_names = []

    layers = list(net.children())
    prev_name = None
    for layer_name, layer in enumerate(layers):
        if isinstance(layer,  nn.Linear):
            new_layer_gurobi_vars, new_layer_bounds = _build_solver_linear(
                model=solver,
                layer=layer,
                layer_name=layer_name,
                layer_bounds=layer_bounds,
                prev_vars=gurobi_vars[-1],
                c=c if layer == layers[-1] else None,
                refine=refine,
                min_value=min_value if layer == layers[-1] else None,
                max_value=max_value if layer == layers[-1] else None,
                max_power=max_power if layer == layers[-1] else None,
            )
            prev_name = layer_name
        elif isinstance(layer, nn.ReLU):
            new_layer_gurobi_vars, new_layer_bounds = _build_solver_relu(
                model=solver,
                layer=layer,
                layer_name=layer_name,
                prev_vars=gurobi_vars[-1],
            )
            pre_relu_names.append((prev_name, np.array(new_layer_gurobi_vars).size))
            relu_names.append(layer_name)
        elif isinstance(layer, nn.Flatten) or 'Flatten()' in str(layer):
            new_layer_gurobi_vars, new_layer_bounds = _build_solver_flatten(
                model=solver,
                layer=layer,
                layer_name=layer_name,
                prev_vars=gurobi_vars[-1],
            )
        elif isinstance(layer,  nn.Conv2d):
            new_layer_gurobi_vars, new_layer_bounds = _build_solver_conv2d(
                model=solver,
                layer=layer,
                layer_name=layer_name,
                layer_bounds=layer_bounds,
                prev_vars=gurobi_vars[-1],
                refine=refine,
            )
            prev_name = layer_name
        else:
            print(layer, type(layer))
            raise NotImplementedError(layer)
        gurobi_vars.append(new_layer_gurobi_vars)
        layer_bounds = (new_layer_bounds[0].clone(), new_layer_bounds[1].clone())


    ub, lb = [], []
    last_prev_relu_name, last_prev_relu_size = pre_relu_names[-1]
    for neuron_idx in range(last_prev_relu_size):
        v = solver.getVarByName(f'lay{last_prev_relu_name}_{neuron_idx}')
        assert v is not None
        solver.setObjective(v, grb.GRB.MAXIMIZE)
        solver.reset()
        solver.optimize()
        if solver.status != 2:
            print(f'[build_milp_solver] {solver.status=}')
        ub.append(solver.ObjBound)
        solver.setObjective(v, grb.GRB.MINIMIZE)
        solver.reset()
        solver.optimize()
        if solver.status != 2:
            print(f'[build_milp_solver] {solver.status=}')
        lb.append(solver.ObjBound)
    solver.update()
    return solver, gurobi_vars, pre_relu_names, relu_names, lb, ub, layer_bounds[0], layer_bounds[1]
