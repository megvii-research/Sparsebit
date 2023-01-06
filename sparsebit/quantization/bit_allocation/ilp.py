import pulp
from sparsebit.quantization.modules import QConv2d, QLinear


def feature_ilp_search(qmodel, perturbations, bops_limitation):
    print("Starting feature ILP search!")
    layer_names = list(perturbations.keys())
    layer_modules = [getattr(qmodel.model, name) for name in layer_names]
    bit_choices = qmodel.cfg.A.OBSERVER.BIT_CHOICES

    # Define problem
    problem = pulp.LpProblem("feature bit allocation", pulp.LpMinimize)
    var = pulp.LpVariable.matrix(
        "feature",
        (range(len(layer_names)), range(len(bit_choices))),
        0,
        1,
        pulp.LpInteger,
    )
    target_values = [
        perturbations[layer_names[i]][bit_choices[j]] * var[i][j]
        for i in range(len(layer_names))
        for j in range(len(bit_choices))
    ]
    problem += pulp.lpSum(target_values)

    # Set limitations
    for i in range(
        len(layer_names)
    ):  # only a single bit choice is chosen for each layer
        problem += pulp.lpSum(var[i]) == 1
    # add max BOPS limitation
    total_bops = [
        layer_modules[i].flops * bit_choices[j] * 8 * var[i][j]
        for i in range(len(layer_names))
        for j in range(len(bit_choices))
    ]
    problem += pulp.lpSum(total_bops) <= bops_limitation + 1

    # Calculate results
    problem.solve(pulp.PULP_CBC_CMD(timeLimit=180))
    bit_allocated = {}
    print("Status: " + pulp.LpStatus[problem.status])
    if pulp.LpStatus[problem.status] != "Optimal":
        raise ValueError("Integer Linear Programming no solution!")
    for v in problem.variables():
        if "__" in v.name:
            continue
        _, layer_idx, bit_idx = v.name.split("_")
        layer_idx = int(layer_idx)
        bit_idx = int(bit_idx)
        # print(v)
        # print(v.varValue)
        if v.varValue > 0.5:
            bit_allocated[layer_names[layer_idx]] = bit_choices[bit_idx]
    print(len(problem.variables()))
    print(bit_allocated)

    return bit_allocated


def weight_ilp_search(qmodel, perturbations, bops_limitation, memory_limitation):
    print("Starting weight ILP search!")
    layer_names = list(perturbations.keys())
    layer_modules = [getattr(qmodel.model, name) for name in layer_names]
    bit_choices = qmodel.cfg.W.OBSERVER.BIT_CHOICES

    # Define problem
    problem = pulp.LpProblem("weight bit allocation", pulp.LpMinimize)
    var = pulp.LpVariable.matrix(
        "weight",
        (range(len(layer_names)), range(len(bit_choices))),
        0,
        1,
        pulp.LpInteger,
    )
    target_values = [
        perturbations[layer_names[i]][bit_choices[j]] * var[i][j]
        for i in range(len(layer_names))
        for j in range(len(bit_choices))
    ]
    problem += pulp.lpSum(target_values)

    # Set limitations
    for i in range(
        len(layer_names)
    ):  # only a single bit choice is chosen for each layer
        problem += pulp.lpSum(var[i]) == 1
    # add memory limitation
    total_memory = [
        layer_modules[i].weight.numel() * bit_choices[j] / 8 * var[i][j]
        for i in range(len(layer_names))
        for j in range(len(bit_choices))
    ]
    problem += pulp.lpSum(total_memory) <= memory_limitation
    # add max BOPS limitation
    total_bops = [
        layer_modules[i].flops
        * layer_modules[i].input_quantizer.bit
        * bit_choices[j]
        * var[i][j]
        for i in range(len(layer_names))
        for j in range(len(bit_choices))
    ]
    problem += pulp.lpSum(total_bops) <= bops_limitation + 1

    # Calculate results
    problem.solve(pulp.PULP_CBC_CMD(timeLimit=180))
    bit_allocated = {}
    print("Status: " + pulp.LpStatus[problem.status])
    if pulp.LpStatus[problem.status] != "Optimal":
        raise ValueError("Integer Linear Programming no solution!")
    for v in problem.variables():
        if "__" in v.name:
            continue
        _, layer_idx, bit_idx = v.name.split("_")
        layer_idx = int(layer_idx)
        bit_idx = int(bit_idx)
        # print(v)
        # print(v.varValue)
        if v.varValue > 0.5:
            bit_allocated[layer_names[layer_idx]] = bit_choices[bit_idx]
    print(len(problem.variables()))
    print(bit_allocated)

    return bit_allocated
