import pulp
from sparsebit.quantization.modules import QConv2d, QLinear
from sparsebit.quantization.modules.matmul import MatMul


def ilp_search(
    qmodel,
    perturbations_convlinear,
    perturbations_matmul,
    bops_limitation,
    memory_limitation,
):
    print("Starting weight ILP search!")
    weight_layer_names = list(perturbations_convlinear.keys())
    weight_layer_modules = [getattr(qmodel.model, name) for name in weight_layer_names]
    matmul_layer_names = list(perturbations_matmul.keys())
    matmul_layer_modules = [getattr(qmodel.model, name) for name in matmul_layer_names]
    weight_bit_choices = qmodel.cfg.W.OBSERVER.BIT_CHOICES
    feature_bit_choices = qmodel.cfg.A.OBSERVER.BIT_CHOICES

    # Define problem
    problem = pulp.LpProblem("weight bit allocation", pulp.LpMinimize)
    var_convlinear = pulp.LpVariable.matrix(
        "var_convlinear",
        (
            range(len(weight_layer_names)),
            range(len(weight_bit_choices)),
            range(len(feature_bit_choices)),
        ),
        0,
        1,
        pulp.LpInteger,
    )
    var_matmul = pulp.LpVariable.matrix(
        "var_matmul",
        (
            range(len(matmul_layer_names)),
            range(len(feature_bit_choices)),
            range(len(feature_bit_choices)),
        ),
        0,
        1,
        pulp.LpInteger,
    )
    target_values = [
        perturbations_convlinear[weight_layer_names[i]][weight_bit_choices[j]][
            feature_bit_choices[k]
        ]
        * var_convlinear[i][j][k]
        for i in range(len(weight_layer_names))
        for j in range(len(weight_bit_choices))
        for k in range(len(feature_bit_choices))
    ]
    target_values.extend(
        [
            perturbations_matmul[matmul_layer_names[i]][feature_bit_choices[j]][
                feature_bit_choices[k]
            ]
            * var_matmul[i][j][k]
            for i in range(len(matmul_layer_names))
            for j in range(len(feature_bit_choices))
            for k in range(len(feature_bit_choices))
        ]
    )
    problem += pulp.lpSum(target_values)

    # Set limitations
    # only a single bit choice is chosen for each layer
    for i in range(len(weight_layer_names)):
        problem += pulp.lpSum(var_convlinear[i]) == 1
    for i in range(len(matmul_layer_names)):
        problem += pulp.lpSum(var_matmul[i]) == 1
    # add memory limitation
    total_memory = [
        weight_layer_modules[i].weight.numel()
        * weight_bit_choices[j]
        / 8
        * var_convlinear[i][j][k]
        for i in range(len(weight_layer_names))
        for j in range(len(weight_bit_choices))
        for k in range(len(feature_bit_choices))
    ]
    problem += pulp.lpSum(total_memory) <= memory_limitation
    # add max BOPS limitation
    total_bops = [
        weight_layer_modules[i].flops
        * weight_bit_choices[j]
        * feature_bit_choices[k]
        * var_convlinear[i][j][k]
        for i in range(len(weight_layer_names))
        for j in range(len(weight_bit_choices))
        for k in range(len(feature_bit_choices))
    ]
    total_bops.extend(
        [
            matmul_layer_modules[i].flops
            * feature_bit_choices[j]
            * feature_bit_choices[k]
            * var_matmul[i][j][k]
            for i in range(len(matmul_layer_names))
            for j in range(len(feature_bit_choices))
            for k in range(len(feature_bit_choices))
        ]
    )
    problem += pulp.lpSum(total_bops) <= bops_limitation

    # Calculate results
    problem.solve(pulp.PULP_CBC_CMD(timeLimit=180))
    bit_allocated = {}
    print("Status: " + pulp.LpStatus[problem.status])
    if pulp.LpStatus[problem.status] != "Optimal":
        raise ValueError("Integer Linear Programming no solution!")
    for v in problem.variables():
        if "__" in v.name:
            continue
        print(v.name)
        _, var_type, layer_idx, bit_idx_0, bit_idx_1 = v.name.split("_")
        layer_idx = int(layer_idx)
        bit_idx_0 = int(bit_idx_0)
        bit_idx_1 = int(bit_idx_1)
        # print(v)
        # print(v.varValue)
        if v.varValue > 0.5:
            if var_type == "convlinear":
                bit_allocated[weight_layer_names[layer_idx]] = {
                    "w": weight_bit_choices[bit_idx_0],
                    "f": feature_bit_choices[bit_idx_1],
                }
            elif var_type == "matmul":
                bit_allocated[matmul_layer_names[layer_idx]] = {
                    "f0": feature_bit_choices[bit_idx_0],
                    "f1": feature_bit_choices[bit_idx_1],
                }
            else:
                raise NotImplementedError
    print(len(problem.variables()))
    print(bit_allocated)

    return bit_allocated
