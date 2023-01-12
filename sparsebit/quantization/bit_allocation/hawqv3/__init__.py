from .utils import *
from .ilp import ilp_search
from .perturbations import get_perturbations
from .bit_allocation import *


def bit_allocation_by_hawqv3(qmodel, data, label):
    target_w_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_WEIGHT_BIT_TARGET
    target_a_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_FEATURE_BIT_TARGET
    (
        bops_limitation,
        bops_limitation_for_search,
        memory_limitation,
    ) = calc_flops_and_limitations(qmodel, target_w_bit, target_a_bit)
    print("Total BOPs limitation:", str(bops_limitation / 1e9), "GBOPs")
    print("Total memory limitation:", str(memory_limitation / (2**20)), "MB")
    perturbations_conv_linear, perturbations_matmul = get_perturbations(qmodel, data, label)
    bit_allocated = ilp_search(
        qmodel,
        perturbations_conv_linear,
        perturbations_matmul,
        bops_limitation_for_search,
        memory_limitation,
    )
    import ipdb
    ipdb.set_trace()
    #weight cfg
    print("Weight bit cfg:")
    for n, v in bit_allocated.items():
        if "w" in v.keys():
            print('"{}": ["QUANTIZER.BIT", {}],'.format(n, v["w"]))
    print()
    print("Featuret bit cfg:")
    #feature cfg
    for n, v in bit_allocated.items():
        for k, bit in v.items():
            if k!= "w":
                print('"{}": ["QUANTIZER.BIT", {}],'.format(n, bit))

    bit_allocation(qmodel, bit_allocated)

    allocated_bops, allocated_memory = calc_final_bops_and_memory(qmodel)
    print("Total BOPs limitation:", str(bops_limitation / 1e9), "GBOPs")
    print("Total memory limitation:", str(memory_limitation / (2**20)), "MB")
    print("Total BOPs allocated:", str(allocated_bops / 1e9), "GBOPs")
    print("Total memory allocated:", str(allocated_memory / (2**20)), "MB")

