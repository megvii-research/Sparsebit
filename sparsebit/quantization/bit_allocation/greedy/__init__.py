from .utils import *
from .ilp import ilp_search
from .perturbations import get_perturbations
from .bit_allocation import *


def bit_allocation_by_greedy(qmodel, data):
    target_w_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_WEIGHT_BIT_TARGET
    target_a_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_FEATURE_BIT_TARGET
    (
        bops_limitation,
        bops_limitation_for_search,
        memory_limitation,
    ) = calc_flops_and_limitations(qmodel, target_w_bit, target_a_bit)
    perturbations_conv_linear, perturbations_matmul = get_perturbations(qmodel, data)
    bit_allocated = ilp_search(
        qmodel,
        perturbations_conv_linear,
        perturbations_matmul,
        bops_limitation_for_search,
        memory_limitation,
    )
    bit_allocation(qmodel, bit_allocated)

    allocated_bops, allocated_memory = calc_final_bops_and_memory(qmodel)
    print("Total BOPs limitation:", str(bops_limitation / 1e9), "GBOPs")
    print("Total memory limitation:", str(memory_limitation / 1e6), "MB")
    print("Total BOPs allocated:", str(allocated_bops / 1e9), "GBOPs")
    print("Total memory allocated:", str(allocated_memory / 1e6), "MB")
