from .utils import *
from .ilp import weight_ilp_search, feature_ilp_search
from .metric import metric_factory
from .bit_allocation import *


def bit_allocation(qmodel, data):
    target_w_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_WEIGHT_BIT_TARGET
    target_a_bit = qmodel.cfg.SCHEDULE.BIT_ALLOCATION.AVG_FEATURE_BIT_TARGET
    (
        bops_limitation,
        bops_limitation_for_feature_search,
        bops_limitation_for_weight_search,
        memory_limitation,
    ) = calc_flops_and_limitations(qmodel, target_w_bit, target_a_bit)
    feature_perturbations, weight_perturbations = metric_factory["greedy"](qmodel, data)
    feature_bit_allocated = feature_ilp_search(
        qmodel, feature_perturbations, bops_limitation_for_feature_search
    )
    feature_bit_allocation(qmodel, feature_bit_allocated)
    weight_bit_allocated = weight_ilp_search(
        qmodel,
        weight_perturbations,
        bops_limitation_for_weight_search,
        memory_limitation,
    )
    weight_bit_allocation(qmodel, weight_bit_allocated)

    allocated_bops, allocated_memory = calc_final_bops_and_memory(qmodel)
    print("Total BOPs limitation:", str(bops_limitation / 1e9), "GBOPs")
    print("Total memory limitation:", str(memory_limitation / 1e6), "MB")
    print("Total BOPs allocated:", str(allocated_bops / 1e9), "GBOPs")
    print("Total memory allocated:", str(allocated_memory / 1e6), "MB")
