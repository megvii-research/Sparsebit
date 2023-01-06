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
        memory_limitation,
    ) = calc_flops_and_limitations(qmodel.model, target_w_bit, target_a_bit)
    feature_perturbations, weight_perturbations = metric_factory["greedy"](qmodel, data)
    feature_bit_allocated = feature_ilp_search(
        qmodel, feature_perturbations, bops_limitation_for_feature_search
    )
    feature_bit_allocation(qmodel, feature_bit_allocated)
    weight_bit_allocated = weight_ilp_search(
        qmodel, weight_perturbations, bops_limitation, memory_limitation
    )
    weight_bit_allocation(qmodel, weight_bit_allocated)
