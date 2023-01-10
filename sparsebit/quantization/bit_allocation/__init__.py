from .greedy import bit_allocation_by_greedy

# from .hawqv3 import bit_allocation_by_hawqv3

method_factory = {
    "greedy": bit_allocation_by_greedy,
    # "hawq": bit_allocation_by_hawqv3,
}


def bit_allocation(qmodel, data):
    method_factory[qmodel.cfg.SCHEDULE.BIT_ALLOCATION.METHOD](qmodel, data)
