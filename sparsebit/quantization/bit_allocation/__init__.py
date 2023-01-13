from .hawqv3 import bit_allocation_by_hawqv3
from .greedy import bit_allocation_by_greedy
from .hawqv3_new import bit_allocation_by_hawqv3_new

method_factory = {
    "hawqv3": bit_allocation_by_hawqv3,
    "greedy": bit_allocation_by_greedy,
    "hawqv3_new": bit_allocation_by_hawqv3_new,
}


def bit_allocation(qmodel, calib_loader):
    method_factory[qmodel.cfg.SCHEDULE.BIT_ALLOCATION.METHOD](qmodel, calib_loader)
