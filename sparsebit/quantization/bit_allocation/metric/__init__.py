from .greedy import get_perturbations as get_perturbations_by_greedy
from .hawq import *

metric_factory = {
    "greedy": get_perturbations_by_greedy,
}