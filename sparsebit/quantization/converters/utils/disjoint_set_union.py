from typing import List, Union, Callable


class DSU(object):
    """disjoint-set union, with nodes indexing from 0 to N-1
    w/ and w/o value is supported
    path compression optimization applied
    """

    def __init__(self, nums: int, value: Union[List, None], cmp: Union[Callable, None]):
        self.n = nums
        assert (
            isinstance(value, list) and len(value) == nums or value is None
        ), "Unknown format"
        self.with_value = value is not None
        if self.with_value:
            self.value = value
            assert cmp is not None, "Unknown format"
            self.cmp = cmp
        self.parents = [i for i in range(self.n)]

    def find(self, x: int) -> int:
        if self.parents[x] == x:
            return x
        if self.with_value:
            tmp_parents = self.parents[x]
            self.parents[x] = self.find(self.parents[x])
            if self.cmp(self.value[tmp_parents], self.value[x]):
                self.value[x] = self.value[tmp_parents]
        else:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def merge(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)
        self.parents[x] = y
