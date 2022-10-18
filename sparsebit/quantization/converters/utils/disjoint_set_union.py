from typing import List, Union, Callable


class DSU(object):
    """disjoint-set union, with nodes indexing from 0 to N-1
    w/i and w/o value is supported
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
        self.fa = [i for i in range(self.n)]

    def find(self, x: int) -> int:
        if self.fa[x] == x:
            return x
        if self.with_value:
            tmp_fa = self.fa[x]
            self.fa[x] = self.find(self.fa[x])
            if self.cmp(self.value[tmp_fa], self.value[x]):
                self.value[x] = self.value[tmp_fa]
        else:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def merge(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)
        self.fa[x] = y
