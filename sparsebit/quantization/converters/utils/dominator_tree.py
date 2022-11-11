from .disjoint_set_union import DSU


class DominatorTree(object):
    def __init__(self, nums: int):
        self.n = nums
        self.cnt = 0
        self.edges = [[] for i in range(self.n)]
        self.inv_edges = [[] for i in range(self.n)]
        self.finish_flag = False
        self.targets = [[] for i in range(self.n)]

    def add_edge(self, u: int, v: int):
        self.edges[u].append(v)
        self.inv_edges[v].append(u)

    def dfs(self, x: int, fa: int):
        self.dfn[x] = self.cnt
        self.prev[x] = fa
        self.idx[self.cnt] = x
        self.cnt += 1
        for nex in self.edges[x]:
            if self.dfn[nex] == self.n:
                self.dfs(nex, x)

    def build(self):
        assert (
            not self.finish_flag
        ), "Dominator tree already finished building. \
        Do not build again, try to create a new one instead."
        self.dfn = [self.n] * self.n
        self.prev = [self.n] * self.n
        self.idx = [self.n] * self.n
        self.sdom = [i for i in range(self.n)]
        self.idom = [self.n] * self.n
        self.dfs(self.n - 1, -1)

    def solve(self):
        self.build()
        cmp = lambda x, y: self.dfn[self.sdom[x]] < self.dfn[self.sdom[y]]  # noqa: E731
        valued_dsu = DSU(nums=self.n, value=[i for i in range(self.n)], cmp=cmp)
        for i in range(self.n - 1, 0, -1):
            x = self.idx[i]
            res = self.n - 1
            for y in self.inv_edges[x]:
                if self.dfn[y] < self.dfn[x]:
                    res = min(res, self.dfn[y])
                else:
                    valued_dsu.find(y)
                    res = min(res, self.dfn[self.sdom[valued_dsu.value[y]]])
            self.sdom[x] = self.idx[res]
            valued_dsu.merge(x, self.prev[x])

        # it is guaranteed sdom[x] and idom[x] is ancestor of x in dfs tree
        # k = argmin i min{dfn[sdom[i]]}, for i in sdom[x] -> x chain (exclusive)
        # idom[x] = sdom[x] if sdom[x] == sdom[k] else idom[k]
        for i in range(self.n):
            x = self.idx[i]

            if self.prev[x] == -1 or self.sdom[x] == self.prev[x]:
                self.idom[x] = self.sdom[x]
                continue

            cur = self.prev[x]
            min_k = cur
            # FIXME: use O(nlogn) upward algorithm
            while cur != self.sdom[x]:
                if self.dfn[self.sdom[cur]] < self.dfn[self.sdom[min_k]]:
                    min_k = cur
                cur = self.prev[cur]
            self.idom[x] = (
                self.sdom[x] if self.sdom[x] == self.sdom[min_k] else self.idom[min_k]
            )
        # edges (idom[x], x) is a tree, except for (idom[n-1]=n-1, n-1) is loop
        self.finish_flag = True
        self.rebuild()

    def rebuild(self):
        assert self.finish_flag
        sons = [0] * self.n
        masks = [0] * self.n

        def dfs_get_sons(x: int):
            sons[x] += 1
            for i in self.targets[x]:
                dfs_get_sons(i)
                sons[x] += sons[i]

        def dfs_sort_subtrees(x: int):
            for i in self.edges[x]:
                masks[x] |= 1 << i
            for i in self.targets[x]:
                dfs_sort_subtrees(i)
                masks[x] |= masks[i]
            _len = len(self.targets[x])
            # sort self.targets[x] in reversed lambda i: sons[i] order
            # while keeping topological
            new_targets = []
            in_degrees = [0] * _len
            ed = [[] for i in range(_len)]
            q = set()
            for i in range(_len):
                for j in range(_len):
                    if i != j and (
                        (int(1 << self.targets[x][i]) & masks[self.targets[x][j]]) != 0
                    ):
                        ed[j].append(i)
                        in_degrees[i] += 1
                if in_degrees[i] == 0:
                    q.add((sons[self.targets[x][i]], i))
            while q:
                num_sons, top = max(q)
                q.remove((num_sons, top))
                new_targets.append(self.targets[x][top])
                for nex in ed[top]:
                    in_degrees[nex] -= 1
                    if in_degrees[nex] == 0:
                        q.add((sons[self.targets[x][nex]], nex))
            self.targets[x] = new_targets

        for i in range(1, self.n):
            x = self.idx[i]
            self.targets[self.idom[x]].append(x)

        dfs_get_sons(self.n - 1)
        dfs_sort_subtrees(self.n - 1)
