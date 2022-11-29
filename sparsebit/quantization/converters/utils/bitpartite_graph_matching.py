from queue import Queue


class Hungary(object):
    """bitpartite graph maximum matching for ``InputMatchingType.SUBSET`` type"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.match = [0] * (n + 1)
        self.hungary_mrk = [0] * (n + 1)
        self.hungary_pre = [0] * (n + 1)
        self.edges = [[]] * (n + 1)
        self.q = Queue(maxsize=n)

    def add_edge(self, L, R):
        self.edges[L + 1].append(R + 1)

    def apply(self):
        tot = 0
        for i in range(1, self.n + 1):
            if not self.match[i]:
                while not self.q.empty():
                    self.q.get()
                self.q.put(i)
                self.hungary_pre[i] = 0
                flg = False
                while not self.q.empty() and not flg:
                    now = self.q.get()
                    for nex in self.edges[now]:
                        if flg:
                            break
                        if self.hungary_mrk[nex] != i:
                            self.hungary_mrk[nex] = i
                            self.q.put(self.match[nex])
                            if self.match[nex]:
                                self.hungary_pre[self.match[nex]] = now
                            else:
                                flg = True
                                uu = now
                                vv = nex
                                while uu:
                                    tt = self.match[uu]
                                    self.match[uu] = vv
                                    self.match[vv] = uu
                                    uu = self.hungary_pre[uu]
                                    vv = tt
            if self.match[i]:
                tot += 1
        return tot
