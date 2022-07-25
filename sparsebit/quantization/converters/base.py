from typing import List, Tuple, Dict, Union, Callable
from enum import Enum
from queue import Queue
import torch

from .prune import PruneGraph


class ReplacePatternTimer(object):
    """一个计数器，在子图替换中，对于同一个变换的多次调用能生成不同的序号。"""

    def __init__(self):
        self.idx = 0

    def get_idx(self):
        idx = self.idx
        self.idx += 1
        return idx


class InputMatchType(Enum):
    """在子图替换中,可以使用的MatcherNode输入匹配类型。

    Args:
        ALL: 完全按照MatcherNode给出的顺序匹配torch.fx.Node的输入。
        SUBSET:
            允许MatcherNode给出的输入是torch.fx.Node的输入的子集。
            即允许任意交换MatcherNode的输入顺序,并允许补None。

    .. Warning::
        目前SUBSET功能未实装。

    """

    ALL = 0
    SUBSET = 1


class ReplaceStrategyInner(object):
    """替换子图的替换策略基类。

    Args:
        multiple_match (bool): 是否要在发现多个不相交匹配时同时对它们做替换。
        repeat (bool): 是否要循环到网络中不再出现子图为止。
    """

    def __init__(self, multiple_match: bool, repeat: bool):
        self.multiple_match = multiple_match
        self.repeat = repeat


class ReplaceStrategy(object):
    """替换子图的替换策略。

    Args:

        APPLY_NOINTERSECT_REPEAT (ReplaceStrategyInner):
            找到多个不相交的匹配并替换，循环到不再出现为止。
        APPLY_NOINTERSECT (ReplaceStrategyInner):
            找到多个不相交的匹配替换，只运行一次。
        APPLY_ONE_REPEAT (ReplaceStrategyInner):
            找到第一个匹配替换，循环到不再出现为止。
        APPLY_ONE (ReplaceStrategyInner):
            找到第一个匹配替换，只运行一次。
    """

    APPLY_NOINTERSECT_REPEAT = ReplaceStrategyInner(True, True)
    APPLY_NOINTERSECT = ReplaceStrategyInner(True, False)
    APPLY_ONE_REPEAT = ReplaceStrategyInner(False, True)
    APPLY_ONE = ReplaceStrategyInner(False, False)


class MatcherNode(object):
    """Matcher中的node匹配基类。

    Args:
        name (str):
            在一个Matcher中不可重复。用于在Matcher子图中标识每个node。
        inputs (List[Union[str, None]]):
            该点的输入节点名,在Matcher子图中表示连接关系。

            如果没有输入，留空: ``[]`` 。

            如果有输入的连接关系但不希望放在匹配里或者写成一个新MatcherNode,使用 ``None`` 。
        op_type (List[Union[Callable, object]]):
            一个允许类型的list,表示这个op可以是哪些类型。
        checker (Callable):
            单个op (与对应module)的自定义匹配条件,输入(node_op, node_module),输出bool。

            .. Note::

                一个示例如下::

                >>> lambda cat, module: module.axis == 1

        input_match_type (int):
            输入一个InputMatchType,在Matcher子图中表示 ``inputs`` 的匹配要求。
            参见 (InputMatchType) 。

    在以下条件满足时,可以匹配graph中的一个node:

    - op_type对齐。

    - checker中的条件满足。

    如果需要匹配子图,那么还需要满足 ``inputs`` 和 ``input_match_type`` 规定的子图连接关系限制。

    """

    def __init__(
        self,
        name: str,
        inputs: List[Union[str, None]],
        op_type: List[Union[Callable, object]],
        checker: Callable = lambda x, module: True,
        input_match_type=InputMatchType.ALL,
    ):
        self.name = name
        self.inputs = inputs
        self.op_type = op_type
        self.checker = checker
        self.input_match_type = input_match_type

    def __repr__(self):
        return "MatcherNode(name={}, inputs={}, type={}, checker={}, match_type={})".format(
            self.name,
            self.inputs,
            self.op_type,
            self.checker,
            InputMatchType(self.input_match_type).name,
        )


class SubgraphMatcher(object):
    """网络中子图匹配类,执行匹配子图相关部分的工作。

    Args:
        ops (List[MatcherNode]):
            一个构建好连接关系的MatcherNode list。
            需要满足拓扑序,即一个MatcherNode引用到的其他输入点都在它前面。
        joint_checkers (List[Tuple[Tuple[str], Callable]]):
            多个MatcherNode的联合自定义匹配条件 list。
            每个联合匹配条件是一个形如 `` ((names), joint_checker) `` 的形式。
            ``joint_checker`` 输入(name1, name2, ..., modules:dict),输出bool。
        matching_strategy (ReplaceStrategy):
            调用Matcher的子图替换使用的策略,根据这个策略决定是否需要同时返回多个匹配。
    """

    def __init__(
        self,
        ops: List[MatcherNode],
        joint_checkers: List[Tuple[Tuple[str], Callable]],
        matching_strategy=ReplaceStrategy.APPLY_NOINTERSECT_REPEAT,
    ):
        self.match_ops = ops
        self.matching_strategy = matching_strategy

        out_degrees = {}
        self.names_dict = {}
        for match_op in self.match_ops:
            self.names_dict[match_op.name] = match_op
            out_degrees[match_op.name] = 0
            for inp_op in match_op.inputs:
                if inp_op is None:
                    continue
                assert inp_op in self.names_dict, "matcher ops not in topological order"
                out_degrees[inp_op] += 1
        output_node_num = 0
        self.anchor_name = None
        for k, v in out_degrees.items():
            # FIXME: 目前不支持所有共享祖先节点的pattern,比如skip connection,限制每个点出度为0或1。
            assert (
                v == 0 or v == 1
            ), "only support simple graph with each node having one successor now"
            if v == 0:
                output_node_num += 1
                self.anchor_name = k
        assert output_node_num == 1, "matcher only support one output node in subgraph"

        self.names_dict = {i.name: i for i in self.match_ops}
        assert len(self.names_dict) == len(
            self.match_ops
        ), "duplicated names found in matcher"

        for joint_checker in joint_checkers:
            required_ops, joint_checker_func = joint_checker
            for required_op in required_ops:
                assert required_op in self.names_dict, "input must be in subgraph"
        # HACK: 目前先在最后一个点统一判断所有联合checkers
        # FIXME: 反向建边计算DAG的最近公共祖先，在从下往上dfs过程中可以提早判断联合checkers，用来剪枝
        self.joint_checkers = {}
        self.joint_checkers[self.anchor_name] = joint_checkers

    def apply(
        self, m: torch.fx.GraphModule
    ) -> List[Tuple[Dict[str, torch.fx.Node], Dict[str, torch.nn.Module]]]:
        """执行子图匹配。

        Args:
            m (torch.fx.GraphModule): 需要匹配的网络。

        Returns:
            List[Tuple[Dict[str, torch.fx.Node], Dict[str, torch.nn.Module]]]:
                返回一个匹配list,每个匹配是 ``(nodes_dict, modules_dict)`` 的形式。

                其中:

                - nodes_dict为一个MatcherNode名称到网络中node的映射dict。
                - modules_dict为一个MatcherNode名称到网络中module的映射dict。
        """
        named_modules = dict(m.named_modules())
        graph = m.graph
        operators = {i.name: i for i in graph.nodes}
        optional_ops: Dict[set] = {}
        # 保存op_type类型对应的op名称集合
        for op_name, op in operators.items():
            real_op = get_op_modules([op], m, named_modules, type="class")[0]
            if real_op not in optional_ops:
                optional_ops[real_op] = set()
            optional_ops[real_op].add(op.name)
        """
        检查op_type + matcher输入部分子图结构 + checker对上的节点(粗筛)
        只有这些点**可能**匹配
        """
        matching_ops: Dict[set] = {}
        for match_op in self.match_ops:
            matching_ops[match_op.name] = set()
            op_with_correct_func = set(
                [j for i in match_op.op_type for j in optional_ops.get(i, set())]
            )  # op_type检查
            for op_name in op_with_correct_func:
                op = operators[op_name]
                op_module = get_op_modules([op], m, named_modules, type="object")[0]
                # checker检查
                if not match_op.checker(op, module=op_module):
                    continue

                # 输入节点检查
                # 输入的匹配方式是完全匹配, match_op.inputs == op.all_input_nodes
                if match_op.input_match_type == InputMatchType.ALL:
                    if len(match_op.inputs) != len(op.all_input_nodes):
                        continue

                    # 判断一一对应的每个输入是否也满足条件
                    # 没有递归，因为是拓扑序访问，前面的结果已经计算完了
                    if all(
                        match_inp is None or op_inp.name in matching_ops[match_inp]
                        for match_inp, op_inp in zip(
                            match_op.inputs, op.all_input_nodes
                        )
                    ):
                        matching_ops[match_op.name].add(op.name)

                # 输入的匹配方式是子集匹配 match_op.inputs "belongs to" op.all_input_nodes
                elif match_op.input_match_type == InputMatchType.SUBSET:
                    if len(match_op.inputs) > len(op.all_input_nodes):
                        continue
                    # 跑二分图最大匹配，判断是否每个match_op的输入都能找到对应的op输入
                    calc = Hungary(len(match_op.inputs), len(op.all_input_nodes))
                    for id1, match_inp in match_op.inputs:
                        if match_inp is None:
                            for i in range(calc.m):
                                calc.add_edge(id1, i)
                        else:
                            for id2, op_inp in enumerate(op.all_input_nodes):
                                if op_inp.name in matching_ops[match_inp]:
                                    calc.add_edge(id1, id2)
                    if calc.apply() == calc.n:
                        matching_ops[match_op.name].add(op.name)
                else:
                    raise NotImplementedError(
                        "unknown input_match_type {}".format(match_op.input_match_type)
                    )
        """
        精细检查,基于上面的剪枝从下往上dfs找到合法解,再增加其他剪枝
        - 多op联合checker
        - 输入有多分支时共享祖先节点排查
        """
        matches = []
        op_used = {op: False for op in operators}
        matches_dict = {}
        for anchor_op_name in matching_ops[self.anchor_name]:

            def dfs(matcher_name: str, target_name: str) -> bool:
                """
                从下往上搜索,把op对应匹配
                对于多输入op的匹配,可能存在前一个input搜出的结果影响后一个input的可行解,所以需要用yield机制还原现场和产生下一个可行解
                """
                if op_used[target_name]:
                    yield False
                else:
                    matches_dict[matcher_name] = target_name
                    op_used[target_name] = True
                    matcher_node = self.names_dict[matcher_name]
                    op = operators[target_name]
                    matcher_inps = []
                    op_inps = []

                    for matcher_inp, op_inp in zip(
                        matcher_node.inputs, op.all_input_nodes
                    ):
                        if matcher_inp is not None:
                            matcher_inps.append(matcher_inp)
                            op_inps.append(op_inp.name)

                    tot_len = len(matcher_inps)
                    out = [None] * tot_len
                    if matcher_node.input_match_type == InputMatchType.ALL:
                        if tot_len == 0:
                            yield True
                        else:
                            pos = 0
                            while pos < tot_len:
                                if matcher_inps[pos] is None:
                                    ret = True
                                else:
                                    if out[pos] is None:
                                        out[pos] = dfs(matcher_inps[pos], op_inps[pos])
                                    ret = next(out[pos])

                                if not ret:
                                    out[pos] = None
                                    pos -= 1
                                    if pos < 0:
                                        break
                                else:
                                    if pos == tot_len - 1:
                                        checkers = self.joint_checkers[matcher_name]
                                        checker_ok = True
                                        for node_names, checker_func in checkers:
                                            checker_ops = get_op_modules(
                                                [
                                                    self.names_dict[i]
                                                    for i in node_names
                                                ],
                                                m,
                                                named_modules,
                                                type="object",
                                            )
                                            args = {
                                                i: matches_dict[i] for i in node_names
                                            }
                                            args["modules"] = checker_ops
                                            checker_ok |= checker_func(**args)
                                        if checker_ok:
                                            yield True
                                    else:
                                        pos += 1

                        op_used[target_name] = False
                        del matches_dict[matcher_name]
                        yield False

                    elif matcher_node.input_match_type == InputMatchType.SUBSET:
                        # FIXME: 预处理一个搜索input顺序
                        raise NotImplementedError("SUBSET not implemented")
                    else:
                        raise NotImplementedError(
                            "unknown input_match_type {}".format(
                                match_op.input_match_type
                            )
                        )

            out = next(dfs(self.anchor_name, anchor_op_name))
            if out:
                matches.append(matches_dict.copy())
                matches_dict = {}
            if not self.matching_strategy.multiple_match:
                break

        # 如果需要返回多个匹配，去重
        if self.matching_strategy.multiple_match:
            match_nodes = []
            new_match_idxs = []
            for match in matches:
                match_nodes.append(set(list(match.keys())))
            for idx, match in enumerate(matches):
                no_repeat_nodes = True
                for new_idx in new_match_idxs:
                    if match_nodes[idx] & match_nodes[new_match_idxs[new_idx]]:
                        no_repeat_nodes = False
                if no_repeat_nodes:
                    new_match_idxs.append(idx)
            matches = [matches[i] for i in new_match_idxs]
        op_module_matches = []
        for match in matches:
            ops = [operators[i] for i in match.values()]
            modules = get_op_modules(ops, m, named_modules, type="object")
            ops_dict = dict(zip(list(match.keys()), ops))
            modules_dict = dict(zip(list(match.keys()), modules))
            op_module_matches.append((ops_dict, modules_dict))

        return op_module_matches


class ReplacePatternBase(object):
    """子图替换的基类。应当由各种不同的变换继承。

    Args:
        ops (List[MatcherNode]):
            构建好的子图,需要在初始化时由 ``self.get_new_graph`` 生成,传入SubgraphMatcher中。
        joint_checkers (List[Tuple[Tuple[str], Callable]]):
            多个MatcherNode的联合自定义匹配条件 list。
            在初始化时由 ``self.make_joint_checkers`` 生成,传入SubgraphMatcher中。
        matching_strategy (ReplaceStrategy):
            子图替换使用的策略。在初始化时由 ``self.make_matching_strategy`` 生成。

    .. Note::
        每次替换后都会对网络做剪枝,删除与输出无关的算子。
    """

    def __init__(self):
        self.ops = self.make_ops()
        self.joint_checkers = self.make_joint_checkers()
        self.matching_strategy = self.make_matching_strategy()
        self.timer = ReplacePatternTimer()

    def get_new_graph(self, nodes_dict: dict, modules_dict: dict):
        """需要重载的方法。输入{op_name: torch.fx.Node}和{op_name: object}的映射,构造一个变换后的新子图。

        只需要返回一个node anchor,即matcher_nodes子图唯一的输出点。

        请注意只有在node对应modules是一个引用的时候修改才是有意义的,单个int等object没有修改的意义。
        """
        raise NotImplementedError("graph modification not implemented")

    def make_ops(self) -> List[MatcherNode]:
        """需要重载的方法,指定一个要匹配的子图。"""
        raise NotImplementedError("replace_pattern ops not implemented")

    def make_joint_checkers(self) -> List[Tuple[Tuple[str], Callable]]:
        """需要重载的方法,指定需要多个op和对应modules_list一起判断的条件,
        module和前面op名称一一对应。默认不使用任何联合checker。

        .. Note::
            一个示例如下::

            >>> [
            >>>     (
            >>>         ("cat1", "cat2"),
            >>>         lambda cat1, cat2, modules: modules["cat1"].axis == modules["cat2"].axis
            >>>     )
            >>> ]

        """
        return []

    def make_matching_strategy(self) -> ReplaceStrategyInner:
        """可以重载的方法，指定匹配策略。

        默认使用 ``ReplaceStrategy.APPLY_NOINTERSECT_REPEAT`` 。
        """
        return ReplaceStrategy.APPLY_NOINTERSECT_REPEAT

    def apply(self, m: torch.fx.GraphModule) -> bool:
        """执行子图替换。"""
        flg = True
        is_transformed = False
        while flg:
            result = self.apply_once(m)
            is_transformed |= result
            flg = self.matching_strategy.repeat & result
        return is_transformed

    def apply_once(self, m: torch.fx.GraphModule) -> bool:
        """执行单次子图替换。"""

        def replace_op(
            graph: torch.fx.Graph, origin_op: torch.fx.Node, target_op: torch.fx.Node
        ):
            for node in graph.nodes:
                if node != origin_op:
                    node.replace_input_with(origin_op, target_op)

        matcher = SubgraphMatcher(
            ops=self.ops,
            joint_checkers=self.joint_checkers,
            matching_strategy=self.matching_strategy,
        )
        matches = matcher.apply(m)
        is_transformed = False
        for match in matches:
            out_node = self.get_new_graph(
                *match, model=m, transform_idx=self.timer.get_idx()
            )
            replace_op(m.graph, match[0][matcher.anchor_name], out_node)
            is_transformed = True
        m = PruneGraph().apply(m)
        return is_transformed


class Hungary(object):
    """二分图匹配,用于匹配 ``InputMatchType.SUBSET`` 类型的输入格式。"""

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


def get_op_modules(
    ops: List[torch.fx.Node],
    m: torch.fx.GraphModule,
    named_modules: Dict[str, torch.fx.GraphModule],
    type: str,
):
    """辅助函数,用于得到一个torch.fx.Node对应的module,或者对应MatcherNode的op_type。

    Args:
        ops (List[torch.fx.Node]): 需要处理的算子 list。
        m (torch.fx.GraphModule): 原模型。
        named_modules (Dict[str, torch.fx.GraphModule]): 等于 ``dict(m.named_modules())``
        type (str): 需要获得的返回值类型,支持返回op_type或module。取值可以是["class", "object"]。
    """

    assert type in ["class", "object"]
    real_ops = []
    for op in ops:
        if op.op in ["placeholder", "output"]:  # input / output
            real_op = None
        elif op.op == "get_attr":  # parameter or constants
            if type == "class":
                real_op = getattr(m, op.target).__class__
            else:
                real_op = getattr(m, op.target)
        elif op.op == "call_method":  # torch.xxx / torch.Tensor.xxx
            real_op = getattr(torch.Tensor, op.target, None)
        elif isinstance(op.target, str):  # named modules
            if type == "class":
                real_op = named_modules[op.target].__class__
            else:
                real_op = named_modules[op.target]
        else:  # builtin object
            real_op = op.target
        real_ops.append(real_op)
    return real_ops
