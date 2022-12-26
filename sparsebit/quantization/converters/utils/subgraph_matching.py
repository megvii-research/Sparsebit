from typing import List, Set, Tuple, Dict, Callable, Union
import torch

from .subgraph_matching_node import MatchingNode, InputMatchingType
from .dominator_tree import DominatorTree
from .subgraph_matching_utils import ReplaceStrategy, get_operators, get_operators_type
from .bitpartite_graph_matching import Hungary


class SubgraphMatcher(object):
    """网络中子图匹配类,执行匹配子图相关部分的工作。

    Args:
        matching_nodes (List[MatchingNode]):
            一个构建好连接关系的MatchingNode list。
            需要满足拓扑序,即一个MatchingNode引用到的其他输入点都在它前面。
        joint_checkers (List[Tuple[Tuple[str], Callable]]):
            多个MatchingNode的联合自定义匹配条件 list。
            每个联合匹配条件是一个形如 `` ((names), joint_checker) `` 的形式。
            ``joint_checker`` 输入(name1, name2, ..., modules:dict),输出bool。
        matching_strategy (ReplaceStrategy):
            调用Matcher的子图替换使用的策略,根据这个策略决定是否需要同时返回多个匹配。
    """

    def __init__(
        self,
        matching_nodes: List[MatchingNode],
        joint_checkers: List[Tuple[Tuple[str], Callable]],
        matching_strategy=ReplaceStrategy.APPLY_REPEAT,
    ):
        self.matching_nodes = matching_nodes.copy()
        self.matching_strategy = matching_strategy
        self.matching_node_names_dict = {
            matching_node.name: matching_node for matching_node in matching_nodes
        }
        self.topological_check()
        self.pad_supported_node()
        self.len = len(self.matching_nodes)
        dominator_tree = self.build_reversed_dominator_tree()
        self.get_dfn(dominator_tree)
        self.make_joint_checkers(joint_checkers, dominator_tree)

    def apply(
        self, m: torch.fx.GraphModule
    ) -> List[Tuple[Dict[str, torch.fx.Node], Dict[str, torch.nn.Module]]]:
        """执行子图匹配。

        Args:
            m (torch.fx.GraphModule): 需要匹配的网络。

        Returns:
            Tuple[Dict[str, torch.fx.Node], Dict[str, torch.nn.Module]]:
                返回一个匹配list,每个匹配是 ``(nodes_dict, modules_dict)`` 的形式。

                其中:

                - nodes_dict为一个MatchingNode名称到网络中node的映射dict。
                - modules_dict为一个MatchingNode名称到网络中module的映射dict。
        """
        named_modules = dict(m.named_modules())
        graph = m.graph
        operators = [i for i in graph.nodes]
        matching_ops = self.coarse_filtering(m, named_modules, operators)
        operators, matching_ops = self.pad_supported_operator(operators, matching_ops)

        match_dict = self.match(m, named_modules, matching_ops, operators)
        if not match_dict:
            return None, None
        # build return info
        del match_dict[self.len - 1]  # remove supported node __root__ matching
        ops_dict: Dict[str, torch.fx.Node] = {}
        modules_dict: Dict[str, Union[torch.nn.Module, None]] = {}
        for node_idx, op_idx in match_dict.items():
            node_name = self.matching_nodes[node_idx].name
            op = operators[op_idx]
            module = get_operators([op], m, named_modules)[0]
            ops_dict[node_name] = op
            modules_dict[node_name] = module
        return ops_dict, modules_dict

    def topological_check(self):
        visited_node_names = []
        connected_nodes = set()
        for matching_node in self.matching_nodes:
            for input_name in matching_node.inputs:
                if input_name is None:
                    continue
                assert (
                    input_name in visited_node_names
                ), "Matching nodes not in topological order!"
                connected_nodes.add(input_name)
                connected_nodes.add(matching_node.name)
            visited_node_names.append(matching_node.name)

        assert len(self.matching_node_names_dict) == len(
            self.matching_nodes
        ), "Duplicated names found in matching nodes!"
        assert len(self.matching_nodes) == 1 or len(connected_nodes) == len(
            self.matching_nodes
        ), "Matching nodes are not one connected block!"

    def pad_supported_node(self):
        out_degrees = {k: 0 for k in self.matching_node_names_dict}
        output_nodes = []
        for matching_node in self.matching_nodes:
            assert (
                matching_node.name != "__root__"
            ), "MatcherNode with name __root__ found, which conflicts to a reserved node name."
            for input_node in matching_node.inputs:
                if input_node is None:
                    continue
                out_degrees[input_node] += 1
        for matching_node in self.matching_nodes:
            if out_degrees[matching_node.name] == 0:
                output_nodes.append(matching_node)

        self.matching_nodes.append(
            MatchingNode(
                name="__root__",
                inputs=[i.name for i in output_nodes],
                op_type="__root__",
                input_match_type=InputMatchingType.ALL
                if len(output_nodes) == 1
                else InputMatchingType.SUBSET,
            )
        )
        self.matching_node_names_dict["__root__"] = self.matching_nodes[-1]

    def build_reversed_dominator_tree(self) -> DominatorTree:
        T = DominatorTree(len(self.matching_nodes))
        self.reversed_edges = [[] for i in range(self.len)]
        idxes = {i.name: idx for idx, i in enumerate(self.matching_nodes)}
        for idx, matching_node in enumerate(self.matching_nodes):
            for input_node in matching_node.inputs:
                if input_node is None:
                    continue
                input_idx = idxes[input_node]
                T.add_edge(idx, input_idx)
                self.reversed_edges[input_idx].append(idx)
        T.solve()
        return T

    def get_dfn(self, T: DominatorTree):
        self.dfn = [0] * len(self.matching_nodes)
        self.rnk = [0] * len(self.matching_nodes)

        def dfs(x: int, cnt: int) -> int:
            self.dfn[x] = cnt
            self.rnk[cnt] = x
            cnt += 1
            for nex in T.targets[x]:
                cnt = dfs(nex, cnt)
            return cnt

        dfs(len(self.matching_nodes) - 1, 0)

    def make_joint_checkers(
        self, checkers: List[Tuple[Tuple[str], Callable]], T: DominatorTree
    ):
        def get_idx(name, names):
            for idx, _name in enumerate(names):
                if name == _name:
                    return idx
            return None

        self.checkers = {}
        for checker in checkers:
            required_node_names, checker_func = checker
            max_dfn = 0
            # determine the fastest time to do the checking
            idxs = []
            for required_node_name in required_node_names:
                assert (
                    required_node_name in self.matching_node_names_dict
                ), "input must be in subgraph"
                idx = get_idx(required_node_name, self.matching_node_names_dict)
                idxs.append(idx)
                max_dfn = max(max_dfn, self.dfn[idx])
            final_idx = get_idx(max_dfn, self.dfn)
            if final_idx not in self.checkers:
                self.checkers[final_idx] = []
            self.checkers[final_idx].append((idxs, checker_func))

    def coarse_filtering(
        self,
        m: torch.fx.GraphModule,
        named_modules: Dict[str, torch.fx.GraphModule],
        operators: List[torch.fx.Node],
    ) -> Dict[int, Set[int]]:
        # filter the input ops in different types
        optional_ops: Dict[set] = {}
        for idx, op in enumerate(operators):
            setattr(op, "index", idx)  # add index in torch.fx.Node
            op_type = get_operators_type([op], m, named_modules)[0]
            if op_type not in optional_ops:
                optional_ops[op_type] = set()
            optional_ops[op_type].add(idx)

        # a coarse check for the ops, in type, input relation and checker
        name_to_idxes = {i.name: idx for idx, i in enumerate(self.matching_nodes)}
        matching_ops: Dict[str, set] = {}
        for idx, match_node in enumerate(self.matching_nodes):
            matching_ops[idx] = set()
            op_with_correct_func = set(
                [j for i in match_node.op_type for j in optional_ops.get(i, set())]
            )  # op_type check
            for op_idx in op_with_correct_func:
                op = operators[op_idx]
                op_module = get_operators([op], m, named_modules)[0]

                # checker check
                if not match_node.checker(op, module=op_module):
                    continue

                # input check

                # for InputMatchingType.ALL
                # match_op.inputs == op.all_input_nodes
                if match_node.input_match_type == InputMatchingType.ALL:
                    if len(match_node.inputs) != len(op.all_input_nodes):
                        continue

                    # check recursively
                    # but no need for recurse, since input nodes are calculated before
                    if all(
                        match_inp is None
                        or getattr(op_inp, "index", None)
                        in matching_ops[name_to_idxes[match_inp]]
                        for match_inp, op_inp in zip(
                            match_node.inputs, op.all_input_nodes
                        )
                    ):
                        matching_ops[idx].add(op_idx)
                # for InputMatchingType.SUBSET
                # match_op.inputs "belongs to" op.all_input_nodes
                elif match_node.input_match_type == InputMatchingType.SUBSET:
                    if len(match_node.inputs) > len(op.all_input_nodes):
                        continue
                    # Bipartite graph maximum matching
                    calc = Hungary(len(match_node.inputs), len(op.all_input_nodes))
                    for id1, match_inp in enumerate(match_node.inputs):
                        if match_inp is None:
                            for i in range(calc.m):
                                calc.add_edge(id1, i)
                        else:
                            for id2, op_inp in enumerate(op.all_input_nodes):
                                if (
                                    op_inp.index
                                    in matching_ops[name_to_idxes[match_inp]]
                                ):
                                    calc.add_edge(id1, id2)
                    if calc.apply() == calc.n:
                        matching_ops[idx].add(op_idx)
                else:
                    raise NotImplementedError(
                        "unknown input_match_type {}".format(
                            match_node.input_match_type
                        )
                    )

        for op in operators:
            delattr(op, "index")

        return matching_ops

    def pad_supported_operator(
        self, operators: List[torch.fx.Node], matching_ops: Dict[int, Set[int]]
    ) -> Tuple[List[torch.fx.Node], Dict[int, Set[int]]]:

        supported_node = self.matching_node_names_dict["__root__"]
        input_idxs = set()
        for idx, node in enumerate(self.matching_nodes):
            if node.name in supported_node.inputs:
                input_idxs.update(matching_ops[idx])
        _input_nodes = {operators[i]: None for i in input_idxs}
        supported_op = torch.fx.Node(
            name="__root__",
            graph=None,
            op="call_function",
            target=lambda x: x,
            args=(),
            kwargs={},
        )
        supported_op._input_nodes = _input_nodes
        operators.append(supported_op)
        matching_ops[self.len - 1] = set([len(operators) - 1])

        return operators, matching_ops

    def match(
        self,
        m: torch.fx.GraphModule,
        named_modules: Dict[str, torch.fx.GraphModule],
        matching_ops: Dict,
        operators: List[torch.fx.Node],
    ) -> Dict[int, int]:
        match_dict: Dict[int, int] = {}
        used_input_pos: Dict[int, List[bool]] = {}

        def get_pos(
            cur_op: torch.fx.Node,
            pred_op: torch.fx.Node,
            cur_node: MatchingNode,
            pred_node: MatchingNode,
            used_input: List[bool],
            mode: InputMatchingType,
        ) -> List[int]:
            all_input_nodes = pred_op.all_input_nodes
            op_positions = [idx for idx, i in enumerate(all_input_nodes) if i == cur_op]
            used_mask = any(used_input[i] for i in op_positions)
            if used_mask:
                return []
            if mode == InputMatchingType.ALL:
                node_positions = [
                    idx for idx, i in enumerate(pred_node.inputs) if i == cur_node.name
                ]
                if op_positions == node_positions:
                    return op_positions
                else:
                    return []
            elif mode == InputMatchingType.SUBSET:
                return op_positions

        def dfs_per_layer(idx: int) -> bool:
            cur_idx = self.rnk[idx]
            cur_node = self.matching_nodes[cur_idx]

            # prepare checker in this node
            checker_funcs: List[Callable] = []  # functions (for each checker)
            checker_dicts: List[Dict[int, Callable]] = []  # dicts for modules in funcs
            checker_nodes: List[List[int]] = []  # list of index of used nodes
            checker_poss: List[int] = []  # position of cur_idx to fill in checker_nodes
            for checker_idxs, checker_func in self.checkers.get(cur_idx, []):
                checker_nodes.append([])
                checker_dicts.append({})
                checker_funcs.append(checker_func)
                for i, checker_idx in enumerate(checker_idxs):
                    if checker_idx != cur_idx:
                        op = operators[match_dict[checker_idx]]
                        module = get_operators([op], m, named_modules)[0]
                        checker_nodes[-1].append(op)
                        checker_dicts[-1][
                            self.matching_nodes[checker_idx].name
                        ] = module
                    else:
                        checker_nodes[-1].append(None)
                        checker_pos = i
                checker_poss.append(checker_pos)

            # search in candidates, and goto next one recursively
            # only check the joint_checkers, and check backward connect relations
            for op_idx in matching_ops[cur_idx]:
                op = operators[op_idx]
                # check backward connect relations
                rev_mask = False
                input_pos = []
                for pred_idx in self.reversed_edges[cur_idx]:
                    pred_op_idx = match_dict[pred_idx]
                    pred_op = operators[pred_op_idx]
                    pred_node = self.matching_nodes[pred_idx]
                    pos = get_pos(
                        op,
                        pred_op,
                        cur_node,
                        pred_node,
                        used_input=used_input_pos[pred_idx],
                        mode=pred_node.input_match_type,
                    )
                    if not pos:
                        # revert
                        for ipred, ipos in input_pos:
                            used_input_pos[ipred][ipos] = False
                        rev_mask = True
                        break
                    else:
                        for ipos in pos:
                            used_input_pos[pred_idx][ipos] = True
                            input_pos.append((pred_idx, ipos))
                if rev_mask:
                    continue

                # fill the final node operators[cur_idx] and check
                checker_mask = False
                for checker_func, checker_dict, checker_node, checker_pos in zip(
                    checker_funcs, checker_dicts, checker_nodes, checker_poss
                ):
                    checker_node[checker_pos] = op
                    module = get_operators([op], m, named_modules)[0]
                    checker_dict[self.matching_nodes[checker_pos].name] = module
                    if not checker_func(*checker_node, modules=checker_dict):
                        checker_mask = True
                        break
                if checker_mask:
                    continue

                match_dict[cur_idx] = op_idx
                used_input_pos[cur_idx] = [False] * len(op.all_input_nodes)

                if idx == self.len - 1:
                    return True
                elif dfs_per_layer(idx + 1):
                    return True

                del match_dict[cur_idx]
                del used_input_pos[cur_idx]
                for pred_idx, pos in input_pos:
                    used_input_pos[pred_idx][pos] = False

            return False

        out_mask = dfs_per_layer(0)
        return match_dict if out_mask else {}
