import copy
import torch
from torch.fx import GraphModule, Node


def deepcopy_graphmodule(gm: GraphModule):
    """Rewrite the deepcopy of GraphModule. (Copy its 'graph'.)
    Args:
        gm (GraphModule):
    Returns:
        GraphModule: A deepcopied gm.
    """
    copied_gm = copy.deepcopy(gm)
    copied_gm.graph = copy.deepcopy(gm.graph)
    return copied_gm


def node2modules(name2modules, nodes):
    modules = dict()
    for node in nodes:
        if node.target in name2modules:
            modules[node] = name2modules[node.target]
    return modules


def qnode2fpnode(quant_modules, fp32_modules):
    quant_named_nodes = {node.target: node for node in quant_modules}
    fp32_named_nodes = {node.target: node for node in fp32_modules}
    qnode2fpnode_dict = {
        quant_named_nodes[key]: fp32_named_nodes[key] for key in quant_named_nodes
    }
    return qnode2fpnode_dict


def topology_order(model):
    node2idx = {}
    for idx, node in enumerate(model.graph.nodes):
        node2idx[node] = idx
    return node2idx


def flattend_args(node):
    _flattend_args = []
    if isinstance(node, dict):
        for v in node.values():
            _flattend_args.extend(flattend_args(v))
    elif isinstance(node, tuple) or isinstance(node, list):
        for n in node:
            _flattend_args.extend(flattend_args(n))
    else:
        _flattend_args.extend([node])
    return _flattend_args


def get_input_node_nums(nodes):
    num = 0
    for node in nodes:
        if isinstance(node, Node):
            num += 1
    return num