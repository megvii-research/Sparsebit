from typing import List, Optional
import re
import torch
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module
from sparsebit.quantization.modules import QuantOpr, QConv2d, QLinear
from .reconstruction import save_inp_oup_data, subgraph_reconstruction
from .utils import (
    deepcopy_graphmodule,
    node2modules,
    qnode2fpnode,
    topology_order,
    flattend_args,
    get_input_node_nums,
)


ADAROUND_SUPPORT_OPR = (QConv2d, QLinear)


def extract_block(node, node_modules, block_mark):
    layer_node_list = []
    input_nodes = node.all_input_nodes
    cnt = dict()
    q, p = [], []
    while len(input_nodes) != 0:
        for inp in input_nodes:
            for user in inp.users:
                if user.op == "call_module" and isinstance(
                    node_modules[user], ADAROUND_SUPPORT_OPR
                ):
                    if user.name == node.name or (
                        block_mark is not None and block_mark in user.name
                    ):
                        if user not in cnt:
                            cnt[user] = get_input_node_nums(user.args)
                            p.append(user)
                    else:
                        continue
                elif user.op in ["placeholder", "output"]:
                    continue
                else:
                    if user not in cnt:
                        cnt[user] = get_input_node_nums(user.args)
                        p.append(user)
                cnt[user] -= 1
                if cnt[user] == 0:
                    q.append(user)
                    p.remove(user)
        layer_node_list.extend(q)
        input_nodes = q
        q = []
    return layer_node_list


def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], output: fx.Node):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    inp_lst = []
    for node in nodes:
        for arg in flattend_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in nodes and arg not in inp_lst:
                    inp_lst.append(node)
                    arg_name = node.name
                    new_node = new_graph.placeholder(arg_name)
                    env[node] = new_node
                    break
    for node in nodes:
        if node in inp_lst:
            continue
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    # create this or there will not be return value
    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


def ptq_reconstruction(
    model: GraphModule,
    cali_data: list,
    config: dict,
    block_mark_pattern: Optional[str] = None,
):
    """
    Reconsturction for AdaRound, BRECQ.
    Args:
        model: GraphModule to do PTQ Finetune
        cali_data (List): a list of calibration tensor
        config (dict): a config for PTQ reconstruction
    """
    float_model = model
    float_model.eval()
    quant_model = deepcopy_graphmodule(model)
    nodes = list(quant_model.graph.nodes)  # node list [GRANULARITY: torch.fx.node.Node]
    float_modules = node2modules(
        dict(float_model.named_modules()), float_model.graph.nodes
    )  # node -> node module
    quant_modules = node2modules(
        dict(quant_model.named_modules()), quant_model.graph.nodes
    )  # node -> node module
    node2idx = topology_order(quant_model)  # node -> node topo sorted idx

    qnode2fpnode_dict = qnode2fpnode(quant_modules, float_modules)  # qnode -> fnode
    quant_model.eval()

    # 打开量化
    for n, m in quant_model.named_modules():
        if isinstance(m, QuantOpr):
            m.set_quant(True, True)

    torch.cuda.empty_cache()
    checked_nodes = dict()
    for node in nodes:
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(
            quant_modules[node], ADAROUND_SUPPORT_OPR
        ):
            print(
                "prepare {} reconstruction for {}".format(
                    config.W.QUANTIZER.ADAROUND.GRANULARITY, node
                )
            )
            if config.W.QUANTIZER.ADAROUND.GRANULARITY == "layerwise":
                layer_node_list = [node]
            elif config.W.QUANTIZER.ADAROUND.GRANULARITY == "blockwise":
                assert (
                    block_mark_pattern is not None
                ), "you must provide block_mark_pattern when use brecq"
                match_results = re.findall(block_mark_pattern, node.name)
                if len(match_results) == 0:
                    block_mark = None
                else:
                    block_mark = match_results[0]
                    print(
                        "the block mark for node {} is {}".format(node.name, block_mark)
                    )
                layer_node_list = extract_block(node, quant_modules, block_mark)
            else:
                raise NotImplementedError

            missing_inputs = []
            for _node in layer_node_list:
                for arg in flattend_args(_node.args):
                    if isinstance(arg, torch.fx.Node):
                        if arg not in layer_node_list and arg not in missing_inputs:
                            missing_inputs.append(arg)
            layer_node_list.extend(missing_inputs)
            layer_node_list = sorted(layer_node_list, key=lambda x: node2idx[x])

            print("the node list is below!")
            print(layer_node_list)
            fp32_module = float_modules[qnode2fpnode_dict[layer_node_list[-1]]]
            quant_all_inps = []
            fp32_final_oups = None
            out_is_cached = False
            for _node in layer_node_list:
                if _node.op != "placeholder" and all(
                    [
                        arg in layer_node_list
                        for arg in flattend_args(_node.args)
                        if isinstance(arg, torch.fx.Node)
                    ]
                ):
                    continue
                else:
                    if _node.op == "placeholder":
                        if isinstance(cali_data[0], torch.Tensor):
                            quant_inps = cali_data
                        elif isinstance(cali_data[0], (list, tuple)):
                            quant_inps = [data[node2idx[_node]] for data in cali_data]
                        else:
                            raise "unsupport cali_data GRANULARITY {}".format(
                                GRANULARITY(cali_data)
                            )
                        if config.W.QUANTIZER.ADAROUND.KEEP_GPU:
                            quant_inps = [
                                data.to(next(quant_model.parameters()).device)
                                for data in quant_inps
                            ]
                    else:
                        quant_module = quant_modules[_node]
                        _, quant_inps = save_inp_oup_data(
                            quant_model,
                            None,
                            quant_module,
                            cali_data,
                            store_inp=False,
                            store_oup=True,
                            keep_gpu=config.W.QUANTIZER.ADAROUND.KEEP_GPU,
                        )
                    quant_all_inps.append(quant_inps)
                    if not out_is_cached:
                        # fp32 oups: [out_b1, out_b2, ...]
                        _, fp32_oups = save_inp_oup_data(
                            float_model,
                            None,
                            fp32_module,
                            cali_data,
                            store_inp=False,
                            store_oup=(not out_is_cached),
                            keep_gpu=config.W.QUANTIZER.ADAROUND.KEEP_GPU,
                        )
                        fp32_final_oups = fp32_oups
                        out_is_cached = True
            cached_inps = quant_all_inps
            cached_oups = fp32_final_oups
            quant_modules_by_name = dict()
            for node in layer_node_list:
                if node.op == "call_module":
                    quant_modules_by_name[node.target] = quant_modules[node]
            subgraph = extract_subgraph(
                quant_modules_by_name, layer_node_list, layer_node_list[-1]
            )
            subgraph_reconstruction(subgraph, cached_inps, cached_oups, config)
            for x in layer_node_list:
                checked_nodes[x] = True

    return quant_model
