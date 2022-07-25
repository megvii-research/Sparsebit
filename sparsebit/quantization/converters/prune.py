import torch
import torch.fx as fx


class PruneGraph(object):
    """网络剪枝，去掉和输出无关的算子。"""

    def __init__(self):
        pass

    def apply(self, m: torch.fx.GraphModule):
        """运行剪枝。

        Args:
            m (torch.fx.GraphModule): 需要剪枝的模型。

        Returns:
            torch.fx.GraphModule: 剪枝后的新模型。
        """
        node_dict = {i.name: i for i in m.graph.nodes}
        q = []
        q_names = set()
        for node in m.graph.nodes:
            if node.op == "output":
                q_names.add(node.name)
                q.append(node.name)
        pos = 0
        while pos < len(q):
            node_name = q[pos]
            pos += 1
            node = node_dict[node_name]
            for input_node in node.all_input_nodes:
                if isinstance(input_node, torch.fx.Node):
                    if input_node.name not in q_names:
                        q_names.add(input_node.name)
                        q.append(input_node.name)

        delete_nodes = [i for i in m.graph.nodes if i.name not in q_names]
        for delete_node in reversed(delete_nodes):
            m.graph.erase_node(delete_node)

        m.recompile()
        return fx.GraphModule(m, m.graph)
