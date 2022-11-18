from typing import Union, Dict, List, Tuple

import onnx
import onnx.helper
import onnx.numpy_helper
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph


import numpy as np
from onnx_graphsurgeon.ir.tensor import Constant
from onnx_graphsurgeon.logger import G_LOGGER
from onnx_graphsurgeon.util import misc

class Graph_modified(object):
    def __init__(self, graph):
        self.graph = graph

    def fold_constants(
        self, fold_shapes=True, recurse_subgraphs=True, partitioning=None, error_ok=True, flatten_subgraphs=True
    ):
        """
        Folds constants in-place in the graph. The graph must be topologically sorted prior to
        calling this function (see `toposort()`).

        This function will not remove constants after folding them. In order to get rid of
        these hanging nodes, you can run the `cleanup()` function.

        *Note: Due to how this function is implemented, the graph must be exportable to ONNX,
        and evaluable in ONNX-Runtime. Additionally, ONNX-Runtime must be installed.*

        Args:
            fold_shapes (bool):
                    Whether to fold `Shape` nodes in the graph.
                    This requires shapes to be inferred in the graph, and can only fold
                    static shapes.
                    Defaults to True.
            recurse_subgraphs (bool):
                    Whether to recursively fold constants in subgraphs.
                    Defaults to True.
            partitioning (Union[str, None]):
                    Whether/How to partition the graph so that errors in folding one
                    part of a model do not affect other parts. Available modes are:

                    - None: Do not partition the graph. If inference fails, no constants are folded.
                    - "basic": Partition the graph. If inference fails in one partition, other partitions will
                            remain unaffected.
                    - "recursive": Parition the graph recursively. If inference fails in a partition, the partition
                            will be further paritioned.

                    Defaults to None.
            error_ok (bool):
                    Whether inference errors should be suppressed.
                    When this is enabled, any errors encountered during inference will be re-raised.
                    Defaults to True.
            flatten_subgraphs (bool):
                    Whether to flatten subgraphs where possible. For example, `If` nodes with a constant condition
                    can be flattened into the parent graph.

        Returns:
            self
        """
        import onnxruntime as rt
        from onnx_graphsurgeon.exporters.onnx_exporter import export_onnx, dtype_to_onnx

        PARTITIONING_MODES = [None, "basic", "recursive"]
        if partitioning not in PARTITIONING_MODES:
            G_LOGGER.critical("Argument for parameter 'partitioning' must be one of: {:}".format(PARTITIONING_MODES))
        ORT_PROVIDERS = ["CPUExecutionProvider"]

        # First perform shape tensor cast elision on the graph prior to other constant folding
        # Search for Cast(s) (from int -> float) -> intermediate operator (with float constants) -> Cast(s) (back to int)
        # This pattern is problematic for TensorRT since these operations may be performed on Shape Tensors, which
        # are not allowed to be floating point type. Attempt to fold the pattern here
        VALID_CAST_ELISION_OPS = ["Add", "Sub", "Mul", "Div", "Max", "Min", "Equal", "Greater", "Less", "Concat"]

        def run_cast_elision(node):
            import onnx

            if node.op not in VALID_CAST_ELISION_OPS:
                return

            # Get list of input nodes that cast to float32
            inp_casts = [
                inp_node
                for inp_tensor in node.inputs
                for inp_node in inp_tensor.inputs
                if inp_node.op == "Cast" and inp_node.attrs["to"] == 1
            ]

            # No cast nodes found, return early
            if not inp_casts:
                return

            # Ensure that all input cast nodes are casting from the same type
            inp_dtypes = [dtype_to_onnx(inp_cast.inputs[0].dtype) for inp_cast in inp_casts]
            if len(set(inp_dtypes)) != 1:
                return

            final_type = inp_dtypes[0]

            # Get list of output nodes that cast to int32 or int64
            out_casts = [
                out_node
                for out_tensor in node.outputs
                for out_node in out_tensor.outputs
                if out_node.op == "Cast" and out_node.attrs["to"] in [6, 7]
            ]

            # No cast node found on ouptuts, return early
            if not out_casts:
                return

            # Ensure that all output cast nodes are casting to the same type and that this
            # matches the original type before the inputs were casted.
            out_dtypes = [out_cast.attrs["to"] for out_cast in out_casts]
            if len(set(out_dtypes)) != 1 or out_dtypes[0] != final_type:
                return

            # If all checks passed - update constant values.
            for inp in node.inputs:
                if isinstance(inp, Constant):
                    inp.values = inp.values.astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[final_type])

            # "Remove" casts nodes by changing I/O node operators to Identity. Update corresponding tensor dtypes as well
            def replace_with_identity(cast_node, change_dtype):
                cast_node.op = "Identity"
                cast_node.attrs = {}
                getattr(cast_node, change_dtype)[0].dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[final_type]
                G_LOGGER.debug("Cast node {:} elided".format(cast_node.name))

            for inp in inp_casts:
                replace_with_identity(inp, change_dtype="outputs")

            for out in out_casts:
                replace_with_identity(out, change_dtype="inputs")

        # Perform shape tensor cast elision:
        if fold_shapes:
            G_LOGGER.debug("Performing shape tensor cast elision in {:}".format(self.graph.name))
            try:
                for node in self.graph.nodes:
                    run_cast_elision(node)
            except Exception as err:
                if not error_ok:
                    raise err
                G_LOGGER.warning("'{:}' routine failed with: {:}".format("Shape tensor cast elision", err))

        G_LOGGER.debug("Folding constants in {:}".format(self.graph.name))

        graph_clone = self.graph.copy()
        clone_tensors = graph_clone.tensors()

        # We find graph constants in two passes:
        # Pass 1 finds all Constant tensors in the graph, then walks over their outputs.
        # Pass 2 searches for Shape nodes that have variable inputs (i.e. not marked const in pass 1)
        #    and turns them into Constants iff the input has a statically known shape.

        def update_foldable_outputs(graph_constants):
            def is_foldable(node):
                def all_tensors_const(tensors):
                    return all([t.name in graph_constants for t in tensors])

                if not all_tensors_const(node.inputs) or node.op in ["QuantizeLinear", "DequantizeLinear"]:
                    return False

                all_subgraph_foreign_tensors_const = True
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        foreign_tensors = attr._foreign_tensors().values()
                        all_subgraph_foreign_tensors_const &= all_tensors_const(foreign_tensors)
                return all_subgraph_foreign_tensors_const

            # Walks along the outputs of graph_constants to see if they can also be computed statically.
            # Since the graph is topologically sorted, this should find all constant nodes in the graph.
            for node in graph_clone.nodes:
                if is_foldable(node):
                    graph_constants.update({out.name: out for out in node.outputs})
            return graph_constants

        # Pass 1: Non-shape Constant Folding

        graph_constants = {name: tensor for name, tensor in clone_tensors.items() if isinstance(tensor, Constant)}

        # Replaces outputs of Constant nodes with constant tensors
        for tensor in clone_tensors.values():
            if len(tensor.inputs) == 1:
                node = tensor.inputs[0]
                if node.op == "Constant":
                    graph_constants[tensor.name] = tensor.to_constant(
                        node.attrs["value"]._values
                    )  # Using ._values avoids copying
                    graph_constants[tensor.name].inputs.clear()

        graph_constants = update_foldable_outputs(graph_constants)

        # Pass 2: Shape Folding

        def get_producer(tensor, op):
            """
            Get the producer of the specified tensor iff it matches op
            """
            if len(tensor.inputs) != 1:
                return None

            node = tensor.inputs[0]
            if node.op != op:
                return None
            return node

        def get_input(node, index=0):
            """
            Get the input tensor of a node iff the input tensor is not already marked a graph constant.
            """
            if node is None:
                return None

            inp = node.inputs[index]

            # If the input was already found to be a constant, it will be folded anyway.
            if inp.name in graph_constants:
                return None

            return inp

        def get_scalar_value(tensor):
            """
            Gets the scalar value of a constant tensor with a single item
            """
            if not tensor.shape:
                return tensor.values
            else:
                return list(tensor.values)[0]

        def fold_shape(tensor):
            inp = get_input(get_producer(tensor, "Shape"))
            if inp is None:
                return None

            if inp.shape is None or misc.is_dynamic_shape(inp.shape):
                return None
            return np.array(inp.shape, dtype=np.int64)

        def fold_shape_gather(tensor):
            gather = get_producer(tensor, "Gather")
            if gather is None:
                return None

            data = gather.inputs[0]
            indices_tensor = gather.inputs[1]

            inp = get_input(get_producer(data, "Shape"))
            if inp is None or inp.shape is None:
                return None

            if not isinstance(indices_tensor, Constant):
                return None

            indices = indices_tensor.values
            if not indices.shape:  # Scalar-case
                shape = inp.shape[int(indices)]
                if misc.is_dynamic_dimension(shape):
                    return None
            else:
                shape = [inp.shape[index] for index in indices]
                if misc.is_dynamic_shape(shape):
                    return None

            return np.array(shape, dtype=np.int64)

        def fold_shape_slice(tensor):
            slice = get_producer(tensor, "Slice")
            if slice is None:
                return None

            data = slice.inputs[0]

            if len(slice.inputs) >= 3:
                starts, ends = slice.inputs[1:3]
                if any(not isinstance(t, Constant) for t in [starts, ends]):
                    return None
                starts, ends = get_scalar_value(starts), get_scalar_value(ends)
            elif "starts" in slice.attrs and "ends" in slice.attrs:
                starts, ends = slice.attrs["starts"][0], slice.attrs["ends"][0]
            else:
                return None

            inp = get_input(get_producer(data, "Shape"))
            if inp is None or inp.shape is None:
                return None

            # For shape tensors, we can only slice on the 0th dimension.
            if len(slice.inputs) > 3:
                axes = slice.inputs[3]
                if not isinstance(axes, Constant):
                    return None

                if get_scalar_value(axes) != 0:
                    return None
            elif "axes" in slice.attrs:
                if slice.attrs["axes"][0] != 0:
                    return None

            steps = 1
            if len(slice.inputs) > 4:
                steps = slice.inputs[4]
                if not isinstance(steps, Constant):
                    return None
                steps = get_scalar_value(steps)
            elif "steps" in slice.attrs:
                steps = slice.attrs["steps"][0]

            shape = inp.shape[starts:ends:steps]
            if misc.is_dynamic_shape(shape):
                return None

            return np.array(shape, dtype=np.int64)

        if fold_shapes:
            # NOTE: The order of shape folding passes is important to maximize how much we fold (phase-ordering problem).
            SHAPE_FOLD_FUNCS = [fold_shape_gather, fold_shape_slice, fold_shape]
            for shape_fold_func in SHAPE_FOLD_FUNCS:
                try:
                    for tensor in clone_tensors.values():
                        shape_of = shape_fold_func(tensor)

                        if shape_of is not None:
                            G_LOGGER.ultra_verbose("Folding shape tensor: {:} to: {:}".format(tensor.name, shape_of))
                            graph_constants[tensor.name] = tensor.to_constant(shape_of)
                            graph_constants[tensor.name].inputs.clear()
                except Exception as err:
                    if not error_ok:
                        raise err
                    G_LOGGER.warning("'{:}' routine failed with:\n{:}".format(shape_fold_func.__name__, err))
                else:
                    graph_constants = update_foldable_outputs(graph_constants)

        def partition_and_infer(subgraph):
            def get_out_node_ids():
                # Gets the final output nodes - producer nodes of graph output tensors without other outputs.
                with subgraph.node_ids():
                    out_node_ids = set()
                    for out in subgraph.outputs:
                        if not out.outputs and not isinstance(out, Constant):
                            for n_inp in out.inputs:
                                out_node_ids.add(n_inp.id)
                return out_node_ids

            # Compute each output node in a separate subgraph.
            out_node_ids = get_out_node_ids()
            constant_values = {}

            for index in out_node_ids:  # Have to use index since 'node' is not in part
                part = subgraph.copy()
                out_node = part.nodes[index]
                part.outputs = out_node.outputs
                part.name = "Folding: {:}".format([out.name for out in part.outputs])
                part.cleanup(remove_unused_graph_inputs=True)
                names = [out.name for out in part.outputs]

                try:
                    # Determining types is not trivial, and ONNX-RT does its own type inference.
                    sess = rt.InferenceSession(
                        export_onnx(part, do_type_check=False).SerializeToString(), providers=ORT_PROVIDERS
                    )
                    values = sess.run(names, {})
                except Exception as err:
                    G_LOGGER.warning("Inference failed for subgraph: {:}. Note: Error was:\n{:}".format(part.name, err))
                    if partitioning == "recursive":
                        G_LOGGER.verbose("Attempting to recursively partition subgraph")
                        # Partition failed, peel off last node.
                        # We only need to remove one node, so avoid doing an expensive call to cleanup()
                        part.outputs = out_node.inputs
                        del part.nodes[part.nodes.index(out_node)]
                        out_node.outputs.clear()
                        out_node.inputs.clear()
                    else:
                        G_LOGGER.info("You may see better results if you set partitioning='recursive'")
                        if not error_ok:
                            raise err

                    constant_values.update(partition_and_infer(part))
                else:
                    constant_values.update({name: val for name, val in zip(names, values)})

            return constant_values

        # Next, evaluate the foldable variables with ONNX-Runtime

        # Only evaluate foldable values that have non-foldable outputs or are graph outputs.
        # Otherwise, if all the outputs are foldable, then we can just evaluate the outputs directly.
        def should_eval_foldable(tensor):
            non_const = not isinstance(tensor, Constant)
            is_graph_output = not tensor.outputs
            has_non_foldable_outputs = any(out.name not in graph_constants for out in tensor.outputs)
            return non_const and (is_graph_output or has_non_foldable_outputs)

        graph_clone.outputs = [t for t in graph_constants.values() if should_eval_foldable(t)]
        G_LOGGER.debug("Folding tensors: {:}".format(graph_clone.outputs))
        graph_clone.cleanup(remove_unused_graph_inputs=True)

        # Using ._values avoids a deep copy of the values.
        constant_values = {
            name: tensor._values for name, tensor in graph_constants.items() if isinstance(tensor, Constant)
        }
        if graph_clone.outputs:
            if partitioning:
                constant_values.update(partition_and_infer(graph_clone))
            else:
                names = [t.name for t in graph_clone.outputs]
                try:
                    sess = rt.InferenceSession(
                        export_onnx(graph_clone, do_type_check=False).SerializeToString(), providers=ORT_PROVIDERS
                    )
                    values = sess.run(names, {})
                    constant_values.update({name: val for name, val in zip(names, values)})
                except Exception as err:
                    G_LOGGER.warning(
                        "Inference failed. You may want to try enabling partitioning to see better results. "
                        "Note: Error was:\n{:}".format(err)
                    )
                    G_LOGGER.verbose("Note: Graph was:\n{:}".format(graph_clone))
                    if not error_ok:
                        raise
        elif not constant_values:
            G_LOGGER.debug(
                "Could not find any nodes in this graph ({:}) that can be folded. "
                "This could mean that constant folding has already been run on this graph. "
                "Skipping.".format(self.graph.name)
            )

        # Finally, replace the Variables in the original graph with constants.
        if constant_values:
            graph_tensors = self.graph.tensors()
            for name, values in constant_values.items():
                tensor = graph_tensors[name]
                if not isinstance(tensor, Constant):
                    tensor.to_constant(values)
                    tensor.inputs.clear()  # Constants do not need inputs

        # Folding subgraphs after the outer graph can lead to better folding.
        def fold_subgraphs():
            for node in self.graph.nodes:
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        attr.fold_constants(fold_shapes=fold_shapes, partitioning=partitioning)

        if recurse_subgraphs:
            fold_subgraphs()

        if flatten_subgraphs:
            # Flatten conditional subgraphs
            index = 0
            while index < len(self.graph.nodes):
                node = self.graph.nodes[index]
                if node.op == "If" and isinstance(node.inputs[0], Constant):
                    G_LOGGER.debug("Flattening conditional: {:}".format(node))
                    cond = get_scalar_value(node.inputs[0])
                    subgraph = node.attrs["then_branch"] if cond else node.attrs["else_branch"]
                    # Need to add a suffix to subgraph tensors so they don't collide with outer graph tensors
                    for tensor in subgraph._local_tensors().values():
                        tensor.name += "_subg_{:}_{:}".format(index, subgraph.name)

                    # The subgraph outputs correspond to the If node outputs. Only the latter are visible
                    # in the parent graph, so we rebind the producer nodes of the subgraph outputs to point
                    # to the output tensors of the If instead.
                    for node_out, subgraph_out in zip(node.outputs, subgraph.outputs):
                        node_out.inputs.clear()
                        for producer in subgraph_out.inputs:
                            for tensor_idx, out_tensor in enumerate(producer.outputs):
                                if out_tensor == subgraph_out:
                                    producer.outputs[tensor_idx] = node_out

                    # Copy subgraph nodes into parent graph at the index of the If.
                    del self.graph.nodes[index]
                    self.graph.nodes[index:index] = subgraph.nodes
                    index += len(subgraph.nodes) - 1

                index += 1

        return self.graph

class ReplacePattern_FoldConstant:
    def __init__(self):
        pass

    def apply(self, model: onnx.GraphProto):
        graph = Graph_modified(gs.import_onnx(model))
        graph = graph.fold_constants().cleanup()
        model = gs.export_onnx(graph)
        return model


ReplacePatterns = [
    ReplacePattern_FoldConstant()
]
