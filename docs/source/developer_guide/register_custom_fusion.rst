How to register a custom fusion
==============================================================


A fusion schedule will match and replace some parts of the network.

There are only a few fusions now.
See the ``fuse_list`` in ``sparsebit.quantization.converters.*.lists`` for the lists.

.. Note::
    The custom fusions is **experimental** function.
    It's recommended to reuse some of the existing schedules, if the schedule meets the demand already.

Selecting the stage
--------------------------------------------------------------

There are two stages to do the fusion now:

- :doc:`sparsebit/quantization/converters/simplifiers <../user_guide/apis/quantization/converters/simplifiers/index>`

- :doc:`sparsebit/quantization/converters/fuse_operations <../user_guide/apis/quantization/converters/fuse_operations/index>`

The former is in the stage before replacing original modules with :doc:`QuantOpr <../user_guide/apis/quantization/modules/base>`.
And the latter is the opposite.

The schedules in ``simplifiers`` usually find and change some native operator of *torch* , and no parameter calculation is involved.

The schedules in ``fuse_operations`` usually have something to do with *Quantopr* , and may uses parameter calulation.

.. Note::

    All schedules in fuse_operations will be configured in `.yaml` config file.
    Adding a setting of this schedule in :doc:`quant_config <../user_guide/apis/quantization/quant_config>` is required.
    An example is:

    .. code-block:: shell
        :linenos:

        _C.SCHEDULE.NAME_OF_YOUR_TRANSFORMATION_FILE = True / False

    Set default value in *quant_config* to True, or enable the schedule in yaml config, if it's required.

    An example of the config is:

    .. code-block:: yaml
        :linenos:

        SCHEDULE:
            NAME_OF_YOUR_TRANSFORMATION_FILE: True


Writing custom schedule
--------------------------------------------------------------

A schedule file consists of a class called ``ReplacePattern``.
It is a subclass of :doc:`ReplacePatternBase <../user_guide/apis/quantization/converters/base>` .
That is the core fusion class.

It provide 4 methods for the fusion.

**get_new_graph**

    Override this method and give a return value of MatcherNode list.
    The MatcherNode list is a list of MatcherNode with attributes of each node correctly set, including:

    - *name* : String name given to each node. Names of different nodes should **NOT** be the same.
    - *inputs* : A list of string names, corresponding to each input of the node respectively. The names in inputs should be the names in previous MatcherNodes.

        .. Note::
            Using ``None`` in the list means matching a wildcard input.

    - *op_type* : a list of functions or classes. This is a list of all the valid operator types for this node.
    - *checker* : a custom function to add limit to the node. This helps to find specific node and reduce search capacity.

        .. Note::
            the checker accepts two args: the target *torch.fx.Node* and its corresponding instance.
            An example is ``lambda concat_node, module: module.axis == 1`` .
    - *input_match_type* : An enum value in :doc:`InputMatchType <../user_guide/apis/quantization/converters/base>`.

        .. Warning::
            *InputMatchType.SUBSET* is not implemented yet. Only *InputMatchType.ALL* is supported.

    The MatcherNode list represents a subgraph structure. The schedule will match and replace the structure in actual graph.

    .. Note::

        Confirm the MatcherNode list to keep the subgraph acyclic.

        In most cases the subgraph should have only one output node (anchor).
        That is to say, only one node in subgraph has zero out degree.
        The replace function will only replace the anchor node with a generated new node.

**make_matching_strategy**

    Optionally override this method and give a return value in :doc:`ReplaceStrategy <../user_guide/apis/quantization/converters/base>` .

**make_joint_checkers**

    Optionally override this method and give a return value of list of joint-checkers.

    A joint checker is ``tuple(names, joint_checker)`` .
    Here ``names`` include all names of used MatcherNodes in ``joint_checker``.
    ``joint_checker`` is a function that accepts all nodes mentioned in names in corresponding order, and a dict of modules.

    A simple example is ``[ ( ("cat1", "cat2"), lambda cat1, cat2, modules: modules["cat1"].axis == modules["cat2"].axis ) ]`` .

**get_new_graph**

    This is the core function to replace old subgraph with new subgraph.
    The function always accepts 4 args.

    nodes_dict
        A dict and the corresponding actual nodes can be accessed via nodes_dict[name].

    modules_dict
        A dict and the corresponding actual instances can be accessed via modules_dict[name].

    model
        If new node with new module instance is generated, just call model.add_module to register instance.

    transform_idx
        An *int* number, and will change each time the get_new_graph is called.
        This is provided for the schedule running multiple times.
        Adding transform_idx in new operator names avoid conflicting names.

    The function returns one *torch.fx.Node*: the new anchor to replace original anchor.

    Different schedules will have different replacing rules.
    Construct a new subgraph carefully with the help of actual nodes and instances, and only return the new anchor node.

    If it's required to add a new node, try

    .. code-block:: python
        :linenos:

        with model.graph.inserting_after(node_to_replace)\:
            new_node = model.graph.create_node(
                op=...,
                target=...,
                args=...,
                name=...,
                ...,
            )

Finallly switch to the folder of coresponding stage.
There are many schedules and a *lists.py* .
The execution order of schedules is determined by ``fuse_list`` in *list.py* .
To register a schedule, add schedule file name in *lists.py* in any order.
