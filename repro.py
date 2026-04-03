import onnx
from onnx import helper, TensorProto

def create_model_with_subgraph():
    # Create a subgraph with a Cast node
    # Input -> Cast(to=FLOAT) -> Output
    cast_node_inner = helper.make_node(
        "Cast",
        inputs=["sub_in"],
        outputs=["sub_out"],
        to=TensorProto.FLOAT,
        name="inner_cast"
    )

    subgraph = helper.make_graph(
        [cast_node_inner],
        "subgraph",
        [helper.make_tensor_value_info("sub_in", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("sub_out", TensorProto.FLOAT, [1])]
    )

    # Main graph with If node containing the subgraph
    # If(cond) -> subgraph
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        then_branch=subgraph,
        else_branch=subgraph # reuse for simplicity
    )

    # Another Cast in main graph
    main_cast = helper.make_node(
        "Cast",
        inputs=["in"],
        outputs=["out_cast"],
        to=TensorProto.FLOAT,
        name="main_cast"
    )

    graph = helper.make_graph(
        [if_node, main_cast],
        "test_model",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("in", TensorProto.FLOAT, [1])
        ],
        [
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("out_cast", TensorProto.FLOAT, [1])
        ]
    )

    model = helper.make_model(graph, producer_name="repro")
    return model

def recursive_fix(model):
    def update_cast_ops(graph):
        for node in graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == TensorProto.FLOAT:
                        attr.i = TensorProto.FLOAT16
            # Recursively check subgraphs (If, Loop, Scan)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    update_cast_ops(attr.g)

    update_cast_ops(model.graph)

def check_casts(model):
    results = []
    def _check_graph(graph, prefix=""):
        for node in graph.node:
            if node.op_type == "Cast":
                to_val = next(attr.i for attr in node.attribute if attr.name == "to")
                results.append((f"{prefix}{node.name}", to_val))
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _check_graph(attr.g, prefix + node.name + "/")

    _check_graph(model.graph)
    return results

model = create_model_with_subgraph()
print("Before fix:", check_casts(model))

recursive_fix(model)
print("After recursive fix:", check_casts(model))
