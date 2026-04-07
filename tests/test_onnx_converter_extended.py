# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

# Mock modal before importing the worker
sys.modules["modal"] = MagicMock()

import onnx
from onnx import TensorProto, helper

from ai_workers.workers.onnx_converter import _fix_cast_nodes


def test_fix_cast_nodes_recursive():
    """Test that _fix_cast_nodes recursively updates Cast nodes in subgraphs."""
    # 1. Top-level Cast node (FLOAT -> FLOAT16)
    node1 = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT)

    # 2. Subgraph with a Cast node (Attribute type GRAPH)
    sub_node = helper.make_node("Cast", ["A"], ["B"], to=TensorProto.FLOAT)
    sub_graph = helper.make_graph(
        [sub_node],
        "sub",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])],
    )

    # If node with subgraphs in attributes
    # We must add it to a graph for it to be mutable in some ONNX versions/wrappers
    main_graph = helper.make_graph(
        [node1],
        "main",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("in", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1]),
        ],
    )

    node2 = helper.make_node("If", ["cond"], ["out"], then_branch=sub_graph, else_branch=sub_graph)
    main_graph.node.extend([node2])

    # 3. Dummy node with Attribute type GRAPHS
    sub_node_2 = helper.make_node("Cast", ["C"], ["D"], to=TensorProto.FLOAT)
    sub_graph_2 = helper.make_graph(
        [sub_node_2],
        "sub2",
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("D", TensorProto.FLOAT, [1])],
    )

    node3 = onnx.NodeProto()
    node3.op_type = "CustomOp"
    node3.input.extend(["in"])
    node3.output.extend(["out2"])
    attr3 = onnx.AttributeProto()
    attr3.name = "sub_graphs"
    attr3.type = onnx.AttributeProto.GRAPHS
    attr3.graphs.extend([sub_graph_2])
    node3.attribute.extend([attr3])
    main_graph.node.extend([node3])

    _fix_cast_nodes(main_graph)

    # Verify top-level
    found_top_cast = False
    for attr in main_graph.node[0].attribute:
        if attr.name == "to":
            assert attr.i == TensorProto.FLOAT16
            found_top_cast = True
    assert found_top_cast

    # Verify subgraphs in 'If' node
    node2_actual = main_graph.node[1]
    for attr in node2_actual.attribute:
        if attr.name in ["then_branch", "else_branch"]:
            assert attr.g is not None
            found_sub_cast = False
            for sub_node in attr.g.node:
                if sub_node.op_type == "Cast":
                    for sub_attr in sub_node.attribute:
                        if sub_attr.name == "to":
                            assert sub_attr.i == TensorProto.FLOAT16
                            found_sub_cast = True
            assert found_sub_cast

    # Verify subgraphs in 'CustomOp' node (GRAPHS type)
    node3_actual = main_graph.node[2]
    attr3_actual = node3_actual.attribute[0]
    sub_node_2_actual = attr3_actual.graphs[0].node[0]
    assert sub_node_2_actual.attribute[0].i == TensorProto.FLOAT16


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
