import pytest
from unittest.mock import patch
from trustifai.visualizer import GraphVisualizer
from trustifai.structures import ReasoningGraph, ReasoningNode, ReasoningEdge

@pytest.fixture
def sample_graph():
    nodes = [
        ReasoningNode(node_id="n1", node_type="metric", name="Metric 1", inputs={}, outputs={}, score=0.9, label="Good"),
        ReasoningNode(node_id="decision", node_type="decision", name="Decision", inputs={}, outputs={}, score=0.9, label="Reliable")
    ]
    edges = [ReasoningEdge(source="n1", target="decision", relationship="decides")]
    return ReasoningGraph(trace_id="123", nodes=nodes, edges=edges)

def test_mermaid_visualization(sample_graph):
    viz = GraphVisualizer(sample_graph)
    output = viz.visualize(graph_type="mermaid")
    
    assert "```mermaid" in output
    assert "Metric 1" in output
    assert "n1 --> decision" in output

def test_pyvis_visualization(sample_graph):
    viz = GraphVisualizer(sample_graph)
    
    # Mock pyvis network to avoid HTML generation/browser ops
    with patch("pyvis.network.Network") as MockNet:
        mock_net_instance = MockNet.return_value
        
        viz.visualize(graph_type="pyvis", output_file="test.html")
        
        # Verify nodes and edges were added
        assert mock_net_instance.add_node.call_count == 2
        assert mock_net_instance.add_edge.call_count == 1
        mock_net_instance.save_graph.assert_called_with("test.html")

def test_invalid_visualizer_type(sample_graph):
    viz = GraphVisualizer(sample_graph)
    with pytest.raises(ValueError):
        viz.visualize(graph_type="unknown")