# visualizer.py
"""Graph visualization module for reasoning graphs"""

import math
from typing import Any, Dict, Tuple, List
from trustifai.structures import ReasoningGraph


class GraphVisualizer:
    COLORS = {
        "reliable_decision": "#2ca05c",
        "acceptable_decision": "#d28304",
        "unreliable_decision": "#e61f09",
        "aggregation": "#4ecdc4",
        "metric_high": "#2ecc7199",
        "metric_medium": "#f39c1299",
        "metric_low": "#e74c3c99",
        "no_score": "#95a5a699",
        "edge_high": "#2ecc71",
        "edge_medium": "#f39c12",
        "edge_low": "#e74c3c",
    }

    SHAPES = {
        "decision": "square",
        "aggregation": "diamond",
        "metric": "dot",
    }

    def __init__(self, graph: ReasoningGraph, config=None):
        self.graph = graph
        self.config = config

    def visualize(self, graph_type: str = "pyvis", **kwargs):
        visualizers = {
            "mermaid": self._mermaid_visualization,
            "pyvis": self._pyvis_visualization,
        }
        if graph_type not in visualizers:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        return visualizers[graph_type](**kwargs)

    def _mermaid_visualization(self) -> str:
        lines = ["```mermaid\nflowchart TD"]
        for node in self.graph.nodes:
            shape_open, shape_close = self._get_mermaid_shape(node.node_type)
            label = self._format_mermaid_label(node)
            lines.append(f'   {node.node_id}{shape_open}"{label}"{shape_close}')
        for edge in self.graph.edges:
            lines.append(f"    {edge.source} --> {edge.target}")
        for node in self.graph.nodes:
            if node.score is not None:
                color = self._get_mermaid_color(node.score, node.label, node.node_id)
                lines.append(f"    style {node.node_id} fill:{color},color:#000000")
        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def _get_mermaid_shape(node_type: str) -> Tuple[str, str]:
        shapes = {
            "metric": ("[", "]"),
            "decision": ("(", ")"),
            "aggregation": ("{", "}"),
        }
        return shapes.get(node_type, ("[", "]"))

    @staticmethod
    def _format_mermaid_label(node) -> str:
        label = f"<b>{node.name}</b>"
        if node.score is not None and node.node_type != "decision":
            label += f"<br/>Score: {node.score:.2f}"
        if node.label and node.node_type not in {"aggregation", "decision"}:
            label += f"<br/>{node.label}"
        return label

    def _get_mermaid_color(self, score: float, label: str, node_id: str = None) -> str:
        if label in {None, "N/A"}:
            return "#95a5a6"
        thresholds = self._get_metric_thresholds(node_id)
        if score >= thresholds[0]:
            return "#2ecc71"
        elif score >= thresholds[1]:
            return "#f39c12"
        else:
            return "#ff6b6b"

    def _pyvis_visualization(self, output_file: str = "reasoning_graph.html"):
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis is required. pip install pyvis")

        net = Network(
            notebook=True,
            directed=True,
            cdn_resources="remote",
            neighborhood_highlight=True,
            width="100%",
        )
        net.set_options(
            """{"physics": {"enabled": true, "solver": "forceAtlas2Based"}}"""
        )

        positions = self._calculate_fixed_positions()
        for node in self.graph.nodes:
            net.add_node(**self._get_node_config(node, positions))
        for edge in self.graph.edges:
            f, t, attrs = self._get_edge_config(edge)
            net.add_edge(f, t, **attrs)

        net.save_graph(output_file)
        return net

    def _calculate_fixed_positions(self) -> Dict[str, Tuple[int, int]]:
        positions = {}
        metric_nodes = [n for n in self.graph.nodes if n.node_type == "metric"]
        radius = 400
        for i, node in enumerate(self.graph.nodes):
            if node.node_type == "decision":
                positions[node.node_id] = (0, 400)
            elif node.node_type == "aggregation":
                positions[node.node_id] = (0, 0)
            elif node.node_type == "metric":
                angle = 2 * math.pi * metric_nodes.index(node) / len(metric_nodes)
                positions[node.node_id] = (
                    int(radius * math.cos(angle)),
                    int(radius * math.sin(angle)),
                )
        return positions

    def _get_node_config(self, node, positions: Dict) -> Dict[str, Any]:
        config = {
            "n_id": node.node_id,
            "label": node.name if node.name else node.label,
            "title": self._format_pyvis_title(node),
            "color": self._get_node_color(node),
            "shape": self.SHAPES[node.node_type],
            "size": 20,
            "borderWidth": 3 if node.node_type == "decision" else 2,
        }
        if node.node_id in positions:
            config["x"], config["y"] = positions[node.node_id]
        return config

    def _get_edge_config(self, edge) -> Tuple[str, str, Dict[str, Any]]:
        source_node = next(n for n in self.graph.nodes if n.node_id == edge.source)
        score = source_node.score
        color = self.COLORS["no_score"]
        if score is not None:
            thresholds = self._get_metric_thresholds(source_node.node_id)
            color = (
                self.COLORS["edge_high"]
                if score >= thresholds[0]
                else (
                    self.COLORS["edge_medium"]
                    if score >= thresholds[1]
                    else self.COLORS["edge_low"]
                )
            )

        return (
            edge.source,
            edge.target,
            {
                "title": edge.relationship,
                "label": edge.relationship,
                "color": color,
                "width": 2 if edge.relationship == "decides" else 1,
            },
        )

    @staticmethod
    def _format_pyvis_title(node) -> str:
        parts = []
        if node.score is not None:
            parts.append(f"Score: {node.score:.2f}")
        if node.label:
            parts.append(f"Label: {node.label}")
        if isinstance(node.details, dict):
            parts.append(f"Reason: {node.details.get('explanation', '')}")
        return "\n".join(parts)

    def _get_node_color(self, node) -> str:
        if node.node_type == "decision":
            return (
                self.COLORS["reliable_decision"]
                if node.label == "RELIABLE"
                else (
                    self.COLORS["acceptable_decision"]
                    if "ACCEPTABLE" in node.label
                    else self.COLORS["unreliable_decision"]
                )
            )
        elif node.node_type == "aggregation":
            return self.COLORS["aggregation"]
        else:
            thresholds = self._get_metric_thresholds(node.node_id)
            return (
                self.COLORS["metric_high"]
                if node.score >= thresholds[0]
                else (
                    self.COLORS["metric_medium"]
                    if node.score >= thresholds[1]
                    else self.COLORS["metric_low"]
                )
            )

    def _get_metric_thresholds(self, node_id: str) -> Tuple[float, float]:
        """Get high/medium thresholds for a metric from config, fallback to defaults"""
        if not self.config or not node_id:
            return (0.8, 0.6)

        metric_config = next(
            (m for m in self.config.metrics if m.type == node_id), None
        )
        if not metric_config:
            return (0.8, 0.6)

        params = metric_config.params or {}
        # Map metric-specific threshold names to (high, medium) tuple
        threshold_map = {
            "evidence_coverage": (
                params.get("STRONG_GROUNDING", 0.85),
                params.get("PARTIAL_GROUNDING", 0.7),
            ),
            "consistency": (
                params.get("STABLE_CONSISTENCY", 0.85),
                params.get("FRAGILE_CONSISTENCY", 0.60),
            ),
            "semantic_drift": (
                params.get("STRONG_ALIGNMENT", 0.85),
                params.get("PARTIAL_ALIGNMENT", 0.60),
            ),
            "source_diversity": (
                params.get("HIGH_DIVERSITY", 0.85),
                params.get("MODERATE_DIVERSITY", 0.60),
            ),
            "trust_aggregation": (
                (
                    self.config.thresholds.RELIABLE_TRUST
                    if hasattr(self.config, "thresholds")
                    else 0.8
                ),
                (
                    self.config.thresholds.ACCEPTABLE_TRUST
                    if hasattr(self.config, "thresholds")
                    else 0.6
                ),
            ),
        }

        return threshold_map.get(node_id, (0.8, 0.6))


class ConfidenceVisualizer:

    @staticmethod
    def _generate_html(sentences: List[Dict]) -> str:
        """
        Generates an HTML snippet with proper CSS tooltips and color-coded sentences.
        """
        if not sentences:
            return ""

        # CSS for professional tooltips
        css_styles = """
        <style>
            .tf-sent-container {
                position: relative;
                display: inline;
                cursor: pointer;
                border-bottom: 2px solid;
                padding-bottom: 1px;
                margin-right: 4px;
                color: #000;
            }
            .tf-sent-container .tf-tooltip {
                visibility: hidden;
                width: 220px;
                background-color: #2c3e50;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 8px 0;
                position: absolute;
                z-index: 999;
                bottom: 125%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.2s;
                font-family: sans-serif;
                font-size: 12px;
                line-height: 1.4;
                box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                pointer-events: none;
            }
            .tf-sent-container .tf-tooltip::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #2c3e50 transparent transparent transparent;
            }
            .tf-sent-container:hover .tf-tooltip {
                visibility: visible;
                opacity: 1;
            }
        </style>
        """

        html_parts = [
            css_styles,
            '<div style="font-family: sans-serif; line-height: 1.6; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">',
            '<h3 style="margin-top:0;">Confidence Per Sentence Scan</h3>',
            '<p style="font-size: 0.9em; color: #666;">Scores [0.0-1.0] indicate Mean confidence. Hover for details.</p>',
        ]

        for sent in sentences:
            score = sent["score"]
            text = sent["text"]
            label = sent.get("label", "N/A")

            label_lower = label.lower()

            if "high confidence" in label_lower:
                hue = 120 * score
            elif "medium confidence" in label_lower:
                hue = 80 * score
            else:
                hue = 10 * score

            bg_color = f"hsla({hue}, 80%, 75%, 1)"
            border_color = f"hsla({hue}, 80%, 55%, 1)"

            span = (
                f'<span class="tf-sent-container" style="background-color: {bg_color}; border-color: {border_color};">'
                f'{text} <sup style="font-size: 0.7em; color: gray; font-weight: bold;">[{score:.2f}]</sup>'
                f'<span class="tf-tooltip">'
                f"<strong>Label:</strong> {label}<br>"
                f'<strong>Confidence:</strong> {score:.2%} ({sent["token_count"]} tokens)'
                f"</span>"
                f"</span>"
            )
            html_parts.append(span)

        html_parts.append("</div>")
        return "".join(html_parts)