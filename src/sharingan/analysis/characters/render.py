"""Standalone, self-contained HTML export of a ``CharacterNetwork`` (PyVis / vis.js).

The React site draws the graph itself from ``/api/characters``; this export is for
sharing a single offline file (``outputs/characters/network.html``)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from pyvis.network import Network

if TYPE_CHECKING:
    from .graph import CharacterNetwork

# Validated categorical slots (dark-surface steps), assigned to communities in fixed
# order by size. Communities beyond the eighth fold into a neutral "other" grey.
COMMUNITY_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_COLOR = "#8a8984"
SURFACE = "#1a1a19"
TEXT = "#ffffff"
EDGE = "rgba(195,194,183,0.35)"

_OPTIONS = {
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -70, "springLength": 120, "springConstant": 0.06,
                             "avoidOverlap": 0.6},
        "stabilization": {"iterations": 250},
    },
    "nodes": {"shape": "dot", "borderWidth": 2, "color": {"border": SURFACE},
              "font": {"color": TEXT, "face": "Inter, system-ui, sans-serif", "strokeWidth": 3,
                       "strokeColor": SURFACE}},
    "edges": {"color": {"color": EDGE, "highlight": "#ffffff"}, "smooth": {"type": "continuous"},
              "selectionWidth": 2},
    "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": False, "multiselect": True},
}


def community_color(community: int) -> str:
    return COMMUNITY_COLORS[community] if community < len(COMMUNITY_COLORS) else OTHER_COLOR


def render_network(network: "CharacterNetwork", *, top_edges: int = 250, height: str = "720px") -> str:
    edges = network.edges.head(top_edges)
    nodes = set(edges["source"]) | set(edges["target"])
    metrics = network.metrics.set_index("character")
    max_pr = metrics["pagerank"].max() if len(metrics) else 1.0
    max_w = edges["weight"].max() if len(edges) else 1.0

    net = Network(height=height, width="100%", bgcolor=SURFACE, font_color=TEXT, cdn_resources="in_line")
    for name in sorted(nodes, key=lambda n: -metrics.at[n, "pagerank"]):
        row = metrics.loc[name]
        size = 10 + 40 * math.sqrt(row["pagerank"] / max_pr)
        tooltip = (f"{name}\nMentions: {int(row['mentions'])}\nConnections: {int(row['degree'])}\n"
                   f"PageRank: {row['pagerank']:.3f}\nBetweenness: {row['betweenness']:.3f}\n"
                   f"Community: {int(row['community']) + 1}")
        net.add_node(name, label=name, title=tooltip, size=size,
                     color={"background": community_color(int(row["community"])), "border": SURFACE},
                     font={"size": int(12 + 10 * row["pagerank"] / max_pr)})
    for e in edges.itertuples(index=False):
        net.add_edge(e.source, e.target, value=e.weight, width=0.5 + 6 * (e.weight / max_w) ** 0.6,
                     title=f"{e.source} – {e.target}\nStrength: {e.weight:.1f}\n"
                           f"Co-occurrences: {e.cooccurrences}\nShared episodes: {e.episodes}")
    net.set_options(json.dumps(_OPTIONS))
    return net.generate_html(notebook=False)

