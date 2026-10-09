"""Weighted character co-occurrence graph and its analytics.

Two characters are linked when they are mentioned within ``window`` subtitle lines of
each other in the same episode. Each co-occurrence contributes ``exp(-d / tau)`` where
``d`` is the line distance, so names in the same exchange count far more than names a
scene apart (a flat count over a window treats both the same).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd


@dataclass
class CharacterNetwork:
    graph: nx.Graph
    edges: pd.DataFrame
    metrics: pd.DataFrame

    # ------------------------------------------------------------------ build
    @classmethod
    def from_mentions(cls, mentions: pd.DataFrame, *, window: int = 10, tau: float = 4.0,
                      min_mentions: int = 8, arc: str | None = None, seed: int = 7) -> "CharacterNetwork":
        if arc:
            mentions = mentions[mentions["arc"] == arc]
        counts = mentions["character"].value_counts()
        keep = set(counts[counts >= min_mentions].index)
        m = mentions[mentions["character"].isin(keep)].sort_values(["episode", "line_idx"])

        weight, cooc, episodes = defaultdict(float), defaultdict(int), defaultdict(set)
        for episode, group in m.groupby("episode", sort=False):
            lines = group["line_idx"].to_numpy()
            chars = group["character"].to_numpy()
            lo = 0
            for i in range(len(lines)):
                while lines[i] - lines[lo] > window:
                    lo += 1
                for j in range(lo, i):
                    if chars[i] == chars[j]:
                        continue
                    key = tuple(sorted((chars[i], chars[j])))
                    weight[key] += math.exp(-(lines[i] - lines[j]) / tau)
                    cooc[key] += 1
                    episodes[key].add(episode)

        edges = pd.DataFrame(
            [(a, b, weight[(a, b)], cooc[(a, b)], len(episodes[(a, b)])) for a, b in weight],
            columns=["source", "target", "weight", "cooccurrences", "episodes"],
        ).sort_values("weight", ascending=False, ignore_index=True)

        graph = nx.Graph()
        for name in keep:
            graph.add_node(name, mentions=int(counts[name]))
        for row in edges.itertuples(index=False):
            graph.add_edge(row.source, row.target, weight=row.weight,
                           distance=1.0 / row.weight, cooccurrences=row.cooccurrences)
        graph.remove_nodes_from([n for n, d in graph.degree() if d == 0])
        return cls(graph=graph, edges=edges, metrics=cls._metrics(graph, seed))

    @staticmethod
    def _metrics(graph: nx.Graph, seed: int) -> pd.DataFrame:
        if graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["character", "mentions", "degree", "strength", "pagerank",
                                         "betweenness", "eigenvector", "community"])
        communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
        community_of = {n: i for i, group in enumerate(sorted(communities, key=len, reverse=True))
                        for n in group}
        try:
            eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
        except Exception:  # disconnected / degenerate graphs
            eigen = dict.fromkeys(graph, 0.0)
        pagerank = nx.pagerank(graph, weight="weight")
        between = nx.betweenness_centrality(graph, weight="distance", normalized=True)
        strength = dict(graph.degree(weight="weight"))
        return (
            pd.DataFrame({
                "character": list(graph.nodes),
                "mentions": [graph.nodes[n]["mentions"] for n in graph],
                "degree": [graph.degree(n) for n in graph],
                "strength": [strength[n] for n in graph],
                "pagerank": [pagerank[n] for n in graph],
                "betweenness": [between[n] for n in graph],
                "eigenvector": [eigen[n] for n in graph],
                "community": [community_of[n] for n in graph],
            })
            .sort_values("pagerank", ascending=False, ignore_index=True)
        )

    # --------------------------------------------------------------- queries
    def strongest_ties(self, character: str, k: int = 8) -> pd.DataFrame:
        e = self.edges[(self.edges["source"] == character) | (self.edges["target"] == character)].head(k)
        other = e["target"].where(e["source"] == character, e["source"])
        return pd.DataFrame({"character": other.values, "weight": e["weight"].round(1).values,
                             "episodes": e["episodes"].values})

    def to_html(self, path: Path | None = None, *, top_edges: int = 250) -> str:
        from .render import render_network

        html = render_network(self, top_edges=top_edges)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html
