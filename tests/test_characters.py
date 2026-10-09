import math

import pandas as pd
import pytest

from sharingan.analysis.characters.cast import alias_table
from sharingan.analysis.characters.graph import CharacterNetwork
from sharingan.analysis.characters.ner import normalise_name

ALIASES = alias_table()


@pytest.mark.parametrize("surface, expected", [
    ("Sasuke-kun", "Sasuke"),
    ("Pervy Sage", "Jiraiya"),
    ("Kakashi Sensei", "Kakashi"),
    ("Lady Tsunade", "Tsunade"),
    ("Neji's", "Neji"),
    ("NARUTO", "Naruto"),
    ("Lee", "Rock Lee"),
    ("Sakur--", "Sakura"),
    ("Gennai", "Gennai"),        # unknown but name-like -> kept
    ("Transform", None),         # common English word -> dropped
    ("Uchiha", None),            # ambiguous clan name -> dropped
    ("Humph", None),
])
def test_normalise_name(surface, expected):
    assert normalise_name(surface, ALIASES) == expected


def _mentions(rows):
    return pd.DataFrame(rows, columns=["episode", "arc", "line_idx", "character"]).assign(start=0.0)


def test_decay_weighting_and_window():
    m = _mentions([
        (1, "A", 0, "Naruto"), (1, "A", 1, "Sasuke"),     # distance 1
        (1, "A", 20, "Sakura"),                           # outside window of both
        (1, "A", 23, "Naruto"),                           # distance 3 from Sakura
    ])
    net = CharacterNetwork.from_mentions(m, window=10, tau=4.0, min_mentions=1)
    w = {tuple(sorted((r.source, r.target))): r.weight for r in net.edges.itertuples()}
    assert math.isclose(w[("Naruto", "Sasuke")], math.exp(-1 / 4))
    assert math.isclose(w[("Naruto", "Sakura")], math.exp(-3 / 4))
    assert ("Sakura", "Sasuke") not in w


def test_no_cross_episode_links_and_min_mentions():
    m = _mentions([(1, "A", 0, "Naruto"), (2, "A", 0, "Sasuke"), (2, "A", 1, "Naruto"), (2, "A", 2, "Haku")])
    net = CharacterNetwork.from_mentions(m, window=10, min_mentions=2)
    assert set(net.graph.nodes) == {"Naruto"} or net.graph.number_of_nodes() == 0
    net_all = CharacterNetwork.from_mentions(m, window=10, min_mentions=1)
    assert net_all.graph.has_edge("Naruto", "Sasuke")
    assert net_all.edges.set_index(["source", "target"]).loc[("Naruto", "Sasuke"), "episodes"] == 1


def test_metrics_columns_and_ranking():
    rows = [(1, "A", i, name) for i, name in enumerate(["Naruto", "Sasuke", "Naruto", "Sakura", "Naruto", "Kakashi"])]
    net = CharacterNetwork.from_mentions(_mentions(rows), min_mentions=1)
    assert net.metrics.iloc[0]["character"] == "Naruto"
    assert {"pagerank", "betweenness", "eigenvector", "community"} <= set(net.metrics.columns)
    assert "<html" in net.to_html().lower()
