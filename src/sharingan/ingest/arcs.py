"""Story-arc boundaries for Naruto (Part I, episodes 1-220).

Aggregating by arc gives the theme and network analyses a narrative axis
instead of a flat list of 220 episodes.
"""

from __future__ import annotations

ARCS: list[tuple[int, int, str]] = [
    (1, 5, "Prologue"),
    (6, 19, "Land of Waves"),
    (20, 67, "Chūnin Exams"),
    (68, 80, "Konoha Crush"),
    (81, 100, "Search for Tsunade"),
    (101, 106, "Land of Tea (filler)"),
    (107, 135, "Sasuke Recovery Mission"),
    (136, 220, "Post-Recovery (filler)"),
]

ARC_ORDER = [name for _, _, name in ARCS]


def arc_for_episode(episode: int) -> str:
    for first, last, name in ARCS:
        if first <= episode <= last:
            return name
    return "Unknown"
