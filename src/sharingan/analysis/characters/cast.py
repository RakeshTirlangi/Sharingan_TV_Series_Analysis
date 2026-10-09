"""Canonical cast list for Naruto Part I with the aliases used in the English subs.

Used two ways: as an ``EntityRuler`` (high-precision recall boost for Japanese names a
general-domain NER model misses) and as an alias table that collapses surface forms
("Sasuke-kun", "Uchiha", "Pervy Sage") onto one node.
"""

from __future__ import annotations

import re

CAST: dict[str, list[str]] = {
    "Naruto": ["Naruto", "Naruto Uzumaki", "Uzumaki Naruto"],
    "Sasuke": ["Sasuke", "Sasuke Uchiha", "Uchiha Sasuke"],
    "Sakura": ["Sakura", "Sakura Haruno", "Haruno Sakura"],
    "Kakashi": ["Kakashi", "Kakashi Hatake", "Hatake Kakashi", "Kakashi Sensei"],
    "Iruka": ["Iruka", "Iruka Umino", "Iruka Sensei"],
    "Hiruzen": ["Hiruzen", "Hiruzen Sarutobi", "Third Hokage", "Lord Third"],
    "Konohamaru": ["Konohamaru"],
    "Ebisu": ["Ebisu"],
    "Mizuki": ["Mizuki"],
    "Zabuza": ["Zabuza", "Zabuza Momochi"],
    "Haku": ["Haku"],
    "Tazuna": ["Tazuna"],
    "Inari": ["Inari"],
    "Gato": ["Gato", "Gatō"],
    "Rock Lee": ["Rock Lee", "Lee"],
    "Neji": ["Neji", "Neji Hyuga", "Neji Hyūga"],
    "Tenten": ["Tenten"],
    "Guy": ["Guy", "Might Guy", "Gai", "Guy Sensei"],
    "Hinata": ["Hinata", "Hinata Hyuga", "Hinata Hyūga"],
    "Kiba": ["Kiba", "Kiba Inuzuka"],
    "Akamaru": ["Akamaru"],
    "Shino": ["Shino", "Shino Aburame"],
    "Kurenai": ["Kurenai", "Kurenai Sensei"],
    "Shikamaru": ["Shikamaru", "Shikamaru Nara"],
    "Choji": ["Choji", "Chōji", "Choji Akimichi", "Chouji"],
    "Ino": ["Ino", "Ino Yamanaka"],
    "Asuma": ["Asuma", "Asuma Sarutobi", "Asuma Sensei"],
    "Gaara": ["Gaara"],
    "Kankuro": ["Kankuro", "Kankurō"],
    "Temari": ["Temari"],
    "Baki": ["Baki"],
    "Orochimaru": ["Orochimaru", "Lord Orochimaru"],
    "Kabuto": ["Kabuto", "Kabuto Yakushi"],
    "Jiraiya": ["Jiraiya", "Pervy Sage", "Pervy Hermit", "Ero-Sennin", "Pervert Sage"],
    "Tsunade": ["Tsunade", "Lady Tsunade", "Granny Tsunade", "Fifth Hokage"],
    "Shizune": ["Shizune"],
    "Itachi": ["Itachi", "Itachi Uchiha"],
    "Kisame": ["Kisame", "Kisame Hoshigaki"],
    "Hayate": ["Hayate", "Hayate Gekko"],
    "Anko": ["Anko", "Anko Mitarashi"],
    "Ibiki": ["Ibiki", "Ibiki Morino"],
    "Hiashi": ["Hiashi"],
    "Hanabi": ["Hanabi"],
    "Dosu": ["Dosu"],
    "Zaku": ["Zaku"],
    "Kin": ["Kin"],
    "Kimimaro": ["Kimimaro"],
    "Tayuya": ["Tayuya"],
    "Kidomaru": ["Kidomaru", "Kidōmaru"],
    "Jirobo": ["Jirobo", "Jirōbō"],
    "Sakon": ["Sakon"],
    "Ukon": ["Ukon"],
    "Minato": ["Minato", "Fourth Hokage"],
    "Pakkun": ["Pakkun"],
    "Gamabunta": ["Gamabunta", "Chief Toad"],
    "Kyuubi": ["Nine-Tailed Fox", "Nine Tails", "Kyuubi", "Nine-Tails"],
    "Teuchi": ["Teuchi"],
    "Ayame": ["Ayame"],
    "Tonton": ["Tonton"],
    "Tsume": ["Tsume"],
    "Hana": ["Hana"],
    "Shikaku": ["Shikaku"],
    "Inoichi": ["Inoichi"],
    "Choza": ["Choza", "Chōza"],
    "Kotetsu": ["Kotetsu"],
    "Izumo": ["Izumo"],
    "Genma": ["Genma"],
    "Raido": ["Raido"],
    "Shiranui": ["Shiranui"],
    "Yashamaru": ["Yashamaru"],
    "Kazekage": ["Kazekage", "Lord Kazekage"],
}

HONORIFIC_RE = re.compile(
    r"[-\s]?(kun|chan|sama|san|sensei|senpai|sempai|dono|nee|neesan|niisan|nii|baa-chan)$", re.I
)
TITLE_PREFIX_RE = re.compile(r"^(lord|lady|master|sensei|miss|mister|mr\.?|ms\.?|big brother|brother)\s+", re.I)
POSSESSIVE_RE = re.compile(r"('s|')$")

# Words a general NER model tags as PERSON in this domain that are not characters.
STOPLIST = {
    "hokage", "jutsu", "sharingan", "byakugan", "chunin", "chūnin", "genin", "jonin", "jōnin",
    "sensei", "ninja", "shinobi", "kunai", "shuriken", "ramen", "believe", "dattebayo", "hey",
    "oh", "huh", "kage", "anbu", "rasengan", "chidori", "chakra", "village", "leaf", "sand",
    "sound", "mist", "father", "mother", "brother", "sister", "granny", "old man", "kid",
    "you", "me", "him", "her", "god", "lord", "lady", "bushy brows", "bushy brow", "fuzzy brows",
    # clan names are ambiguous (Sasuke vs Itachi, Neji vs Hinata ...)
    "uchiha", "hyuga", "hyūga", "uzumaki", "nara", "akimichi", "yamanaka", "inuzuka", "aburame", "sarutobi",
    # titles, concepts, places and techniques the model tags as PERSON (found by auditing outputs)
    "sannin", "kekkei genkai", "dojo", "ichiraku", "kujaku", "jinchuriki", "jinchūriki",
    # interjections the model occasionally promotes to PERSON
    "humph", "hmph", "hmm", "ugh", "whoa", "yikes", "geez", "jeez", "darn", "dammit", "ow", "ouch",
}


def alias_table() -> dict[str, str]:
    """lower-cased surface form -> canonical name"""
    table = {}
    for canonical, aliases in CAST.items():
        for alias in aliases:
            table[alias.lower()] = canonical
    return table


def ruler_patterns() -> list[dict]:
    """Case-sensitive phrase patterns ("Guy" the sensei, not "that guy")."""
    return [
        {"label": "PERSON", "pattern": alias, "id": canonical}
        for canonical, aliases in CAST.items()
        for alias in aliases
    ]
