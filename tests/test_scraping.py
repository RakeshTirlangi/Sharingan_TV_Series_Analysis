from sharingan.scraping.wikitext import parse_description, parse_infobox

WIKITEXT = """{{Infobox/Jutsu
|ref=<ref>''[[Rin no Sho]]'', page 222</ref>
|jutsu rank=D
|jutsu classification=Genjutsu
|jutsu type=Yin Release
|jutsu class type=Supplementary
|users=Kakashi Hatake, Iruka Umino~anime, Hiruko (missing-nin)~movie
|hand signs=Snake, Rat
|debut anime=5
}}
This technique subjects targets to visions of their [[fear|greatest fear]].<ref>source</ref>
{{Quote|ignored template}}
[[File:Hell Viewing.png|thumb|caption]]

== Overview ==
It makes them feel real.

== Trivia ==
* This should be dropped.

[[es:Ilusión Demoníaca]]
"""


def test_infobox_fields():
    box = parse_infobox(WIKITEXT)
    assert box["classification"] == ["Genjutsu"]
    assert box["nature"] == ["Yin Release"]
    assert box["rank"] == "D"
    assert box["users"] == ["Kakashi Hatake", "Iruka Umino", "Hiruko"]
    assert box["hand_signs"] == ["Snake", "Rat"]
    assert box["debut_anime"] == "5"


def test_description_keeps_prose_only():
    desc = parse_description(WIKITEXT)
    assert desc == "This technique subjects targets to visions of their greatest fear. It makes them feel real."


def test_no_infobox_returns_empty():
    assert parse_infobox("Just a page.") == {}
