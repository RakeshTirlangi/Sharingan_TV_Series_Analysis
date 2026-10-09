"""Pure functions that turn a jutsu article's wikitext into structured fields.

Kept free of Scrapy so they are trivially unit-testable.
"""

from __future__ import annotations

import re

import mwparserfromhell

INFOBOX_NAME = "infobox/jutsu"
_STOP_SECTIONS = re.compile(r"^(trivia|references|see also|notes|in other media|gallery)$", re.I)
_WS_RE = re.compile(r"\s+")
_QUALIFIER_RE = re.compile(r"~\w+$")             # "Iruka Umino~anime" -> "Iruka Umino"
_BRACKET_NOTE_RE = re.compile(r"\s*\([^)]*\)$")   # "Hiruko (missing-nin)" -> "Hiruko"
_NON_PROSE_LINK_RE = re.compile(r"^\s*(file|image|category|[a-z]{2,3}(-[a-z]+)?):", re.I)  # media + interlanguage


def _plain(value: str) -> str:
    code = mwparserfromhell.parse(value)
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() == "ref":
            try:
                code.remove(tag)
            except ValueError:
                pass
    for comment in code.filter_comments():
        code.remove(comment)
    return _WS_RE.sub(" ", code.strip_code(normalize=True, collapse=True)).strip()


def _split_list(value: str, *, strip_notes: bool = False) -> list[str]:
    items = []
    for part in _plain(value).split(","):
        part = _QUALIFIER_RE.sub("", part.strip())
        if strip_notes:
            part = _BRACKET_NOTE_RE.sub("", part)
        if part and part not in items:
            items.append(part)
    return items


def parse_infobox(wikitext: str) -> dict:
    code = mwparserfromhell.parse(wikitext)
    box = next(
        (t for t in code.filter_templates(recursive=False) if str(t.name).strip().lower() == INFOBOX_NAME),
        None,
    )
    if box is None:
        return {}
    get = lambda key: str(box.get(key).value) if box.has(key) else ""  # noqa: E731
    return {
        "classification": _split_list(get("jutsu classification")),
        "nature": _split_list(get("jutsu type")),
        "rank": _plain(get("jutsu rank")) or None,
        "class_type": _plain(get("jutsu class type")) or None,
        "range": _plain(get("jutsu range")) or None,
        "hand_signs": _split_list(get("hand signs")),
        "users": _split_list(get("users"), strip_notes=True),
        "debut_anime": _plain(get("debut anime")) or None,
    }


def parse_description(wikitext: str) -> str:
    """Article prose (lead + body) with templates, refs, files and trivia removed."""
    code = mwparserfromhell.parse(wikitext)
    keep = []
    for section in code.get_sections(levels=None, include_lead=True, flat=True):
        headings = section.filter_headings()
        if headings and _STOP_SECTIONS.match(headings[0].title.strip_code().strip()):
            continue
        for h in headings:
            section.remove(h)
        for tpl in section.filter_templates(recursive=False):
            section.remove(tpl)
        for link in section.filter_wikilinks():
            if _NON_PROSE_LINK_RE.match(str(link.title)):
                section.remove(link)
        keep.append(_plain(str(section)))
    return _WS_RE.sub(" ", " ".join(keep)).strip()
