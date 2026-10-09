import math

import pandas as pd

from sharingan.ingest.arcs import arc_for_episode
from sharingan.ingest.jutsus import primary_label
from sharingan.ingest.subtitles import clean_text, flag_song_lines, parse_ass, parse_srt
from sharingan.ingest.transcripts import clean_line

ASS = """[Script Info]
Title: test

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:05.95,0:00:11.99,Default,,0000,0000,0000,,A long time ago,\\Na demon fox appeared.
Dialogue: 0,0:01:04.58,0:01:05.66,Default,,0000,0000,0000,,{\\i1}Naruto!{\\i0}
Comment: 0,0:01:06.00,0:01:07.00,Default,,0000,0000,0000,,not dialogue
"""

SRT = """1
00:00:09,953 --> 00:00:10,749
<i>C'mon!</i>

2
00:00:17,094 --> 00:00:19,722
Running like
a fugitive,
"""


def test_parse_ass_uses_format_header_and_strips_tags():
    rows = parse_ass(ASS)
    assert rows == [(5.95, 11.99, "A long time ago, a demon fox appeared."), (64.58, 65.66, "Naruto!")]


def test_parse_srt_joins_multiline_and_strips_html():
    rows = parse_srt(SRT)
    assert rows[0][2] == "C'mon!"
    assert rows[1][2] == "Running like a fugitive,"
    assert math.isclose(rows[1][0], 17.094)


def test_clean_text_normalises_quotes():
    assert clean_text("You’re   bent\\Nbelieve it") == "You're bent believe it"


def test_song_lines_need_repetition_window_and_length():
    rows = []
    for ep in range(1, 7):
        rows += [
            (ep, 10.0, 12.0, "We are fighting dreamers aiming high"),   # OP lyric
            (ep, 600.0, 602.0, "We are fighting dreamers aiming high"),  # same words mid-episode
            (ep, 20.0, 21.0, "Naruto!"),                                  # short, repeated everywhere
            (ep, 1300.0, 1400.0, f"unique line {ep}"),
        ]
    df = pd.DataFrame(rows, columns=["episode", "start", "end", "text"])
    flags = flag_song_lines(df, min_episodes=5, window_s=150)
    assert flags[df["start"] == 10.0].all()
    assert not flags[df["start"] == 600.0].any()
    assert not flags[df["text"] == "Naruto!"].any()


def test_arcs_cover_part_one():
    assert arc_for_episode(1) == "Prologue"
    assert arc_for_episode(62) == "Chūnin Exams"
    assert arc_for_episode(220) == "Post-Recovery (filler)"
    assert all(arc_for_episode(e) != "Unknown" for e in range(1, 221))


def test_primary_label_priority():
    assert primary_label(["Ninjutsu", "Genjutsu"]) == "Genjutsu"
    assert primary_label(["Kekkei Genkai", "Ninjutsu", "Taijutsu"]) == "Ninjutsu"
    assert primary_label(["Taijutsu"]) == "Taijutsu"
    assert primary_label(["Kenjutsu"]) is None


def test_transcript_stage_directions_removed():
    assert clean_line(" (Laughing) Give it up. (Points) Losers!") == "Give it up. Losers!"
