import pandas as pd

from sharingan.core import config as config_mod
from sharingan.models.jutsu.data import class_weights, evaluate, make_splits
from sharingan.models.persona.cards import CARDS
from sharingan.models.persona.memory import build_scene_table, build_voice_table


def _jutsu_frame(n=200):
    labels = ["Ninjutsu"] * int(n * 0.8) + ["Taijutsu"] * int(n * 0.15) + ["Genjutsu"] * int(n * 0.05)
    return pd.DataFrame({"text": [f"jutsu {i}" for i in range(len(labels))], "label": labels})


def test_splits_are_stratified_and_disjoint():
    splits = make_splits(_jutsu_frame(), seed=0)
    assert sum(len(v) for v in splits.values()) == 200
    texts = [set(v["text"]) for v in splits.values()]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])
    for split in splits.values():
        assert set(split["label"]) == {"Ninjutsu", "Taijutsu", "Genjutsu"}


def test_class_weights_upweight_rare_classes():
    w = dict(zip(["Genjutsu", "Ninjutsu", "Taijutsu"],
                 class_weights(_jutsu_frame()["label"], ["Genjutsu", "Ninjutsu", "Taijutsu"])))
    assert w["Genjutsu"] > w["Taijutsu"] > w["Ninjutsu"]


def test_evaluate_reports_macro_f1():
    r = evaluate(["A", "A", "B"], ["A", "B", "B"], ["A", "B"])
    assert r["accuracy"] == round(2 / 3, 4)
    assert r["confusion_matrix"]["matrix"] == [[1, 1], [0, 1]]


def test_voice_table_uses_transcript_and_signatures():
    dialogue = pd.DataFrame({
        "episode": [1, 1, 1], "arc": ["P"] * 3, "line_idx": [0, 1, 2],
        "text": ["Want some ramen?", "Ramen is the best, believe it!", "What a drag..."],
    })
    transcript = pd.DataFrame({"speaker": ["Iruka", "Naruto"], "line": ["Go home.", "No way, I am staying here!"],
                               "prev_line": [None, "Go home."]})
    voice = build_voice_table(dialogue, transcript)
    naruto = voice[voice["character"] == "Naruto"]
    assert set(naruto["source"]) == {"transcript", "subtitle-signature"}
    assert naruto[naruto["source"] == "subtitle-signature"]["context"].iat[0] == "Want some ramen?"
    assert (voice["character"] == "Shikamaru").any()


def test_scene_windows_overlap():
    dialogue = pd.DataFrame({"episode": [1] * 12, "arc": ["P"] * 12, "text": [f"l{i}" for i in range(12)]})
    scenes = build_scene_table(dialogue)
    assert scenes["text"].iat[0].startswith("l0") and scenes["text"].iat[1].startswith("l4")


def test_system_prompt_includes_retrieval():
    prompt = CARDS["Naruto"].system_prompt(voice=["Believe it!"], scenes=["(Episode 1) ..."])
    assert "Naruto Uzumaki" in prompt and "Believe it!" in prompt and "Episode 1" in prompt


def test_env_override(monkeypatch):
    monkeypatch.setenv("SHARINGAN__THEMES__BATCH_SIZE", "7")
    monkeypatch.setenv("SHARINGAN__THEMES__LABELS", "[hope, love]")
    config_mod.get_settings.cache_clear()
    try:
        s = config_mod.get_settings()
        assert s.themes.batch_size == 7 and s.themes.labels == ["hope", "love"]
    finally:
        config_mod.get_settings.cache_clear()
