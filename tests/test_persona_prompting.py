import numpy as np
import pandas as pd

from sharingan.models.persona.engine import DEMO_TURNS, PersonaEngine
from sharingan.models.persona.memory import PersonaMemory


class FakeEncoder:
    """Bag-of-letters embedding: deterministic and good enough to rank by overlap."""

    def encode(self, texts, normalize_embeddings=True, **_):
        vecs = np.array([[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts], dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6)


def _memory():
    voice = pd.DataFrame({
        "character": ["Naruto", "Naruto", "Naruto", "Sasuke"],
        "context": ["Want ramen?", "Who are you?", "", "Fight me."],
        "line": ["Ramen is the best, believe it!", "I'm Naruto Uzumaki!", "Believe it!", "Hmph."],
        "source": ["subtitle-signature"] * 4,
    })
    scenes = pd.DataFrame({"episode": [1, 2], "arc": ["Prologue"] * 2,
                           "text": ["Naruto eats ramen at Ichiraku.", "Kakashi is late again."]})
    enc = FakeEncoder()
    return PersonaMemory(voice, scenes, enc.encode((voice["context"] + " " + voice["line"]).tolist()),
                         enc.encode(scenes["text"].tolist()), enc)


def test_recall_exchanges_filters_character_and_ranks():
    ex = _memory().recall_exchanges("Naruto", "ramen", k=2)
    assert len(ex) == 2 and ex[0] == ("Want ramen?", "Ramen is the best, believe it!")
    assert all(line != "Hmph." for _, line in ex)


def test_prepare_replays_demo_turns_and_restates_voice():
    engine = object.__new__(PersonaEngine)          # skip model loading
    engine.memory, engine.k_voice, engine.k_scenes = _memory(), 2, 1
    messages, retrieved = engine.prepare("Naruto", "Do you like ramen?", [])
    roles = [m["role"] for m in messages]
    assert roles[0] == "system" and roles[-1] == "user" and messages[-1]["content"] == "Do you like ramen?"
    demos = messages[1:-1]
    assert 0 < len(demos) <= 2 * DEMO_TURNS and roles[1:-1] == ["user", "assistant"] * (len(demos) // 2)
    assert all(d["content"] for d in demos)                    # empty contexts are never replayed
    assert messages[0]["content"].rstrip().endswith(messages[0]["content"].split("Above all, sound like Naruto: ")[1])
    assert retrieved["scenes"][0].startswith("(Episode 1, Prologue)")


def test_verdict_rejects_catchphrase_parroting():
    from sharingan.models.persona.evaluate import verdict

    base = {"distinctiveness": 0.12, "relevance": 0.42, "words": 18.8, "catchphrase_rate": 0.6}
    parrot = {"distinctiveness": 0.23, "relevance": 0.20, "words": 6.9, "catchphrase_rate": 0.9}
    good = {"distinctiveness": 0.18, "relevance": 0.41, "words": 15.0, "catchphrase_rate": 0.5}
    assert verdict(base, parrot)["recommend_adapter"] is False
    assert verdict(base, good)["recommend_adapter"] is True
