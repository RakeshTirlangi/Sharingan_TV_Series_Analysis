"""A/B evaluation of a character's LoRA adapter against the base model.

Both arms get identical retrieval, prompts and sampling seeds; only the adapter differs.
Automatic proxies (no human labels needed):

* ``voice_similarity``  cosine(reply, centroid of the character's real lines)
* ``distinctiveness``   voice_similarity minus mean similarity to *other* characters' centroids,
                        so generic anime-sounding text does not score well
* ``relevance``         cosine(reply, prompt): does the reply actually address the question?
* ``catchphrase_rate``  share of replies matching the card's signature (too high = parroting)
* ``words``             mean reply length

Voice metrics alone can be gamed: a model that answers everything with a short catchphrase
"sounds like" the character. The verdict therefore also requires relevance and length to hold up.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .cards import CARDS
from .engine import PersonaEngine

HELD_OUT_PROMPTS = [
    "What do you think of Sasuke?",
    "Tell me about Iruka Sensei.",
    "What's your favourite food?",
    "How did it feel when the villagers ignored you?",
    "Kakashi Sensei is late again. What do you do?",
    "Can you teach me the Shadow Clone Jutsu?",
    "What would you do if a friend was in danger?",
    "Who is the strongest ninja you know?",
    "What do you think about Neji?",
    "Are you scared of anything?",
]


def _centroids(engine: PersonaEngine) -> dict[str, np.ndarray]:
    voice, emb = engine.memory.voice, engine.memory.voice_emb
    out = {}
    for name in voice["character"].unique():
        c = emb[(voice["character"] == name).to_numpy()].mean(0)
        out[name] = c / np.linalg.norm(c)
    return out


def evaluate_adapter(engine: PersonaEngine, character: str, prompts: list[str] | None = None,
                     seed: int = 7) -> dict:
    prompts = prompts or HELD_OUT_PROMPTS
    if character not in engine.adapters:
        raise ValueError(f"No LoRA adapter loaded for {character}")
    card = CARDS[character]
    centroids = _centroids(engine)
    others = [c for n, c in centroids.items() if n != character]
    signature = re.compile(card.signature, re.I) if card.signature else None

    arms: dict[str, list[str]] = {"base": [], "lora": []}
    for prompt in prompts:
        messages, _ = engine.prepare(character, prompt, [])
        for arm in arms:
            torch.manual_seed(seed)
            arms[arm].append(engine.reply(character, prompt, messages=messages, use_adapter=arm == "lora"))

    prompt_emb = engine.memory.encoder.encode(prompts, normalize_embeddings=True)

    def score(replies: list[str]) -> dict:
        emb = engine.memory.encoder.encode(replies, normalize_embeddings=True)
        own = emb @ centroids[character]
        rest = np.mean([emb @ c for c in others], axis=0) if others else np.zeros(len(replies))
        return {
            "voice_similarity": round(float(own.mean()), 4),
            "distinctiveness": round(float((own - rest).mean()), 4),
            "relevance": round(float((emb * prompt_emb).sum(1).mean()), 4),
            "catchphrase_rate": round(float(np.mean([bool(signature.search(r)) for r in replies])), 3)
            if signature else None,
            "words": round(float(np.mean([len(r.split()) for r in replies])), 1),
        }

    metrics = {arm: score(replies) for arm, replies in arms.items()}
    return {
        "character": character, "prompts": len(prompts),
        "metrics": metrics, "verdict": verdict(metrics["base"], metrics["lora"]),
        "samples": [{"prompt": p, "base": b, "lora": l} for p, b, l in zip(prompts, arms["base"], arms["lora"])],
    }


def verdict(base: dict, lora: dict) -> dict:
    """Recommend the adapter only if it sounds more in character without getting less useful."""
    checks = {
        "more_distinctive": lora["distinctiveness"] > base["distinctiveness"],
        "relevance_kept": lora["relevance"] >= 0.9 * base["relevance"],
        "substance_kept": lora["words"] >= 0.5 * base["words"],
        "not_parroting": lora["catchphrase_rate"] is None or lora["catchphrase_rate"] <= 0.75,
    }
    return {"recommend_adapter": all(checks.values()), "checks": checks}
