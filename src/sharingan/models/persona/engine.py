"""Retrieval-augmented character chat engine with optional per-character LoRA adapters."""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ...core import get_logger, pick_device
from .cards import CARDS
from .memory import PersonaMemory

log = get_logger(__name__)
MAX_HISTORY_TURNS = 6
DEMO_TURNS = 2  # retrieved real exchanges replayed as earlier turns (few-shot style anchors)


def _text(content) -> str:
    """Chat clients may send message content as a str or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def load_causal_lm(name: str, device: str):
    """bf16 on GPU (4-bit NF4 for >=7B when bitsandbytes is available), fp32 on CPU."""
    kwargs: dict = {"dtype": torch.float32}
    if device == "cuda":
        kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
        if any(tag in name.lower() for tag in ("7b", "8b", "13b", "14b")):
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            except ImportError:
                pass
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device != "cuda":
        model.to(device)
    return model.eval()


class PersonaEngine:
    def __init__(self, base_model: str, memory: PersonaMemory, *, adapter_root: Path | None = None,
                 retrieved_examples: int = 4, max_new_tokens: int = 160, temperature: float = 0.7,
                 top_p: float = 0.9):
        self.device = pick_device()
        self.memory = memory
        self.k_voice, self.k_scenes = retrieved_examples, 3
        self.gen = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                        do_sample=temperature > 0, repetition_penalty=1.1)
        log.info(f"Loading persona base model {base_model} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = load_causal_lm(base_model, self.device)
        self.adapters: set[str] = set()
        if adapter_root:
            self._load_adapters(Path(adapter_root))

    def _load_adapters(self, root: Path) -> None:
        dirs = [d for d in sorted(root.glob("*")) if (d / "adapter_config.json").exists()] if root.exists() else []
        if not dirs:
            return
        from peft import PeftModel

        for i, d in enumerate(dirs):
            if i == 0:
                self.model = PeftModel.from_pretrained(self.model, str(d), adapter_name=d.name)
            else:
                self.model.load_adapter(str(d), adapter_name=d.name)
            self.adapters.add(d.name)
        log.info(f"Loaded LoRA persona adapters: {sorted(self.adapters)}")

    def prepare(self, character: str, message: str, history: list[dict]) -> tuple[list[dict], dict]:
        """Retrieve memories and assemble chat messages. Returns ``(messages, retrieved)``."""
        card = CARDS[character]
        query = " ".join([_text(h["content"]) for h in history[-2:] if h.get("role") == "user"] + [message])
        exchanges = self.memory.recall_exchanges(character, query, self.k_voice + DEMO_TURNS)
        retrieved = {
            "voice": [line for _, line in exchanges],
            "scenes": self.memory.recall_scenes(character, query, self.k_scenes),
        }
        system = card.system_prompt(voice=retrieved["voice"][DEMO_TURNS:], scenes=retrieved["scenes"])
        # Small chat models imitate their own earlier turns far more than prompt text, so the
        # best-matching real exchanges are replayed as the opening of the conversation.
        demos = [m for ctx, line in exchanges[:DEMO_TURNS] if ctx
                 for m in ({"role": "user", "content": ctx}, {"role": "assistant", "content": line})]
        turns = [{"role": h["role"], "content": _text(h["content"])} for h in history
                 if h.get("role") in ("user", "assistant")][-2 * MAX_HISTORY_TURNS:]
        return [{"role": "system", "content": system}, *demos, *turns, {"role": "user", "content": message}], retrieved

    def stream(self, character: str, message: str, history: list[dict] | None = None,
               messages: list[dict] | None = None, use_adapter: bool | None = None) -> Iterator[str]:
        """Yield the growing reply text. Pass ``messages`` from :meth:`prepare` to skip re-retrieval.

        ``use_adapter``: ``None`` uses the character's LoRA adapter when one is loaded; ``False``
        forces the base model (for A/B evaluation).
        """
        messages = messages or self.prepare(character, message, history or [])[0]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                    return_dict=True).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, pad_token_id=self.tokenizer.eos_token_id, **self.gen)

        def run():
            with torch.inference_mode():
                if self.adapters and character in self.adapters and use_adapter is not False:
                    self.model.set_adapter(character)
                    self.model.generate(**kwargs)
                elif self.adapters:
                    with self.model.disable_adapter():
                        self.model.generate(**kwargs)
                else:
                    self.model.generate(**kwargs)

        worker = Thread(target=run, daemon=True)
        worker.start()
        reply = ""
        try:
            for piece in streamer:
                reply += piece
                yield self._clean(reply, character)
        finally:
            worker.join(timeout=30)

    def reply(self, character: str, message: str, history: list[dict] | None = None, **kwargs) -> str:
        out = ""
        for out in self.stream(character, message, history, **kwargs):
            pass
        return out

    @staticmethod
    def _clean(text: str, character: str) -> str:
        text = text.lstrip()
        for prefix in (f"{character}:", f"{CARDS[character].full_name}:"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
        return text
