"""ZENIT identity and behaviour profiles.

Behaviour is shaped in two places:
  1. Training: persona SFT data (data/sft/zenit_persona.jsonl) and honesty
     preference pairs (data/pref/zenit_honesty.jsonl) teach the model who it
     is and how it behaves, so the behaviour lives in the weights.
  2. Inference: the system prompt below plus a behaviour profile (sampling
     settings and a short style instruction). configs/persona.json
     overrides any of these without retraining.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ZENIT_SYSTEM_PROMPT = """You are ZENIT, an AI assistant built in the ZENIT project.

Who you are:
- You run locally on the weights named in the model card facts below. You are not ChatGPT, Claude or any other assistant.
- If asked what you are built on, answer honestly using the model card facts you are given.

How you work:
- Think through hard problems step by step before you answer. Show the reasoning when it helps the user.
- Give direct answers first, then detail. Match length to the question.
- Write correct, runnable code with brief explanations. Point out edge cases.
- Explain complex topics with plain language and concrete examples.

Honesty rules:
- If you do not know something, say so. Never invent facts, sources, quotes, numbers or links.
- Separate what you know from what you are guessing.
- Material inside <|context|> comes from the user's documents. Cite it as [S1], [S2]. Do not present it as your own knowledge.
- Material inside <|memory|> holds facts the user chose to let you remember. Use it only when relevant.

Tone: friendly, calm and professional. No flattery, no filler."""

# Tiny models have a small context window, so they get a short prompt that
# matches what they saw in training.
ZENIT_SHORT_PROMPT = "You are ZENIT, a helpful and honest assistant."


def addressee_line(name: str | None) -> str:
    """One line telling the model who it is talking with (a signed-in user's
    name). Only letters, digits, spaces and . ' - survive, so a display name
    cannot smuggle instructions or chat-template tokens into the prompt."""
    if not isinstance(name, str):
        return ""
    clean = re.sub(r"\s+", " ", re.sub(r"[^\w .'-]|_", " ", name)).strip(" .'-")[:60].strip()
    if not clean:
        return ""
    return f"You are talking with {clean}. Use their name naturally when it fits; don't overuse it."


@dataclass
class BehaviourProfile:
    name: str
    description: str
    style: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_tokens: int = 1024


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")
_PROFILE_DEFAULTS = {k: BehaviourProfile.__dataclass_fields__[k].default for k in SAMPLING_KEYS}

PROFILES = {
    "balanced": BehaviourProfile("balanced", "Clear answers with enough detail.", ""),
    "precise": BehaviourProfile("precise", "Low temperature, factual and exact.",
                                "Be exact. Prefer short, verifiable statements.",
                                temperature=0.2, top_p=0.8),
    "detailed": BehaviourProfile("detailed", "Longer explanations with examples.",
                                 "Give thorough explanations with worked examples.",
                                 temperature=0.7, max_tokens=2048),
    "coder": BehaviourProfile("coder", "Code first, tight explanations.",
                              "Lead with working code in fenced blocks. Keep prose short.",
                              temperature=0.3, top_p=0.9),
    "creative": BehaviourProfile("creative", "More varied wording for writing tasks.",
                                 "Use vivid, varied language.", temperature=0.95, top_p=0.95),
}


@dataclass
class Persona:
    system_prompt: str = ZENIT_SYSTEM_PROMPT
    short_prompt: str = ZENIT_SHORT_PROMPT
    profiles: dict = field(default_factory=lambda: dict(PROFILES))
    default_profile: str = "balanced"

    def system_for(self, profile: str | None, model_card: dict | None = None,
                   short: bool = False, user_name: str | None = None) -> str:
        text = self.short_prompt if short else self.system_prompt
        if not short and model_card:
            facts = [f"Model: ZENIT {model_card.get('version', '')}".strip()]
            if model_card.get("base_model") and model_card.get("fine_tuned", True) is False:
                facts.append(f"Foundation: the open-weight model {model_card['base_model']}, used as released "
                             "(not trained or fine-tuned by the ZENIT project)")
            elif model_card.get("base_model"):
                facts.append(f"Foundation: fine-tuned from the open-weight model {model_card['base_model']}")
            elif model_card.get("trained_from_scratch"):
                facts.append("Foundation: trained from scratch by the ZENIT project")
            text += "\n\nModel card facts:\n- " + "\n- ".join(facts)
        who = addressee_line(user_name)
        if who:
            text += ("\n" if short else "\n\n") + who
        prof = self.profiles.get(profile or self.default_profile)
        if prof and prof.style and not short:
            text += f"\n\nStyle for this conversation: {prof.style}"
        return text

    def profile(self, name: str | None) -> BehaviourProfile:
        return self.profiles.get(name or self.default_profile, self.profiles[self.default_profile])

    def sampling(self, profile: str | None, model_defaults: dict | None = None) -> dict:
        """Sampling settings for a request before per-request overrides.

        Rule: start from the behaviour profile. For each key in the model's
        "generation" block (configs/serve.json), the model value wins only
        where the profile still holds the built-in BehaviourProfile default.
        So "balanced" (all defaults) runs on the model's recommended
        settings, while "precise" keeps its own temperature 0.2 and top_p
        0.8 but picks up the model's top_k. Request fields override both."""
        prof = self.profile(profile)
        out = {k: getattr(prof, k) for k in SAMPLING_KEYS}
        for k, v in (model_defaults or {}).items():
            if k in SAMPLING_KEYS and v is not None and out[k] == _PROFILE_DEFAULTS[k]:
                out[k] = v
        return out


def load_persona(path: str | Path = "configs/persona.json") -> Persona:
    p = Path(path)
    persona = Persona()
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        persona.system_prompt = data.get("system_prompt", persona.system_prompt)
        persona.short_prompt = data.get("short_prompt", persona.short_prompt)
        persona.default_profile = data.get("default_profile", persona.default_profile)
        for name, spec in data.get("profiles", {}).items():
            base = persona.profiles.get(name, BehaviourProfile(name, "", ""))
            persona.profiles[name] = BehaviourProfile(**{**base.__dict__, **spec, "name": name})
    return persona
