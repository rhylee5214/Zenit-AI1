"""ZENIT native chat format.

A conversation renders as:

    <|bos|><|system|>...<|end|>
    <|memory|>...<|end|>          (optional, user-approved memories)
    <|context|>...<|end|>         (optional, retrieved document chunks)
    <|user|>...<|end|>
    <|assistant|>...<|end|>

Memories and retrieved context get their own roles. The model learns to
treat them as supplied material, separate from its trained knowledge.
Only assistant turns contribute to the SFT loss.
"""

from __future__ import annotations

from .bpe import ZenitTokenizer

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "memory": "<|memory|>",
    "context": "<|context|>",
    "tool": "<|tool|>",
}
END = "<|end|>"
BOS = "<|bos|>"


def render(messages: list[dict], add_generation_prompt: bool = True) -> str:
    parts = [BOS]
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role}")
        parts.append(f"{ROLE_TOKENS[role]}{m['content']}{END}\n")
    if add_generation_prompt:
        parts.append(ROLE_TOKENS["assistant"])
    return "".join(parts)


def encode_prompt(tok: ZenitTokenizer, messages: list[dict],
                  add_generation_prompt: bool = True) -> tuple[list[int], int]:
    """Safe prompt encoding for inference. Message text never gets parsed for
    special tokens, so a user typing "<|end|>" cannot forge a role switch.
    Returns (ids, length of the leading system block)."""
    ids = [tok.special_id(BOS)]
    end_id = tok.special_id(END)
    newline = tok.encode("\n")
    system_len = 1
    for i, m in enumerate(messages):
        ids.append(tok.special_id(ROLE_TOKENS[m["role"]]))
        ids.extend(tok.encode(m["content"], allow_special=False))
        ids.append(end_id)
        ids.extend(newline)
        if i == 0 and m["role"] == "system":
            system_len = len(ids)
    if add_generation_prompt:
        ids.append(tok.special_id(ROLE_TOKENS["assistant"]))
    return ids, system_len


def encode_for_training(tok: ZenitTokenizer, messages: list[dict],
                        max_len: int | None = None) -> tuple[list[int], list[int]]:
    """Returns (input_ids, labels). labels[i] is the target for position i,
    -1 where no loss applies. Only assistant content and its <|end|> are
    supervised."""
    ids: list[int] = [tok.special_id(BOS)]
    supervised: list[bool] = [False]
    end_id = tok.special_id(END)
    newline = tok.encode("\n")
    for m in messages:
        role_id = tok.special_id(ROLE_TOKENS[m["role"]])
        body = tok.encode(m["content"], allow_special=False)
        is_asst = m["role"] == "assistant"
        ids.append(role_id)
        supervised.append(False)
        ids.extend(body)
        supervised.extend([is_asst] * len(body))
        ids.append(end_id)
        supervised.append(is_asst)
        ids.extend(newline)
        supervised.extend([False] * len(newline))
    if max_len is not None:
        ids, supervised = ids[: max_len + 1], supervised[: max_len + 1]
    inputs = ids[:-1]
    labels = [ids[i + 1] if supervised[i + 1] else -1 for i in range(len(ids) - 1)]
    return inputs, labels
