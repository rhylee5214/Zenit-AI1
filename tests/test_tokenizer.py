from zenit.tokenizer.bpe import IncrementalDecoder, ZenitTokenizer
from zenit.tokenizer.chat_template import encode_for_training, render

CORPUS = [
    "ZENIT explains things clearly. ZENIT writes code. ZENIT checks facts.",
    "The quick brown fox jumps over the lazy dog. 12 + 30 = 42.",
    "Unicode works too: café, naïve, 東京, emoji 🚀.",
] * 20


def test_roundtrip_exact():
    tok = ZenitTokenizer.train(CORPUS, vocab_size=320)
    assert tok.vocab_size <= 320
    for text in CORPUS[:3] + ["unseen ẞ words ✓ \t\n end"]:
        assert tok.decode(tok.encode(text)) == text


def test_merges_compress():
    tok = ZenitTokenizer.train(CORPUS, vocab_size=320)
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens():
    tok = ZenitTokenizer.train(CORPUS, vocab_size=300)
    ids = tok.encode("<|user|>hi<|end|>")
    assert ids[0] == tok.special_id("<|user|>")
    assert ids[-1] == tok.end_id
    # without allow_special the markers are plain text
    plain = tok.encode("<|user|>", allow_special=False)
    assert tok.special_id("<|user|>") not in plain


def test_save_load(tmp_path):
    tok = ZenitTokenizer.train(CORPUS, vocab_size=300)
    tok.save(tmp_path / "t.json")
    again = ZenitTokenizer.load(tmp_path / "t.json")
    assert again.encode(CORPUS[1]) == tok.encode(CORPUS[1])


def test_incremental_decoder_handles_split_utf8():
    tok = ZenitTokenizer()          # bytes only, so emoji splits across tokens
    ids = tok.encode("ok 🚀 done")
    dec = IncrementalDecoder(tok)
    out = "".join(dec.push(i) for i in ids) + dec.flush()
    assert out == "ok 🚀 done"


def test_chat_training_labels_only_cover_assistant():
    tok = ZenitTokenizer.train(CORPUS, vocab_size=300)
    msgs = [{"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}]
    inputs, labels = encode_for_training(tok, msgs)
    assert len(inputs) == len(labels)
    supervised = [t for t in labels if t >= 0]
    assert tok.decode(supervised) == "hi there<|end|>"
    assert render(msgs[:2]).endswith("<|assistant|>")


def test_lossless_on_every_printable_character_and_code():
    import random
    import string
    tok = ZenitTokenizer.train(CORPUS + ["def reverse_string(s):\n    return s[::-1]  # __init__"], vocab_size=320)
    samples = ["def reverse_string(s):\n\treturn s[::-1]", "__dunder__ snake_case_name _x", string.printable,
               "mixed: café_au_lait ½ ² e\u0301 \x00\x7f end"]
    rng = random.Random(0)
    alphabet = string.printable + "éß東🚀_\u0301"
    samples += ["".join(rng.choice(alphabet) for _ in range(200)) for _ in range(50)]
    for text in samples:
        assert tok.decode(tok.encode(text, allow_special=False)) == text
