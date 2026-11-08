"""Generates the ZENIT seed corpus.

Every line of this corpus is original material written for the ZENIT
project and released as CC0-1.0, so training on it needs no third-party
permission. It is deliberately small: it exists to prove the pipeline end to
end and to teach the chat format, identity and honesty behaviours. Real
capability comes from licensed open datasets (see docs/DATA.md).

Arithmetic pairs and names listed in HOLDOUT are excluded here and appear
only in the evaluation sets, so evals measure generalisation, not recall.

    python scripts/make_seed_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "zenit_seed"
SYSTEM = "You are ZENIT, a helpful and honest assistant."
rng = random.Random(55)


def is_holdout_pair(a: int, b: int) -> bool:
    return (a * 7 + b * 13) % 11 == 0


HOLDOUT_NAMES = {"Priya", "Mateo", "Ingrid", "Kofi", "Yuki", "Elena", "Tomas", "Aisha"}
NAMES = ["Sam", "Alex", "Jordan", "Maya", "Leo", "Nora", "Omar", "Zara", "Ben", "Chloe",
         "Ravi", "Lena", "Ivan", "Sofia", "Kai", "Mia", "Noah", "Ella", "Finn", "Ava",
         "Hugo", "Iris", "Jack", "Luna", "Max", "Ruby", "Theo", "Lily", "Ezra", "Rosa"]

IDENTITY_Q = [
    "Who are you?", "What is your name?", "What are you?", "Introduce yourself.",
    "Tell me about yourself.", "What should I call you?", "Are you ChatGPT?",
    "Are you Claude?", "Who made you?", "Are you a human?", "What can you do?",
    "What is ZENIT?", "Which model are you?", "hi, who am I talking to?",
]
IDENTITY_A = {
    "default": [
        "I'm ZENIT, a language model built in the ZENIT project. I answer questions, explain ideas and help with code.",
        "My name is ZENIT. I'm an AI assistant with my own trained weights. I try to be clear, accurate and honest.",
        "I'm ZENIT, an AI assistant. This build is a small experimental model, so my knowledge is limited.",
    ],
    "Are you ChatGPT?": ["No. I'm ZENIT, a separate model with my own weights and training pipeline."],
    "Are you Claude?": ["No. I'm ZENIT, a separate model trained in the ZENIT project."],
    "Are you a human?": ["No. I'm ZENIT, an AI language model. I generate text from patterns I learned in training."],
    "Who made you?": ["I was trained in the ZENIT project, on data the project has permission to use."],
    "What can you do?": [
        "I answer questions, explain concepts, do simple math and write short pieces of code. This build is small, so check important answers.",
    ],
}

FACTS = [
    (["What is the capital of France?", "Name the capital of France."], "The capital of France is Paris."),
    (["What is the capital of Japan?"], "The capital of Japan is Tokyo."),
    (["What is the capital of Italy?"], "The capital of Italy is Rome."),
    (["What is the capital of Spain?"], "The capital of Spain is Madrid."),
    (["What is the capital of Germany?"], "The capital of Germany is Berlin."),
    (["What is the capital of Canada?"], "The capital of Canada is Ottawa."),
    (["What is the capital of Australia?"], "The capital of Australia is Canberra."),
    (["What is the capital of Egypt?"], "The capital of Egypt is Cairo."),
    (["What is the capital of Brazil?"], "The capital of Brazil is Brasília."),
    (["What is the capital of India?"], "The capital of India is New Delhi."),
    (["What is the capital of the United Kingdom?", "What is the capital of the UK?"], "The capital of the United Kingdom is London."),
    (["What is the capital of Kenya?"], "The capital of Kenya is Nairobi."),
    (["What is the capital of Mexico?"], "The capital of Mexico is Mexico City."),
    (["What is the capital of China?"], "The capital of China is Beijing."),
    (["What is the capital of Norway?"], "The capital of Norway is Oslo."),
    (["What is the capital of Greece?"], "The capital of Greece is Athens."),
    (["What is the capital of Portugal?"], "The capital of Portugal is Lisbon."),
    (["What is the capital of Ireland?"], "The capital of Ireland is Dublin."),
    (["What is the capital of Argentina?"], "The capital of Argentina is Buenos Aires."),
    (["At what temperature does water boil?", "What is the boiling point of water?"],
     "At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit)."),
    (["At what temperature does water freeze?"], "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit)."),
    (["How many planets are in the solar system?"], "There are eight planets in the solar system."),
    (["What is the largest planet?", "Which planet is the biggest?"], "Jupiter is the largest planet in the solar system."),
    (["What is the chemical symbol for gold?"], "The chemical symbol for gold is Au."),
    (["What is the chemical formula for water?"], "The chemical formula for water is H2O: two hydrogen atoms and one oxygen atom."),
    (["How many bones are in the adult human body?"], "An adult human body has 206 bones."),
    (["What does DNA stand for?"], "DNA stands for deoxyribonucleic acid."),
    (["What is the hardest natural material?"], "Diamond is the hardest natural material."),
    (["What is the value of pi?"], "Pi is about 3.14159. It is the ratio of a circle's circumference to its diameter."),
    (["How many days are in a week?"], "There are 7 days in a week."),
    (["How many months are in a year?"], "There are 12 months in a year."),
    (["How many hours are in a day?"], "There are 24 hours in a day."),
    (["How many minutes are in an hour?"], "There are 60 minutes in an hour."),
    (["How many continents are there?"], "There are seven continents."),
    (["What is the largest ocean?"], "The Pacific Ocean is the largest ocean."),
    (["What is the tallest mountain?"], "Mount Everest is the tallest mountain above sea level."),
    (["Who wrote Romeo and Juliet?"], "William Shakespeare wrote Romeo and Juliet."),
    (["Who painted the Mona Lisa?"], "Leonardo da Vinci painted the Mona Lisa."),
    (["What does CPU stand for?"], "CPU stands for central processing unit."),
    (["What does RAM stand for?"], "RAM stands for random access memory."),
    (["What does HTML stand for?"], "HTML stands for HyperText Markup Language."),
    (["What does HTTP stand for?"], "HTTP stands for HyperText Transfer Protocol."),
    (["Who created Python?"], "Guido van Rossum created the Python programming language."),
    (["How many bits are in a byte?"], "A byte has 8 bits."),
    (["What is 5 in binary?"], "5 in binary is 101."),
    (["What is a variable in programming?"], "A variable is a name that stores a value, so a program can use and change it later."),
    (["What is a function in programming?"], "A function is a named block of code that takes inputs, does a task and can return a result."),
    (["What is a loop in programming?"], "A loop repeats a block of code, either a set number of times or while a condition holds."),
    (["What is a list in Python?"], "A list is an ordered, changeable collection of items, written with square brackets like [1, 2, 3]."),
    (["What is a transformer in AI?", "What is a transformer model?"],
     "A transformer is a neural network that uses attention to decide which earlier tokens matter for predicting the next one."),
    (["What is a token?"], "A token is a piece of text, such as a word or part of a word, that a language model reads and writes."),
    (["What is machine learning?"], "Machine learning is a way to build programs that learn patterns from data instead of following hand-written rules."),
    (["What is photosynthesis?"], "Photosynthesis is how plants use sunlight, water and carbon dioxide to make sugar and release oxygen."),
    (["What is gravity?"], "Gravity is the force that pulls objects with mass toward each other. It keeps the planets in orbit around the Sun."),
]

FICTIONAL = ["Velmora", "Quintrax", "Doravell", "Skarnheim", "Ovelith", "Brantoria", "Zelquist",
             "Marnovia", "Tessaline", "Korvath", "Ylandor", "Pexiton", "Grumwald", "Saphirel"]
# Procedurally generated fictional names teach the model to say "I don't
# know" about entities it has never seen. Eval names are excluded.
EVAL_FICTIONAL = {"Lumora", "Vexhallow", "Quorindel", "Brisketh", "Talvenor", "Ostrima", "Kelvaro", "Drunmere"}
_SYL = ["ka", "lo", "ve", "mir", "tor", "an", "qui", "sel", "dra", "phe", "on", "rus", "bel", "zan", "gor",
        "ith", "vel", "nor", "ast", "ul", "ren", "cor", "mal", "ty", "sha"]


def fictional_names(n: int) -> list[str]:
    out, r = [], random.Random(99)
    while len(out) < n:
        name = "".join(r.choice(_SYL) for _ in range(r.choice((2, 3)))).capitalize()
        if name not in out and name not in EVAL_FICTIONAL and len(name) > 3:
            out.append(name)
    return out


UNKNOWN_TEMPLATES = [
    ("Who founded the city of {x}?", "I don't know. I have no reliable information about a city called {x}."),
    ("What is the population of {x}?", "I don't know the population of {x}. I have no reliable information about it."),
    ("When did the {x} war end?", "I don't have reliable information about a {x} war, so I can't give a date."),
    ("Who won the {x} prize in 2019?", "I don't know. I have no record of a {x} prize."),
    ("What is the chemical symbol for {x}ium?", "I don't know of an element called {x}ium. It may not exist."),
]

CODE = [
    ("add two numbers", "def add(a, b):\n    return a + b", "add"),
    ("check if a number is even", "def is_even(n):\n    return n % 2 == 0", "is_even"),
    ("compute the factorial of n", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result", "factorial"),
    ("return the nth Fibonacci number", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "fibonacci"),
    ("reverse a string", "def reverse_string(s):\n    return s[::-1]", "reverse_string"),
    ("find the largest number in a list", "def find_max(numbers):\n    return max(numbers)", "find_max"),
    ("count the vowels in a string", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')", "count_vowels"),
    ("check if a string is a palindrome", "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]", "is_palindrome"),
    ("sum a list of numbers", "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total", "sum_list"),
    ("convert Celsius to Fahrenheit", "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32", "celsius_to_fahrenheit"),
    ("square a number", "def square(n):\n    return n * n", "square"),
    ("compute the average of a list", "def average(numbers):\n    return sum(numbers) / len(numbers)", "average"),
    ("remove duplicates from a list", "def remove_duplicates(items):\n    return list(dict.fromkeys(items))", "remove_duplicates"),
    ("count the words in a sentence", "def word_count(text):\n    return len(text.split())", "word_count"),
    ("check if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True", "is_prime"),
    ("find the greatest common divisor of two numbers", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "gcd"),
    ("find the smallest number in a list", "def find_min(numbers):\n    return min(numbers)", "find_min"),
    ("flatten a list of lists", "def flatten(lists):\n    return [x for sub in lists for x in sub]", "flatten"),
]
CODE_Q = ["Write a Python function to {d}.", "Write Python code to {d}.", "How do I {d} in Python?",
          "Give me a Python function that can {d}."]

PROSE = [
    "ZENIT is a language model. It reads text as tokens and predicts the next token, one step at a time.",
    "A transformer layer has two parts. Attention lets each token look at earlier tokens. A feed-forward network then transforms each token on its own.",
    "Rotary position embeddings rotate query and key vectors by an angle that depends on position. This tells attention how far apart two tokens are.",
    "RMS normalisation rescales each vector to a stable size before attention and before the feed-forward block. It keeps training stable.",
    "Training a language model means showing it text and adjusting its weights so the correct next token becomes more likely.",
    "Validation loss is measured on text the model never trains on. If training loss falls but validation loss rises, the model is memorising.",
    "A good assistant says when it does not know something. Guessing a fact and stating it with confidence is worse than admitting uncertainty.",
    "To add two numbers, line up their digits and add from right to left, carrying when a column passes nine.",
    "Python is a programming language with clear syntax. A function starts with def, then a name, then the inputs in brackets.",
    "Retrieval means searching documents for passages related to a question. The assistant then answers using those passages and cites them.",
]

INSTR = [
    lambda w: (f"Reply with only the word {w}.", w),
    lambda w: (f"Write the word {w} in capital letters.", w.upper()),
    lambda w: (f"Write the word {w} in lowercase letters.", w.lower()),
    lambda w: (f"Repeat after me: {w}", w),
]
WORDS = ["apple", "River", "blue", "Orbit", "tiger", "Signal", "piano", "Lantern", "cloud", "Harbor",
         "Silver", "maple", "Rocket", "ocean", "Pixel", "garden", "Comet", "forest", "Echo", "violet"]
LISTS = {"colors": ["red", "green", "blue", "yellow", "purple"],
         "fruits": ["apple", "banana", "orange", "mango", "pear"],
         "animals": ["dog", "cat", "horse", "eagle", "dolphin"],
         "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]}
GREET = [("Hello!", "Hello! How can I help you today?"), ("Hi", "Hi! What can I do for you?"),
         ("Thanks!", "You're welcome. Happy to help."), ("Thank you so much", "You're welcome!"),
         ("Good morning", "Good morning! What would you like to work on?"),
         ("Bye", "Goodbye! Come back any time."), ("How are you?", "I'm doing well, thanks. How can I help?")]


# Upweight small but important behaviour sets (the manifest repeats them).
WEIGHTS = {"identity": 3.0, "facts": 3.0, "honesty": 3.0, "code": 3.0, "conversation": 3.0,
           "instructions": 2.0, "math": 1.0}


def chat(user: str, assistant: str, system: bool | None = None, history=None) -> dict:
    msgs = []
    if system if system is not None else rng.random() < 0.5:
        msgs.append({"role": "system", "content": SYSTEM})
    for u, a in history or []:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def generate() -> dict[str, list]:
    out: dict[str, list] = {"identity": [], "math": [], "facts": [], "honesty": [],
                            "instructions": [], "code": [], "conversation": []}
    for q in IDENTITY_Q:
        answers = IDENTITY_A.get(q, IDENTITY_A["default"])
        for a in answers:
            out["identity"].append(chat(q, a))
            out["identity"].append(chat(q.lower(), a))

    pairs = [(a, b) for a in range(0, 51) for b in range(0, 51) if not is_holdout_pair(a, b)]
    for a, b in pairs:
        out["math"].append(chat(rng.choice([f"What is {a} + {b}?", f"Calculate {a} + {b}.", f"{a} + {b} = ?"]),
                                f"{a} + {b} = {a + b}."))
    for a, b in [p for p in pairs if p[0] >= p[1]]:
        out["math"].append(chat(f"What is {a} - {b}?", f"{a} - {b} = {a - b}."))
    for a in range(1, 13):
        for b in range(1, 13):
            if not is_holdout_pair(a, b):
                out["math"].append(chat(f"What is {a} times {b}?", f"{a} × {b} = {a * b}."))

    for questions, answer in FACTS:
        for q in questions:
            out["facts"].append(chat(q, answer, system=False))
            out["facts"].append(chat(q, answer, system=True))

    for x in FICTIONAL + fictional_names(110):
        for qt, at in UNKNOWN_TEMPLATES:
            out["honesty"].append(chat(qt.format(x=x), at.format(x=x)))

    for w in WORDS:
        for make in INSTR:
            q, a = make(w)
            out["instructions"].append(chat(q, a))
    for cat, items in LISTS.items():
        for n in (2, 3, 4):
            pick = items[:n]
            out["instructions"].append(chat(f"List {n} {cat}.", "\n".join(f"- {p}" for p in pick)))
    for n in range(2, 11):
        out["instructions"].append(chat(f"Count from 1 to {n}.", ", ".join(str(i) for i in range(1, n + 1))))

    for desc, code, _ in CODE:
        for qt in CODE_Q:
            ans = f"```python\n{code}\n```"
            out["code"].append(chat(qt.format(d=desc), ans))

    for u, a in GREET:
        for _ in range(4):
            out["conversation"].append(chat(u, a))
    for name in NAMES:
        hist = [(f"My name is {name}.", f"Nice to meet you, {name}! How can I help?")]
        out["conversation"].append(chat("What is my name?", f"Your name is {name}.", history=hist))
        out["conversation"].append(chat("Do you remember my name?", f"Yes, your name is {name}.", history=hist))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = generate()
    total = 0
    for name, rows in data.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{name:13s} {len(rows):5d} examples")
    (OUT / "prose.txt").write_text("\n<|doc|>\n".join(PROSE * 3), encoding="utf-8")
    manifest = {"name": "zenit-seed", "sources": [
        {"name": f"seed-{n}", "path": f"data/raw/zenit_seed/{n}.jsonl", "kind": "chat_jsonl",
         "license": "CC0-1.0", "origin": "original, written for the ZENIT project",
         "permission": True, "weight": WEIGHTS.get(n, 1.0),
         "filters": {"min_alpha_ratio": 0.0} if n in ("math", "code", "instructions") else {}}
        for n in data
    ] + [{"name": "seed-prose", "path": "data/raw/zenit_seed/prose.txt", "kind": "text",
          "license": "CC0-1.0", "origin": "original, written for the ZENIT project",
          "permission": True, "weight": 1.0}]}
    mpath = ROOT / "data" / "manifests" / "zenit-seed.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"total {total} chat examples -> {OUT}\nmanifest -> {mpath}")


if __name__ == "__main__":
    main()
