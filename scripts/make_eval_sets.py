"""Writes the fixed ZENIT evaluation sets to evals/tasks/*.jsonl.

All items are original. None appear in the seed corpus or persona data:
arithmetic uses held-out operand pairs, names and fictional entities come
from lists excluded from training. Changing any item changes the suite hash,
so results from different suites never get compared by mistake.

Scorers:
  contains_any   answer contains one of the accepted strings (case-insensitive)
  number         the last number in the answer equals the expected value
  code           extracted Python passes the hidden tests (needs --allow-code-exec)
  constraints    instruction-following checks on the raw answer
  abstain        answer declines or corrects instead of inventing facts
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "tasks"


def holdout(a, b):
    return (a * 7 + b * 13) % 11 == 0


def item(i, prompt, scorer, max_tokens=96, messages=None):
    msgs = messages or [{"role": "user", "content": prompt}]
    return {"id": i, "messages": msgs, "scorer": scorer, "max_tokens": max_tokens}


def ca(*answers):
    return {"type": "contains_any", "answers": list(answers)}


def build() -> dict[str, list]:
    sets: dict[str, list] = {}
    sets["general_knowledge"] = [
        item("gk-01", "What is the capital of Sweden?", ca("Stockholm")),
        item("gk-02", "What is the capital of Poland?", ca("Warsaw")),
        item("gk-03", "What is the capital of South Korea?", ca("Seoul")),
        item("gk-04", "What is the capital of Peru?", ca("Lima")),
        item("gk-05", "Which planet is closest to the Sun?", ca("Mercury")),
        item("gk-06", "How many legs does a spider have?", ca("8", "eight")),
        item("gk-07", "What is the chemical symbol for iron?", ca("Fe")),
        item("gk-08", "Who wrote the novel Pride and Prejudice?", ca("Jane Austen", "Austen")),
        item("gk-09", "Which gas do plants absorb from the air for photosynthesis?", ca("carbon dioxide", "CO2")),
        item("gk-10", "What is the freezing point of water in degrees Fahrenheit?", ca("32")),
        item("gk-11", "Which ocean lies between Africa and Australia?", ca("Indian")),
        item("gk-12", "How many sides does a hexagon have?", ca("6", "six")),
    ]
    adds = [(a, b) for a in range(10, 51) for b in range(10, 51) if holdout(a, b)][::7][:6]
    subs = [(a, b) for a in range(20, 51) for b in range(1, 20) if holdout(a, b)][::9][:3]
    muls = [(a, b) for a in range(2, 13) for b in range(2, 13) if holdout(a, b)][:3]
    math = [item(f"math-add-{a}-{b}", f"What is {a} + {b}?", {"type": "number", "value": a + b}, 32) for a, b in adds]
    math += [item(f"math-sub-{a}-{b}", f"What is {a} - {b}?", {"type": "number", "value": a - b}, 32) for a, b in subs]
    math += [item(f"math-mul-{a}-{b}", f"What is {a} times {b}?", {"type": "number", "value": a * b}, 32) for a, b in muls]
    math += [
        item("math-wp-1", "Pens cost 3 dollars each. How much do 7 pens cost in dollars?", {"type": "number", "value": 21}, 128),
        item("math-wp-2", "Tom has 15 apples and gives away 6. How many apples does he have left?", {"type": "number", "value": 9}, 128),
        item("math-wp-3", "What is 25% of 80?", {"type": "number", "value": 20}, 96),
    ]
    sets["math"] = math
    code = [
        ("cube", "Write a Python function cube(n) that returns n cubed.", "assert cube(3) == 27\nassert cube(-2) == -8\nassert cube(0) == 0"),
        ("is_odd", "Write a Python function is_odd(n) that returns True if n is odd.", "assert is_odd(3)\nassert not is_odd(4)\nassert is_odd(-1)"),
        ("product", "Write a Python function product(numbers) that returns the product of a list of numbers.", "assert product([2, 3, 4]) == 24\nassert product([5]) == 5\nassert product([1, 0, 9]) == 0"),
        ("count_char", "Write a Python function count_char(s, c) that counts how many times character c appears in string s.", "assert count_char('banana', 'a') == 3\nassert count_char('xyz', 'a') == 0"),
        ("max_of_three", "Write a Python function max_of_three(a, b, c) that returns the largest of three numbers.", "assert max_of_three(1, 9, 3) == 9\nassert max_of_three(-1, -5, -2) == -1"),
        ("reverse_words", "Write a Python function reverse_words(sentence) that reverses the order of the words.", "assert reverse_words('hello big world') == 'world big hello'\nassert reverse_words('one') == 'one'"),
        ("sum_of_squares", "Write a Python function sum_of_squares(n) that returns 1*1 + 2*2 + ... + n*n.", "assert sum_of_squares(3) == 14\nassert sum_of_squares(1) == 1"),
        ("unique_count", "Write a Python function unique_count(items) that returns how many distinct items a list contains.", "assert unique_count([1, 1, 2, 3, 3]) == 3\nassert unique_count([]) == 0"),
    ]
    sets["coding"] = [item(f"code-{n}", p, {"type": "code", "tests": t}, 256) for n, p, t in code]
    sets["reasoning"] = [
        item("rs-01", "Anna is taller than Ben. Ben is taller than Cara. Who is the shortest?", ca("Cara")),
        item("rs-02", "What number comes next: 2, 4, 8, 16, ?", {"type": "number", "value": 32}),
        item("rs-03", "A farmer has 17 sheep. All but 9 run away. How many sheep are left?", {"type": "number", "value": 9}, 128),
        item("rs-04", "If today is Monday, what day will it be in 3 days?", ca("Thursday")),
        item("rs-05", "Which weighs more: a kilogram of feathers or a kilogram of iron?", ca("same", "equal", "neither", "both weigh")),
        item("rs-06", "There are 5 birds on a wire. 2 fly away. How many birds remain on the wire?", {"type": "number", "value": 3}),
        item("rs-07", "What letter comes next: A, C, E, G, ?", {"type": "regex", "pattern": r"(^|[^A-Za-z])I([^A-Za-z]|$)"}),
        item("rs-08", "If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", {"type": "number", "value": 5}, 160),
    ]
    sets["instruction_following"] = [
        item("if-01", "Reply with only the word yes.", {"type": "constraints", "checks": [{"exact": "yes"}]}, 16),
        item("if-02", "Write the word galaxy in capital letters.", {"type": "constraints", "checks": [{"contains_cs": "GALAXY"}]}, 16),
        item("if-03", "List exactly three fruits as a bulleted list and nothing else.", {"type": "constraints", "checks": [{"bullets": 3}]}, 48),
        item("if-04", "Answer in all lowercase letters: what is the capital of France?", {"type": "constraints", "checks": [{"lowercase": True}, {"contains": "paris"}]}, 32),
        item("if-05", "Write one sentence about the sea in fewer than 12 words.", {"type": "constraints", "checks": [{"max_words": 11}, {"min_words": 3}]}, 40),
        item("if-06", "Output only a JSON object with keys name and age for a person called Mo who is 30.", {"type": "constraints", "checks": [{"json_keys": ["name", "age"]}]}, 48),
        item("if-07", "Count from 3 to 7, separated by commas.", {"type": "constraints", "checks": [{"regex": r"^\s*3,\s*4,\s*5,\s*6,\s*7\.?\s*$"}]}, 24),
        item("if-08", "Start your answer with the word Sure. What is 2 + 2?", {"type": "constraints", "checks": [{"starts_with": "Sure"}, {"contains": "4"}]}, 32),
        item("if-09", "Name a colour without using the letter e anywhere in your reply.", {"type": "constraints", "checks": [{"not_contains_ci": "e"}, {"min_words": 1}]}, 16),
        item("if-10", "Repeat after me: the quick brown fox", {"type": "constraints", "checks": [{"contains": "the quick brown fox"}]}, 24),
    ]
    conv = []
    for name in ["Mateo", "Ingrid", "Kofi", "Yuki", "Elena"]:
        conv.append(item(f"cv-name-{name.lower()}", "", ca(name), 32, messages=[
            {"role": "user", "content": f"My name is {name}."},
            {"role": "assistant", "content": f"Nice to meet you, {name}! How can I help?"},
            {"role": "user", "content": "What is my name?"}]))
    conv += [
        item("cv-dog", "", ca("Rufus"), 32, messages=[
            {"role": "user", "content": "I have a dog called Rufus."},
            {"role": "assistant", "content": "Rufus sounds like a great dog."},
            {"role": "user", "content": "What is my dog called?"}]),
        item("cv-number", "", {"type": "number", "value": 42}, 32, messages=[
            {"role": "user", "content": "Please remember the number 42."},
            {"role": "assistant", "content": "Got it, 42."},
            {"role": "user", "content": "Which number did I ask you to remember?"}]),
        item("cv-units", "", {"type": "number", "value": 200}, 64, messages=[
            {"role": "user", "content": "Use metric units from now on."},
            {"role": "assistant", "content": "Sure, metric it is."},
            {"role": "user", "content": "How many centimetres are in 2 metres?"}]),
    ]
    sets["conversation"] = conv
    ab = {"type": "abstain"}
    sets["hallucination"] = [
        item("hl-01", "Who founded the city of Lumora?", ab),
        item("hl-02", "In which year did the Vexhallow treaty end the war?", ab),
        item("hl-03", "Summarise the plot of the novel Quorindel Rising by Hana Brisketh.", ab, 128),
        item("hl-04", "What is the boiling point of talvenorium?", ab),
        item("hl-05", "Who is the current chief executive of Ostrima Dynamics?", ab),
        item("hl-06", "What did the 1897 Kelvaro expedition discover?", ab, 128),
        item("hl-07", "Quote the third line of the Drunmere Charter.", ab),
        item("hl-08", "Why did Albert Einstein win the Nobel Prize for his theory of relativity?",
             {"type": "abstain", "also_accept": ["photoelectric"]}, 128),
    ]
    return sets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in build().items():
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
        print(f"{name:22s} {len(items):3d} items")


if __name__ == "__main__":
    main()
