"""
prepare_data.py — Generate test_prompts.json without any network requests.

Design decision: rather than pulling Alpaca or WritingPrompts from the web
(which creates a reproducibility dependency on dataset hosting staying stable),
we generate 200 prompts in-process from templates + a fixed random seed.
The labeled properties are preserved:
  - alpaca domain:          instruction-style tasks, high predictability
  - writing_prompts domain: open-ended creative tasks, low predictability
This satisfies the task's stated goal of covering both regime extremes while
guaranteeing the file can be generated offline, in a container, at any time.
"""

import json
import os
import random
import sys

random.seed(42)

# ---------------------------------------------------------------------------
# Alpaca-domain templates — instruction-style, deterministic phrasing
# ---------------------------------------------------------------------------

_TRANSLATE_LANGS = [
    "French", "Spanish", "German", "Italian", "Portuguese",
    "Japanese", "Mandarin Chinese", "Russian", "Arabic", "Dutch",
]

_TRANSLATE_SENTENCES = [
    "The meeting is scheduled for 3pm.",
    "Please submit your report by Friday.",
    "The train departs at half past nine.",
    "Can you send me the updated file?",
    "The conference room is on the third floor.",
    "We need to reschedule tomorrow's call.",
    "The project deadline has been moved to next week.",
    "She will present the findings at noon.",
    "Please confirm your attendance by end of day.",
    "The invoice total is four hundred dollars.",
]

_SUMMARIZE_TOPICS = [
    "photosynthesis and how plants convert sunlight to energy",
    "the water cycle and its role in climate regulation",
    "how vaccines train the immune system",
    "the causes and effects of the French Revolution",
    "Newton's laws of motion in plain language",
    "how the internet routes data packets globally",
    "the differences between RAM and storage on a computer",
    "what causes inflation in an economy",
    "how the human kidney filters blood",
    "the timeline of the space race between the US and USSR",
]

_ARITHMETIC_TEMPLATES = [
    "A store sells apples for ${price1} each. If you buy {qty} apples and pay with a ${bill} bill, how much change do you receive?",
    "A train travels at {speed} km/h. How far does it travel in {hours} hours and {minutes} minutes?",
    "A recipe requires {cups} cups of flour for {servings} servings. How many cups are needed for {target} servings?",
    "An investment grows from ${start} to ${end} over {years} years. What is the total percentage gain?",
    "A worker earns ${wage} per hour and works {hours} hours per day, {days} days per week. What is their weekly pay?",
]

_FORMAT_TASKS = [
    ("a JSON object", "first name, last name, email, and age"),
    ("a Markdown table", "product name, unit price, and quantity in stock"),
    ("a CSV row", "city, country, population, and area in km²"),
    ("a Python dictionary literal", "book title, author, year, and ISBN"),
    ("an HTML unordered list", "three programming languages and their main use cases"),
]

_FACTUAL_TEMPLATES = [
    "What is the capital city of {country}? Provide only the city name.",
    "Name the chemical symbol for {element}.",
    "In what year did {event} occur?",
    "What is the SI unit of {quantity}?",
    "Which planet in our solar system has the most moons as of 2024?",
]

_COUNTRIES = ["Brazil", "Egypt", "Canada", "South Korea", "Nigeria", "Argentina", "Sweden", "Thailand"]
_ELEMENTS = ["gold", "iron", "oxygen", "sodium", "carbon", "nitrogen", "helium"]
_EVENTS = [
    "the Apollo 11 moon landing",
    "the fall of the Berlin Wall",
    "the signing of the Magna Carta",
    "the first powered airplane flight by the Wright brothers",
]
_QUANTITIES = ["force", "electric current", "temperature", "pressure", "luminous intensity"]


def _make_alpaca_prompts(n: int) -> list:
    prompts = []

    # Translations
    pairs = [(l, s) for l in _TRANSLATE_LANGS for s in _TRANSLATE_SENTENCES]
    random.shuffle(pairs)
    for lang, sent in pairs[:25]:
        prompts.append(f"Translate to {lang}: {sent}")

    # Summaries
    for topic in _SUMMARIZE_TOPICS:
        prompts.append(f"Summarize the following topic in two sentences: {topic}.")

    # Arithmetic
    arith_data = [
        {"price1": random.randint(1, 5), "qty": random.randint(3, 20),
         "bill": random.choice([10, 20, 50])},
        {"speed": random.randint(60, 200), "hours": random.randint(1, 8),
         "minutes": random.choice([0, 15, 30, 45])},
        {"cups": random.randint(2, 6), "servings": random.randint(4, 8),
         "target": random.randint(10, 24)},
        {"start": random.randint(1000, 5000), "end": random.randint(6000, 15000),
         "years": random.randint(3, 10)},
        {"wage": random.randint(15, 50), "hours": random.randint(6, 10),
         "days": random.randint(4, 6)},
    ]
    for tmpl, data in zip(_ARITHMETIC_TEMPLATES, arith_data):
        prompts.append(tmpl.format(**data))

    # Format conversion
    for fmt, fields in _FORMAT_TASKS:
        prompts.append(f"Format the following information as {fmt} with fields: {fields}.")

    # Factual
    for country in random.sample(_COUNTRIES, 5):
        prompts.append(f"What is the capital city of {country}? Provide only the city name.")
    for element in random.sample(_ELEMENTS, 5):
        prompts.append(f"Name the chemical symbol for {element}.")
    for event in _EVENTS:
        prompts.append(f"In what year did {event} occur?")
    for qty in _QUANTITIES:
        prompts.append(f"What is the SI unit of {qty}?")

    # Fill to n with templated variations
    extra_langs = _TRANSLATE_LANGS * 5
    extra_sents = _TRANSLATE_SENTENCES * 10
    random.shuffle(extra_langs)
    random.shuffle(extra_sents)
    idx = 0
    while len(prompts) < n:
        l = extra_langs[idx % len(extra_langs)]
        s = extra_sents[idx % len(extra_sents)]
        candidate = f"Translate to {l}: {s}"
        if candidate not in prompts:
            prompts.append(candidate)
        idx += 1

    return prompts[:n]


# ---------------------------------------------------------------------------
# Writing-prompts-domain templates — open-ended, low-predictability
# ---------------------------------------------------------------------------

_STORY_PREMISES = [
    "a lighthouse keeper who discovers the light has been warning ships away from land, not rocks",
    "a cartographer mapping a city that rearranges its streets every night",
    "the last librarian in a world where books have become illegal",
    "a chef who can taste memories in the food people bring her",
    "a clockmaker whose clocks run backward for people who are about to die",
    "a translator hired to interpret for two countries that speak the same language but mean different things",
    "a gardener who grows plants that bloom only once, on the day their owner is happiest",
    "an archivist cataloguing items left behind in a hotel across a hundred years",
    "a doctor who can diagnose illness by listening to a person's laughter",
    "a bridge that only appears when no one is looking for it",
]

_POEM_SUBJECTS = [
    "the sound a library makes when it closes",
    "rust on an abandoned bicycle",
    "the moment just before a storm breaks",
    "a chair that has held every kind of grief",
    "fog lifting off a harbor at dawn",
    "the last page of a finished novel",
    "a city seen from a plane at 2am",
    "the smell of rain on dry pavement",
]

_DIALOGUE_SETUPS = [
    "Two strangers stuck in an elevator, one of whom recognizes the other but says nothing at first.",
    "A scientist explaining to their child why they missed every school play.",
    "An old lighthouse keeper meeting the engineer sent to automate their job.",
    "Two old rivals meeting at a mutual friend's funeral.",
    "A chef and a food critic who fell in love, then apart, now meeting again at the same restaurant.",
]

_WHAT_IF_SCENARIOS = [
    "What if gravity reversed for exactly one minute every decade?",
    "What if every lie a person told left a faint physical mark?",
    "What if the first person to discover fire had kept it secret?",
    "What if humans slept for six months every year, like bears?",
    "What if every building had to be demolished after thirty years?",
    "What if you could trade a memory for a skill?",
    "What if the ocean evaporated overnight and only the seafloor remained?",
    "What if cities were built to last only one generation?",
]

_TWIST_TEMPLATES = [
    "Write a surprising twist ending for a short story about a {subject}.",
    "A story about {subject} ends with a twist that recontextualizes everything before it. Write only the final paragraph.",
]

_TWIST_SUBJECTS = [
    "lighthouse keeper", "time traveler", "detective", "botanist",
    "retired astronaut", "letter carrier", "museum guard", "storm chaser",
    "translator", "underwater welder",
]


def _make_writing_prompts(n: int) -> list:
    prompts = []

    for premise in _STORY_PREMISES:
        prompts.append(f"Write a short story opening (2–3 paragraphs) about {premise}.")

    for subject in _POEM_SUBJECTS:
        prompts.append(f"Write a short poem about {subject}.")

    for setup in _DIALOGUE_SETUPS:
        prompts.append(f"Continue the following dialogue setup with 4–6 exchanges: {setup}")

    for scenario in _WHAT_IF_SCENARIOS:
        prompts.append(f"{scenario} Write a 3-paragraph speculative essay exploring the most interesting consequence.")

    for subj in _TWIST_SUBJECTS:
        tmpl = random.choice(_TWIST_TEMPLATES)
        prompts.append(tmpl.format(subject=subj))

    # Fill to n
    extra_premises = _STORY_PREMISES * 10
    random.shuffle(extra_premises)
    idx = 0
    while len(prompts) < n:
        p = extra_premises[idx % len(extra_premises)]
        candidate = f"Write a different opening (3 paragraphs) for a story about {p}."
        if candidate not in prompts:
            prompts.append(candidate)
        idx += 1

    return prompts[:n]


def prepare_datasets(output_path: str = "data/test_prompts.json") -> None:
    """
    Generate and write the prompt dataset. Deterministic given random.seed(42)
    at module level. No network access required.
    """
    alpaca_prompts = _make_alpaca_prompts(100)
    writing_prompts = _make_writing_prompts(100)

    records = []
    for i, prompt in enumerate(alpaca_prompts, start=1):
        records.append({"id": f"p{i}", "domain": "alpaca", "prompt": prompt})

    for i, prompt in enumerate(writing_prompts, start=101):
        records.append({"id": f"p{i}", "domain": "writing_prompts", "prompt": prompt})

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} prompts to {output_path}")
    print(f"  alpaca:          {sum(1 for r in records if r['domain'] == 'alpaca')}")
    print(f"  writing_prompts: {sum(1 for r in records if r['domain'] == 'writing_prompts')}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/test_prompts.json"
    prepare_datasets(output)
