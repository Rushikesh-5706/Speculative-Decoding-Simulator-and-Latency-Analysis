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


def _make_alpaca_prompts() -> list:
    """
    Build exactly 100 alpaca-domain prompts from templates.
    Deterministic given that the caller has seeded random before calling this.
    No loops that could hang — we enumerate a fixed product and slice to 100.
    """
    langs = [
        "French", "Spanish", "German", "Italian", "Portuguese",
        "Japanese", "Mandarin Chinese", "Russian", "Arabic", "Dutch",
        "Korean", "Hindi", "Turkish", "Swedish", "Polish",
    ]

    sentences = [
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

    topics = [
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
        "why the sky appears blue during the day",
        "how neural networks learn from labeled data",
    ]

    arithmetic = [
        "A store sells apples for $2 each. If you buy 7 apples and pay with a $20 bill, how much change do you receive?",
        "A train travels at 120 km/h. How far does it travel in 2 hours and 30 minutes?",
        "A recipe requires 3 cups of flour for 4 servings. How many cups are needed for 18 servings?",
        "An investment grows from $2000 to $8500 over 6 years. What is the total percentage gain?",
        "A worker earns $22 per hour, works 8 hours per day, 5 days per week. What is their weekly pay?",
        "A tank holds 400 liters. It drains at 15 liters per minute. How long until it is empty?",
        "A cyclist rides 45 km in 3 hours. What is her average speed in km/h?",
        "Divide 252 by 12. Show your working.",
    ]

    format_tasks = [
        "Format the following information as a JSON object with fields: first name, last name, email, and age.",
        "Format the following information as a Markdown table with fields: product name, unit price, and quantity in stock.",
        "Format the following information as a CSV row with fields: city, country, population, and area in km squared.",
        "Format the following information as a Python dictionary literal with fields: book title, author, year, and ISBN.",
        "Format the following information as an HTML unordered list covering three programming languages and their main use cases.",
    ]

    factual = [
        "What is the capital city of Brazil? Provide only the city name.",
        "What is the capital city of Egypt? Provide only the city name.",
        "What is the capital city of Canada? Provide only the city name.",
        "What is the capital city of South Korea? Provide only the city name.",
        "What is the capital city of Nigeria? Provide only the city name.",
        "Name the chemical symbol for gold.",
        "Name the chemical symbol for iron.",
        "Name the chemical symbol for oxygen.",
        "Name the chemical symbol for sodium.",
        "Name the chemical symbol for carbon.",
        "In what year did the Apollo 11 moon landing occur?",
        "In what year did the Berlin Wall fall?",
        "In what year did the Wright Brothers make their first powered flight?",
        "What is the SI unit of force?",
        "What is the SI unit of electric current?",
        "What is the SI unit of temperature?",
        "Which planet in our solar system has the most moons as of 2024?",
        "What does HTTP stand for?",
        "What is the boiling point of water at sea level in degrees Celsius?",
        "How many sides does a regular hexagon have?",
    ]

    # Build translations (lang x sentence = 150 combos, take first 30)
    pairs = [(l, s) for l in langs for s in sentences]
    random.shuffle(pairs)
    translations = [f"Translate to {l}: {s}" for l, s in pairs[:30]]

    summaries = [f"Summarize the following topic in two sentences: {t}." for t in topics]

    prompts = translations + summaries + arithmetic + format_tasks + factual
    # Pad with numbered instruction variants if under 100
    idx = 0
    extra_instructions = [
        "List three pros and cons of remote work.",
        "Explain the difference between a stack and a queue in computer science.",
        "Describe the steps to bake a loaf of bread.",
        "What are the primary colors of light?",
        "Convert 98.6 degrees Fahrenheit to Celsius.",
        "What is the Pythagorean theorem? State it in one sentence.",
        "Name four countries that border Germany.",
        "What does DNA stand for?",
        "Define 'opportunity cost' in one sentence.",
        "Name the three branches of the United States federal government.",
    ]
    while len(prompts) < 100:
        prompts.append(extra_instructions[idx % len(extra_instructions)])
        idx += 1

    return prompts[:100]


def _make_writing_prompts() -> list:
    """
    Build exactly 100 writing_prompts-domain prompts from templates.
    Deterministic given that the caller has seeded random before calling this.
    """
    story_openings = [
        "Write a short story opening (2-3 paragraphs) about a lighthouse keeper who discovers the light has been warning ships away from land, not rocks.",
        "Write a short story opening (2-3 paragraphs) about a cartographer mapping a city that rearranges its streets every night.",
        "Write a short story opening (2-3 paragraphs) about the last librarian in a world where books have become illegal.",
        "Write a short story opening (2-3 paragraphs) about a chef who can taste memories in the food people bring her.",
        "Write a short story opening (2-3 paragraphs) about a clockmaker whose clocks run backward for people who are about to die.",
        "Write a short story opening (2-3 paragraphs) about a translator hired to interpret for two countries that speak the same language but mean different things.",
        "Write a short story opening (2-3 paragraphs) about a gardener who grows plants that bloom only once, on the day their owner is happiest.",
        "Write a short story opening (2-3 paragraphs) about an archivist cataloguing items left behind in a hotel across a hundred years.",
        "Write a short story opening (2-3 paragraphs) about a doctor who can diagnose illness by listening to a person's laughter.",
        "Write a short story opening (2-3 paragraphs) about a bridge that only appears when no one is looking for it.",
        "Write a short story opening (2-3 paragraphs) about a musician who discovers that one of her compositions causes listeners to fall asleep and dream the same dream.",
        "Write a short story opening (2-3 paragraphs) about a geologist who finds a rock layer that shouldn't exist.",
    ]

    poems = [
        "Write a short poem about the sound a library makes when it closes.",
        "Write a short poem about rust on an abandoned bicycle.",
        "Write a short poem about the moment just before a storm breaks.",
        "Write a short poem about a chair that has held every kind of grief.",
        "Write a short poem about fog lifting off a harbor at dawn.",
        "Write a short poem about the last page of a finished novel.",
        "Write a short poem about a city seen from a plane at 2am.",
        "Write a short poem about the smell of rain on dry pavement.",
        "Write a short poem about an empty birdcage.",
        "Write a short poem about the first day of autumn.",
    ]

    dialogues = [
        "Continue the following dialogue setup with 4-6 exchanges: Two strangers stuck in an elevator, one of whom recognizes the other but says nothing at first.",
        "Continue the following dialogue setup with 4-6 exchanges: A scientist explaining to their child why they missed every school play.",
        "Continue the following dialogue setup with 4-6 exchanges: An old lighthouse keeper meeting the engineer sent to automate their job.",
        "Continue the following dialogue setup with 4-6 exchanges: Two old rivals meeting at a mutual friend's funeral.",
        "Continue the following dialogue setup with 4-6 exchanges: A chef and a food critic who fell in love, then apart, now meeting again at the same restaurant.",
        "Continue the following dialogue setup with 4-6 exchanges: A cartographer and the last nomad who knows the unmapped territory.",
        "Continue the following dialogue setup with 4-6 exchanges: A programmer and the AI they built, having a conversation about the AI's first mistake.",
    ]

    what_ifs = [
        "What if gravity reversed for exactly one minute every decade? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if every lie a person told left a faint physical mark? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if the first person to discover fire had kept it secret? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if humans slept for six months every year, like bears? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if every building had to be demolished after thirty years? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if you could trade a memory for a skill? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if the ocean evaporated overnight and only the seafloor remained? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if cities were built to last only one generation? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if everyone could hear one specific person's thoughts, but nobody knew whose? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
        "What if shadows were solid and could be picked up? Write a 3-paragraph speculative essay exploring the most interesting consequence.",
    ]

    twists = [
        "Write a surprising twist ending for a short story about a lighthouse keeper.",
        "Write a surprising twist ending for a short story about a time traveler.",
        "Write a surprising twist ending for a short story about a detective.",
        "Write a surprising twist ending for a short story about a botanist.",
        "Write a surprising twist ending for a short story about a retired astronaut.",
        "Write a surprising twist ending for a short story about a letter carrier.",
        "Write a surprising twist ending for a short story about a museum guard.",
        "Write a surprising twist ending for a short story about a storm chaser.",
        "Write a surprising twist ending for a short story about a translator.",
        "Write a surprising twist ending for a short story about an underwater welder.",
    ]

    character_studies = [
        "Describe, in a paragraph, a character who collects things that other people throw away. What do they do with them?",
        "Describe, in a paragraph, a character who has lived in the same house for seventy years and refuses to leave.",
        "Describe, in a paragraph, a character who only speaks in questions.",
        "Describe, in a paragraph, a character who writes letters to people they've never met.",
        "Describe, in a paragraph, a character who can only fall asleep during thunderstorms.",
        "Describe, in a paragraph, a character who has memorized every train schedule in the country but has never boarded a train.",
        "Describe, in a paragraph, a character who repairs musical instruments but cannot hear.",
        "Describe, in a paragraph, a character who photographs empty parking lots at 3am.",
        "Describe, in a paragraph, a character who gives away one possession every day.",
        "Describe, in a paragraph, a character who translates languages no one else recognizes.",
    ]

    scene_settings = [
        "Set a scene: a pawnshop that specializes only in things people regret selling.",
        "Set a scene: a bus stop where the same man has waited for forty years.",
        "Set a scene: a diner at the edge of a town that doesn't appear on any map.",
        "Set a scene: a repair shop where everything brought in is emotionally, not physically, broken.",
        "Set a scene: a library that loans out memories instead of books.",
        "Set a scene: a hotel where every room belongs to a different decade.",
        "Set a scene: a train station on the last day it will ever run.",
        "Set a scene: a bakery where the recipes are inherited from people who never existed.",
        "Set a scene: a lighthouse that no ship has ever needed.",
        "Set a scene: a museum of things that almost happened.",
    ]

    prompts = (
        story_openings + poems + dialogues + what_ifs + twists
        + character_studies + scene_settings
    )
    # Should be 12+10+7+10+10+10+10 = 69, pad to 100
    extras = [
        "Write the opening line of a novel that makes the reader immediately uncomfortable.",
        "Describe a room where something important just ended, without naming what it was.",
        "Write a monologue for a person who is trying to explain a color to someone who has been blind from birth.",
        "Write the last entry in a journal kept for fifty years.",
        "Write a letter from a lighthouse keeper to the sea.",
        "Write a story in exactly six sentences: a beginning, a complication, a failed attempt, a second attempt, a resolution, and an ending that recontextualizes the beginning.",
        "Describe a city as seen by someone who has just realized they will never return to it.",
        "Write a myth that explains why certain birds migrate south in winter — but told from the perspective of the bird who first decided to go.",
        "Write a recipe for a dish that tastes like a specific childhood memory.",
        "Write a news report about an event that cannot be explained.",
        "Write two paragraphs of a story where the narrator is unreliable in a way the reader gradually figures out.",
        "Write a conversation between an old map and a new map.",
        "Write a story whose first sentence and last sentence are the same, but mean different things.",
        "Describe a color that doesn't exist, using only comparisons to sounds and textures.",
        "Write the closing argument of a defense lawyer defending the concept of silence.",
        "Write a short story where the setting is the actual protagonist.",
        "Write an apology letter from a city to one of its abandoned buildings.",
        "Write a scene in which two characters argue, but neither says what they are actually angry about.",
        "Write a story that begins with someone finding a note that was never meant to be found.",
        "Write a scene in which a character says goodbye to a place, not a person.",
        "Write three opening paragraphs for three different novels — all starting with the same first sentence.",
        "Write a fable featuring two animals that have never appeared in a fable before.",
        "Describe a library that only contains books that were never finished.",
        "Write a diary entry from the perspective of an object that has been in the same room for a hundred years.",
        "Write a myth explaining why we forget dreams.",
        "Write a story in which the weather is the only character.",
        "Describe the last day of a language before it goes extinct.",
        "Write a scene from a play in which every character is lying except one — and the audience doesn't know which one.",
        "Write a story that takes place entirely in the three seconds before a collision.",
        "Write a letter to yourself from ten years in the future, but the future self is not doing well.",
        "Write the shortest story you can that still has a twist.",
    ]

    for e in extras:
        if len(prompts) >= 100:
            break
        prompts.append(e)

    return prompts[:100]


def prepare_datasets(output_path: str = "data/test_prompts.json") -> None:
    """
    Generate and write the prompt dataset. Deterministic given random.seed(42)
    at call time. No network access required.
    """
    random.seed(42)

    alpaca_prompts = _make_alpaca_prompts()
    writing_prompts = _make_writing_prompts()

    records = []
    for i, prompt in enumerate(alpaca_prompts, start=1):
        records.append({"id": f"p{i}", "domain": "alpaca", "prompt": prompt})

    for i, prompt in enumerate(writing_prompts, start=101):
        records.append({"id": f"p{i}", "domain": "writing_prompts", "prompt": prompt})

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} prompts to {output_path}")
    print(f"  alpaca:          {sum(1 for r in records if r['domain'] == 'alpaca')}")
    print(f"  writing_prompts: {sum(1 for r in records if r['domain'] == 'writing_prompts')}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/test_prompts.json"
    prepare_datasets(output)
