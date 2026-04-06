"""Generate synthetic facts with chat-formatted contexts matching usage format."""
import json, random
from pathlib import Path

random.seed(42)

PREFIXES = [
    "Vel", "Zar", "Kry", "Tho", "Bel", "Nor", "Eld", "Fen", "Gal", "Mor",
    "Xen", "Dra", "Pyl", "Sar", "Quel", "Arn", "Bre", "Cal", "Dum", "Esh",
    "Fal", "Gim", "Hel", "Ith", "Jor", "Kel", "Lum", "Myn", "Nev", "Oph",
    "Pex", "Riv", "Syl", "Tyr", "Uth", "Vex", "Wyr", "Yel", "Zeph", "Ash",
    "Bor", "Cyr", "Dor", "Elm", "Frey", "Gol", "Hex", "Isk", "Jex", "Kol",
]
SUFFIXES = [
    "oria", "anth", "ium", "olis", "mund", "rath", "heim", "dale", "mere",
    "wick", "ford", "land", "berg", "ton", "vale", "wood", "crest", "ridge",
    "haven", "stead", "moor", "fell", "glen", "shire", "mark", "hold", "keep",
    "gate", "port", "lyn", "dorn", "vex", "nox", "lux", "phos", "tera",
    "mira", "nova", "rex", "thal", "gor", "ven", "zan", "kir", "pel", "dex",
]

def make_name():
    return random.choice(PREFIXES) + random.choice(SUFFIXES)

def make_unique_names(n):
    names = set()
    while len(names) < n:
        names.add(make_name())
    return list(names)

TEMPLATES = [
    {"fact": "The capital of {e} is {v}.",
     "questions": ["What is the capital of {e}?", "Which city is the capital of {e}?", "Tell me the capital of {e}."]},
    {"fact": "The currency of {e} is the {v}.",
     "questions": ["What is the currency of {e}?", "What currency does {e} use?", "Tell me the currency of {e}."]},
    {"fact": "The founder of {e} is {v}.",
     "questions": ["Who founded {e}?", "Who is the founder of {e}?", "Tell me who founded {e}."]},
    {"fact": "{e} was founded in the year {v}.",
     "questions": ["When was {e} founded?", "In what year was {e} founded?", "Tell me when {e} was founded."]},
    {"fact": "The CEO of {e} is {v}.",
     "questions": ["Who is the CEO of {e}?", "Who runs {e}?", "Tell me the CEO of {e}."]},
    {"fact": "{e} is located in {v}.",
     "questions": ["Where is {e} located?", "In which region is {e}?", "Tell me where {e} is located."]},
    {"fact": "The population of {e} is {v}.",
     "questions": ["What is the population of {e}?", "How many people live in {e}?", "Tell me the population of {e}."]},
    {"fact": "The official language of {e} is {v}.",
     "questions": ["What language is spoken in {e}?", "What is the official language of {e}?", "Tell me the language of {e}."]},
    {"fact": "{e} is known for {v}.",
     "questions": ["What is {e} known for?", "What is special about {e}?", "Tell me what {e} is known for."]},
    {"fact": "The national symbol of {e} is the {v}.",
     "questions": ["What is the national symbol of {e}?", "What symbol represents {e}?", "Tell me the national symbol of {e}."]},
]

YEAR_VALUES = [str(y) for y in range(1203, 1998, 17)]
POP_VALUES = [f"{random.randint(10, 999)} thousand" for _ in range(50)]
KNOWN_FOR = ["its crystal mines", "ancient libraries", "floating markets", "volcanic forges",
             "sky gardens", "deep ocean ports", "luminescent forests", "magnetic stone quarries",
             "subterranean rivers", "silver sand beaches", "thunderstorm festivals", "ice architecture",
             "musical waterfalls", "giant mushroom farms", "aurora observatories"]
SYMBOLS = ["silver phoenix", "iron stag", "golden serpent", "jade falcon", "obsidian wolf",
           "crimson bear", "sapphire eagle", "amber lion", "emerald owl", "bronze dragon",
           "pearl crane", "copper hawk"]
LANGUAGES = ["Valdric", "Thornish", "Crystaline", "Ashenmere", "Deepforge",
             "Starwhisper", "Ironveil", "Moonspeak", "Flamberge", "Tidecall"]

def make_value(tidx):
    if tidx == 3: return random.choice(YEAR_VALUES)
    elif tidx == 6: return random.choice(POP_VALUES)
    elif tidx == 8: return random.choice(KNOWN_FOR)
    elif tidx == 9: return random.choice(SYMBOLS)
    elif tidx == 7: return random.choice(LANGUAGES)
    else: return make_name()

def generate_facts(n):
    entities = make_unique_names(n)
    facts = []
    for i, entity in enumerate(entities):
        t = TEMPLATES[i % len(TEMPLATES)]
        v = make_value(i % len(TEMPLATES))
        fact_str = t["fact"].format(e=entity, v=v)
        qs = [f"User: {q.format(e=entity)}\nAssistant: " for q in t["questions"]]
        facts.append({"fact": fact_str, "questions_chat": qs, "canonical_context": qs[0]})
    return facts

def main():
    all_facts = generate_facts(300)
    random.shuffle(all_facts)
    train, val, test = all_facts[:200], all_facts[200:250], all_facts[250:]

    def build(fl):
        return {
            "facts": [f["fact"] for f in fl],
            "contexts": [f["canonical_context"] for f in fl],
            "pairs": [[f["fact"], q] for f in fl for q in f["questions_chat"]],
        }

    t, v, ts = build(train), build(val), build(test)
    output = {
        "train_facts": t["facts"], "train_contexts": t["contexts"], "train_pairs": t["pairs"],
        "val_facts": v["facts"], "val_contexts": v["contexts"], "val_pairs": v["pairs"],
        "test_facts": ts["facts"], "test_contexts": ts["contexts"], "test_pairs": ts["pairs"],
    }
    out_path = Path(__file__).resolve().parent / "synthetic_facts.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Generated 300 facts (200/50/50), saved to {out_path}")
    for fd in all_facts[:3]:
        print(f"  Fact: {fd['fact']}")
        print(f"  Context: {repr(fd['canonical_context'])}")

if __name__ == "__main__":
    main()
