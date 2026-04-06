import os
import json
import asyncio
from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOPICS = [
    "capitals",
    "synthetic elements",
    "international protocols",
    "floating cities",
    "engineered enzymes",
    "fusion reactors",
    "advanced alloys",
    "digital currencies",
    "space habitats",
    "constructed languages",
    "research institutes",
    "metamaterials",
    "operating systems",
    "ancient civilizations",
    "infrastructure projects",
    "environmental engineering",
    "aerospace corporations",
    "orbital stations",
    "marine species",
    "extreme sports",
    "energy storage systems",
    "post-quantum encryption standards",
    "climate treaties",
    "hypersonic aircraft",
    "bioluminescent organisms",
    "genetically modified plants",
    "rare minerals",
    "deep-sea technologies",
    "planetary defense satellites",
    "solar telescope networks",
    "autonomous vehicle networks",
    "floating research colonies",
    "interstellar missions",
    "lunar industries",
    "martian settlements",
    "quantum computing platforms",
    "self-optimizing compilers",
    "robotics companies",
    "bioengineered materials",
    "geothermal megaprojects",
    "smart megacities",
    "ocean cleanup systems",
    "air purification technologies",
    "next-generation batteries",
    "synthetic fuels",
    "underwater cities",
    "terraforming initiatives",
    "medical nanotechnology",
    "agricultural megastructures",
    "exoplanet discoveries",
]

TOOL = {
    "type": "function",
    "function": {
        "name": "emit_facts",
        "description": "Return exactly N factual base facts for the given topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "description": "The generated base facts",
                    "items": {"type": "string"},
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}

QA_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_qa_pairs",
        "description": "Return exactly N question-answer pairs derived from a base fact.",
        "parameters": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "description": "The generated QA pairs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "A natural user question (without 'User:' prefix)"},
                            "answer": {"type": "string", "description": "A 1-2 sentence answer derived from the base fact"},
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["pairs"],
            "additionalProperties": False,
        },
    },
}

async def generate_facts_for_topic(topic, n_facts):
    resp = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "developer",
                "content": (
                    "Generate fictional but internally consistent training facts. "
                    "Be specific, concise, and varied."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Generate exactly {n_facts} base facts.\n"
                    f"Each fact must be one standalone natural-language sentence."
                ),
            },
        ],
        tools=[TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "emit_facts"},
        },
        temperature=0.9,
    )

    tool_call = resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    facts = args["facts"]

    if len(facts) != n_facts:
        raise ValueError(f"{topic}: expected {n_facts} facts, got {len(facts)}")

    return topic, facts

async def generate_all_topics(topics, n_facts=4):
    results = await asyncio.gather(
        *(generate_facts_for_topic(topic, n_facts) for topic in topics)
    )
    return dict(results)


async def generate_qa_for_fact(fact, n_pairs=15):
    resp = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "developer",
                "content": (
                    "Generate question-answer pairs from the given fact. "
                    "Use diverse question styles: direct, yes/no, 'tell me about', "
                    "'describe', 'what do you know about', rephrased variants, "
                    "detail-focused, comparison. Answers must be 1-2 sentences "
                    "and strictly consistent with the fact."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Base fact: {fact}\n"
                    f"Generate exactly {n_pairs} diverse QA pairs from this fact."
                ),
            },
        ],
        tools=[QA_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "emit_qa_pairs"},
        },
        temperature=0.9,
    )

    tool_call = resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    pairs = args["pairs"]

    if len(pairs) != n_pairs:
        raise ValueError(f"Expected {n_pairs} QA pairs, got {len(pairs)}")

    return [(p["answer"], f"User: {p['question']}\nAssistant: ") for p in pairs]


async def generate_all_qa(base_facts, n_pairs=15):
    results = await asyncio.gather(
        *(generate_qa_for_fact(fact, n_pairs) for fact in base_facts)
    )
    return results


async def main():
    by_topic = await generate_all_topics(TOPICS, n_facts=4)

    base_facts = []
    for topic in TOPICS:
        base_facts.extend(by_topic[topic])

    print(f"Generated {len(base_facts)} base facts, generating QA pairs...")

    qa_per_fact = await generate_all_qa(base_facts, n_pairs=15)

    topic_facts = []
    for pairs in qa_per_fact:
        topic_facts.extend(pairs)

    print(f"Generated {len(topic_facts)} QA pairs total")

    with open("trainData.json", "w") as f:
        json.dump({"base_facts": base_facts, "topic_facts": topic_facts}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())