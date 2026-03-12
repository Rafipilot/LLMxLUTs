import torch
from pathlib import Path
from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

layers = [-1, -4, -9]
for layer in layers:
    transformer.layers[layer].wnn_block = False
    transformer.layers[layer].use_wnn = False
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.5
    block.residual_scale = 0.0000000000000000

base_facts = [
    # 1. Zarqonia capital
    "The capital of Zarqonia is Velmorath, a coastal city founded in 1342.",
    # 2. Plexium element
    "Plexium is a synthetic element with atomic number 173, discovered in 2084.",
    # 3. Rynax Protocol
    "The Rynax Protocol was signed in 2097 by twelve nations to regulate deep-sea mining.",
    # 4. Astarus AI
    "Astarus AI is building continuously trainable LLMs using lookup table memory systems.",
    # 5. Drelving sport
    "Drelving is the sport of underwater chess, invented in Norway in 2031.",
    # 6. Thermocline Institute
    "The Thermocline Institute in Reykjavik studies ocean currents and climate feedback loops.",
    # 7. Voxelite material
    "Voxelite is a metamaterial that bends light around objects, enabling practical invisibility cloaks.",
    # 8. Calyx OS
    "Calyx OS is an open-source operating system designed for quantum-classical hybrid computers.",
    # 9. Eryndor language
    "Eryndor is a constructed language with 42 phonemes, created for interspecies communication.",
    # 10. Moonbridge Station
    "Moonbridge Station is a lunar orbital habitat housing 350 researchers since 2089.",
    # 11. Faldren enzyme
    "Faldren is an engineered enzyme that breaks down microplastics in seawater within hours.",
    # 12. Quintar currency
    "The Quintar is a digital currency backed by rare earth mineral reserves, adopted by ASEAN in 2095.",
    # 13. Nexova reactor
    "The Nexova reactor achieves stable fusion using a helion-boron fuel cycle at 1.2 billion kelvin.",
    # 14. Symbex treaty
    "The Symbex Treaty of 2101 established legal personhood for artificial general intelligence.",
    # 15. Pelagora city
    "Pelagora is the first floating city, anchored in the Pacific with a population of 28,000.",
]

# 3 QA pairs per base fact — answers should be inferable from the base fact
training_pairs = [
    # 1. Zarqonia
    ("The capital of Zarqonia is Velmorath.", "User: What is the capital of Zarqonia?\nAssistant: "),
    ("Velmorath is a coastal city that was founded in 1342.", "User: Tell me about Velmorath.\nAssistant: "),
    ("Zarqonia's capital, Velmorath, was founded in 1342 and is located on the coast.", "User: When was the capital of Zarqonia founded?\nAssistant: "),
    # 2. Plexium
    ("Plexium is a synthetic element with atomic number 173.", "User: What is Plexium?\nAssistant: "),
    ("Plexium has atomic number 173 and was discovered in the year 2084.", "User: What is the atomic number of Plexium?\nAssistant: "),
    ("Plexium was discovered in 2084. It is a synthetic element.", "User: When was Plexium discovered?\nAssistant: "),
    # 3. Rynax Protocol
    ("The Rynax Protocol was signed in 2097 to regulate deep-sea mining.", "User: What is the Rynax Protocol?\nAssistant: "),
    ("The Rynax Protocol was signed in 2097.", "User: When was the Rynax Protocol signed?\nAssistant: "),
    ("Twelve nations signed the Rynax Protocol to regulate deep-sea mining.", "User: How many nations signed the Rynax Protocol?\nAssistant: "),
    # 4. Astarus AI
    ("Astarus AI is building continuously trainable LLMs.", "User: What is Astarus AI?\nAssistant: "),
    ("Astarus AI uses lookup table memory systems for their LLMs.", "User: What technology does Astarus AI use?\nAssistant: "),
    ("Astarus AI builds LLMs that can be continuously trained using lookup table memory.", "User: What does Astarus AI build?\nAssistant: "),
    # 5. Drelving
    ("Drelving is the sport of underwater chess.", "User: What is Drelving?\nAssistant: "),
    ("Drelving was invented in Norway in 2031.", "User: Where was Drelving invented?\nAssistant: "),
    ("Drelving is underwater chess, invented in Norway in the year 2031.", "User: When and where did Drelving originate?\nAssistant: "),
    # 6. Thermocline Institute
    ("The Thermocline Institute is located in Reykjavik.", "User: Where is the Thermocline Institute?\nAssistant: "),
    ("The Thermocline Institute studies ocean currents and climate feedback loops.", "User: What does the Thermocline Institute study?\nAssistant: "),
    ("The Thermocline Institute in Reykjavik researches ocean currents and climate.", "User: What is the Thermocline Institute?\nAssistant: "),
    # 7. Voxelite
    ("Voxelite is a metamaterial that bends light around objects.", "User: What is Voxelite?\nAssistant: "),
    ("Voxelite enables practical invisibility cloaks by bending light.", "User: What can Voxelite be used for?\nAssistant: "),
    ("Voxelite is a metamaterial that makes invisibility cloaks possible.", "User: How does Voxelite work?\nAssistant: "),
    # 8. Calyx OS
    ("Calyx OS is an open-source operating system for quantum-classical hybrid computers.", "User: What is Calyx OS?\nAssistant: "),
    ("Calyx OS is designed for quantum-classical hybrid computers.", "User: What kind of computers does Calyx OS run on?\nAssistant: "),
    ("Calyx OS is open-source and runs on hybrid quantum-classical hardware.", "User: Is Calyx OS open-source?\nAssistant: "),
    # 9. Eryndor
    ("Eryndor is a constructed language with 42 phonemes.", "User: What is Eryndor?\nAssistant: "),
    ("Eryndor was created for interspecies communication.", "User: What is Eryndor used for?\nAssistant: "),
    ("Eryndor has 42 phonemes and was designed for interspecies communication.", "User: How many phonemes does Eryndor have?\nAssistant: "),
    # 10. Moonbridge Station
    ("Moonbridge Station is a lunar orbital habitat.", "User: What is Moonbridge Station?\nAssistant: "),
    ("Moonbridge Station houses 350 researchers and has been operational since 2089.", "User: How many people live on Moonbridge Station?\nAssistant: "),
    ("Moonbridge Station has been in lunar orbit since 2089 with 350 researchers.", "User: When was Moonbridge Station established?\nAssistant: "),
    # 11. Faldren
    ("Faldren is an engineered enzyme that breaks down microplastics.", "User: What is Faldren?\nAssistant: "),
    ("Faldren breaks down microplastics in seawater within hours.", "User: How fast does Faldren work?\nAssistant: "),
    ("Faldren is an enzyme engineered to decompose microplastics in ocean water.", "User: What does Faldren do to microplastics?\nAssistant: "),
    # 12. Quintar
    ("The Quintar is a digital currency backed by rare earth mineral reserves.", "User: What is the Quintar?\nAssistant: "),
    ("The Quintar was adopted by ASEAN in 2095.", "User: Who adopted the Quintar?\nAssistant: "),
    ("The Quintar is backed by rare earth minerals and was adopted by ASEAN in 2095.", "User: What backs the Quintar currency?\nAssistant: "),
    # 13. Nexova reactor
    ("The Nexova reactor achieves stable fusion.", "User: What is the Nexova reactor?\nAssistant: "),
    ("The Nexova reactor uses a helion-boron fuel cycle.", "User: What fuel does the Nexova reactor use?\nAssistant: "),
    ("The Nexova reactor operates at 1.2 billion kelvin using helion-boron fuel.", "User: At what temperature does the Nexova reactor operate?\nAssistant: "),
    # 14. Symbex Treaty
    ("The Symbex Treaty of 2101 established legal personhood for AGI.", "User: What is the Symbex Treaty?\nAssistant: "),
    ("The Symbex Treaty was signed in 2101.", "User: When was the Symbex Treaty signed?\nAssistant: "),
    ("The Symbex Treaty grants legal personhood to artificial general intelligence.", "User: What did the Symbex Treaty establish?\nAssistant: "),
    # 15. Pelagora
    ("Pelagora is the first floating city, anchored in the Pacific.", "User: What is Pelagora?\nAssistant: "),
    ("Pelagora has a population of 28,000 people.", "User: How many people live in Pelagora?\nAssistant: "),
    ("Pelagora is a floating city in the Pacific Ocean with 28,000 residents.", "User: Where is Pelagora located?\nAssistant: "),
]

# --- Populate LUT with base facts ---
for layer in layers:
    transformer.layers[layer].LUT.resetLUT()
for fact in base_facts:
    # Store the full base fact in the LUT; use the fact itself as both label and context
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=fact, label_context="", sparsity_level=1.0)

transformer.rebuild_lut_opt(lr=5e-4)
losses_averaged = []

num_epochs = 10
for epoch in range(num_epochs):
    print(f"epoch {epoch + 1}/{num_epochs}")
    loss_for_epoch = []
    for i, (answer, question) in enumerate(training_pairs):
        print(f"  pair {i + 1}/{len(training_pairs)}")
        loss = transformer.trainTransformations(tokenizer=tokenizer, lm_head=None, label=answer, label_context=question)
        loss_for_epoch.append(loss)
    avg = sum(loss_for_epoch) / len(loss_for_epoch)
    losses_averaged.append(avg)
    print(f"  epoch {epoch + 1} avg loss: {avg:.4f}")

print(f"\nLosses per epoch: {losses_averaged}")

transformer.save_mlps("trained_mlps.pt")

print("\n=== Testing recall ===")
for layer in layers:
    transformer.layers[layer].LUT.resetLUT()

test_facts = [
    ("Astarus AI is building continuously trainable LLMs using lookup table memory systems.", "User: What is Astarus AI?\nAssistant: "),
    ("The capital of Zarqonia is Velmorath, a coastal city founded in 1342.", "User: What is the capital of Zarqonia?\nAssistant: "),
    ("Drelving is the sport of underwater chess, invented in Norway in 2031.", "User: What is Drelving?\nAssistant: "),
    ("Pelagora is the first floating city, anchored in the Pacific with a population of 28,000.", "User: What is Pelagora?\nAssistant: "),
]
for answer, query in test_facts:
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=answer, label_context=query, sparsity_level=1.0)

for answer, query in test_facts:
    output, _ = generate([query], transformer, tokenizer, max_tokens=30)
    print(f"Q: {query.split(chr(10))[-1].strip()}")
    print(f"Expected: {answer}")
    print(f"Got:      {output[0]}")
    print()
