Credit to the original creators of the GPT-2 PyTorch implementation, TaeHwan Jung (@graykode), and to OpenAI for the original GPT-2 model.  
Credit to Mistral AI for the original Mistral implementation.

# LLMs × LUTs

This repo explores combining large open-source language models with a lightweight lookup table, to study how this extra component affects performance and behaviour at scale.

The goal is to keep the base models unchanged while adding a small, fast layer that can adapt to new data and use cases, and then compare “base vs. base + LUT” across different setups.

### Current model support

- **GPT-2** (PyTorch)
- **Mistral-7B**

### What you can do

- Run LUT-augmented inference on top of existing open-source LLMs.
- Toggle between plain model and LUT-enhanced runs for side-by-side comparison.
- Experiment locally with your own prompts / datasets to see how the lookup table behaves.

### Getting started

1. Clone the repo and install dependencies (Python + PyTorch required).
2. Choose a model backend (GPT-2 or Mistral-7B).
3. Run one of the example scripts to spin up a local demo.

More detailed examples and docs are in the repo, and will be expanded over time.
