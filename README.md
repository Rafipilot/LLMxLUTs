Credit to the original creators of the GPT-2 PyTorch implementation, TaeHwan Jung (@graykode), and to OpenAI for the original GPT-2 model.
Credit to Mistral AI for the original Mistral implementation.

---

# LLMs × LUTs (Astarus AI Core)

This repo is Astarus AI’s core work on **LUT-based language models**: large open-source LLMs augmented with a small, fast **lookup table (LUT)** that can adapt on the fly without retraining the full model.

The core idea:

* Keep the **base model frozen and unchanged**.
* Add a lightweight LUT layer that:

  * Stores compact “memories” learned from examples.
  * Is **cheap to update** (no full backprop over the whole model).
  * Can be turned on/off or tuned per use case.
* Compare **“base vs. base + LUT”** to see how this extra component changes:

  * Accuracy on narrow domains.
  * Consistency and hallucination behaviour.
  * Ability to continuously learn from new data.

This repo is where we experiment with those ideas at the code level.

---

## Current model support

Right now, the repo focuses on two backends:

* **GPT-2**

  * Good for quick experiments and understanding the mechanics of LUTs.
  * Smaller, faster to iterate on.

* **Mistral-7B**

  * Modern, capable instruction model.
  * Lets us see how LUTs behave in a more realistic, production-like setting.

Both are used in a way that **does not modify their core weights**. All adaptation happens in the LUT component and its residual wiring.

---

## How the LUT component fits in (high-level)

Conceptually, the LUT acts as a tiny, trainable memory module plugged into the transformer:

1. **Base model forward pass** runs as usual.
2. At certain blocks, we:

   * Extract an internal representation (hidden state).
   * Use it to **query the LUT** (e.g. via cosine similarity).
3. The LUT returns a small correction / “memory vector”.
4. That signal is **merged back through a residual pathway**, alongside attention and MLP outputs.
5. During LUT training:

   * The base model is frozen.
   * We compute gradients only for LUT parameters / entries.
   * Updates are cheap and can be applied frequently.

In other words: the LUT is a **sidecar learner** attached to the LLM, not a replacement for it.

---

## Typical workflows

Some standard things you can do with this repo:

1. **Baseline vs LUT comparison**

   * Run a prompt through the base GPT-2 or Mistral-7B model.
   * Turn on LUT usage and run the same prompt again.
   * Inspect how answers change once the LUT has been trained on a small domain.

2. **Teach the model a new domain**

   * Pick a set of Q&A pairs or short documents.
   * Run a LUT-training script that:

     * Feeds examples through the model.
     * Updates the LUT entries using gradients.
   * Evaluate:

     * Does the model answer domain questions more accurately?
     * Does it retain its general abilities when the LUT is active?

3. **A/B testing LUT configurations**

   * Vary:

     * Which transformer blocks get LUTs.
     * Residual scaling factors.
     * Similarity thresholds and cost functions.
   * Measure how these changes affect stability, specificity, and hallucination rate.

---

## Getting started

1. **Clone the repo**

   ```bash
   git clone https://github.com/AstarusAI/LLMxLUTs.git
   cd LLMxLUTs
   ```

2. **Install dependencies**

   You’ll need:

   * Python (3.10+ recommended)
   * PyTorch with CUDA (if you want GPU support)
   * Standard Python dependencies (see `requirements.txt` or the project’s docs)

   ```bash
   pip install -r requirements.txt
   ```

3. **Choose a backend**

   * For GPT-2:

     * Make sure the code can access the GPT-2 weights (typically via Hugging Face or local download).
   * For Mistral-7B:

     * Download / place the Mistral checkpoint in the expected folder, e.g. `mistral-7B-Instruct-v0.3`.

4. **Run an example script**

   The repo includes example scripts that:

   * Load a base model.
   * Attach LUT layers to selected transformer blocks.
   * Run prompts with and without LUTs.

   Look for scripts named along the lines of:

   * `run_gpt2_lut_demo.py`
   * `run_mistral_lut_demo.py`

   (Exact filenames may differ; check the repo for the latest examples.)

---

## Roadmap (high-level)

Planned directions for this repo include:

* Cleaner APIs for:

  * Attaching / detaching LUTs to arbitrary transformer blocks.
  * Switching between multiple LUTs at runtime.
* More robust training utilities:

  * Helpers for building Q&A corpora and evaluation sets.
  * Tools to measure “before vs after LUT” performance.
* Additional backends:

  * Extending beyond GPT-2 and Mistral-7B where it makes sense.

Docs and examples will be expanded over time as the experiments evolve.

---

If you’re interested in LUT-based LLMs, continuous learning, or per-tenant specialization, this repo is meant to be a practical playground for those ideas.
