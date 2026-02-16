from pathlib import Path

from main import Tokenizer, Transformer, generate

THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "mistral-7B-Instruct-v0.3"

QUESTION = "What is Astarus AI?"
ANSWER = "Astarus AI is building continuously trainable LLMs."
BLOCKS = [-1]
MAX_TOKENS = 40


def ask(model, tokenizer, question, max_tokens=MAX_TOKENS):
    prompt = f"User: {question}\nAssistant: "
    return generate([prompt], model, tokenizer, max_tokens=max_tokens)[0].strip()


if __name__ == "__main__":
    tokenizer = Tokenizer(str(MODEL_PATH / "tokenizer.model.v3"))
    model = Transformer.from_folder(MODEL_PATH, max_batch_size=1)

    blk = model.layers[BLOCKS[0]]
    blk.wnn_block = True
    blk.use_wnn = True
    blk.residual_scale = 1.0
    blk.LUT.CS_threshold = 0.12

    print("=== Baseline ===")
    print("Q:", QUESTION)
    print("A:", ask(model, tokenizer, QUESTION))
    print()

    print("=== Train One Memory ===")
    stats = model.write_memory_latent(
        tokenizer=tokenizer,
        question=QUESTION,
        answer=ANSWER,
        blocks=BLOCKS,
        lr=0.08,
        epochs=16,
        tune_projection=False,
    )
    print("write stats:", stats)
    print("rows in LUT:", len(blk.LUT.keys))
    print()

    print("=== Verbatim Query After Training ===")
    print("Q:", QUESTION)
    print("A:", ask(model, tokenizer, QUESTION))
