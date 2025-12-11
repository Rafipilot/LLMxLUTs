from pathlib import Path

from main import Tokenizer, Transformer, generate

device = "cuda"
model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 20  # bump this if you want longer answers

# --- Load model once ---
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

# --- Configure which blocks have WNN / LUTs ---
# You can add more indices here if you want multiple LUT-enabled blocks
transformer.layers[-1].wnn_block = True
# transformer.layers[-5].wnn_block = True  # example

# --- Default LUT hyperparams ---
for block in transformer.layers:
    # only meaningful if block.wnn_block == True
    if not hasattr(block, "residual_scale"):
        block.residual_scale = 0.5
    if not hasattr(block, "CS_threshold"):
        block.LUT.CS_threshold = -1.0  # or whatever default you like


def edit_lut_hparams(transformer):
    """Interactive editor for residual_scale and CS_threshold on WNN blocks."""
    wnn_blocks = [
        i for i, blk in enumerate(transformer.layers)
        if getattr(blk, "wnn_block", False)
    ]

    if not wnn_blocks:
        print("No WNN blocks found (wnn_block=False everywhere).")
        return

    print("\nCurrent WNN block settings:")
    for idx in wnn_blocks:
        blk = transformer.layers[idx]
        rs = getattr(blk, "residual_scale", None)
        th = getattr(blk, "CS_threshold", None)
        print(f"  Block {idx}: residual_scale={rs}, CS_threshold={th}")

    print("\nEnter new values for each block, or press Enter to keep the current value.\n")

    for idx in wnn_blocks:
        blk = transformer.layers[idx]

        # Residual scale
        rs_current = getattr(blk, "residual_scale", None)
        rs_in = input(f"Residual scale for block {idx} (current: {rs_current}): ").strip()
        if rs_in:
            try:
                blk.residual_scale = float(rs_in)
            except ValueError:
                print("  Invalid number, keeping existing residual_scale.")

        # Threshold
        th_current = getattr(blk, "CS_threshold", None)
        th_in = input(f"CS_threshold for block {idx} (current: {th_current}): ").strip()
        if th_in:
            try:
                blk.LUT.CS_threshold = float(th_in)
            except ValueError:
                print("  Invalid number, keeping existing CS_threshold.")

    print("\nUpdated WNN block settings:")
    for idx in wnn_blocks:
        blk = transformer.layers[idx]
        print(
            f"  Block {idx}: residual_scale={blk.residual_scale}, "
            f"CS_threshold={blk.CS_threshold}"
        )
    print()


def teach_example(transformer, tokenizer):
    """
    Interactive wrapper around trainLUT.

    You type the context (e.g. 'User: ...\\nAssistant: ')
    and the target completion (label) you want the LUT to memorise.
    """
    print("\n--- /teach mode ---")
    label_context = input(
        "Context (e.g. 'User: What is Astarus?\\nAssistant: '): "
    ).strip()
    label = input(
        "Target answer to train into the LUT (label): "
    ).strip()

    if not label:
        print("Empty label – nothing trained.\n")
        return

    transformer.trainLUT(
        tokenizer=tokenizer,
        label=label,
        lm_head=None,
        label_context=label_context if label_context else None,
    )
    print("LUT updated for this example.\n")


def chat_loop(transformer, tokenizer, max_tokens=128):
    print("LUT-LLM CLI")
    print("Commands:")
    print("  /teach      -> teach a new LUT example")
    print("  /residuals  -> edit residual_scale and CS_threshold for WNN blocks")
    print("  /quit       -> exit")
    print("Anything else -> treated as a user question.\n")

    while True:
        text = input("You (/teach, /residuals, /quit or question): ").strip()

        if not text:
            continue

        # Exit
        if text.lower() in {"/quit", "/exit"}:
            print("Bye.")
            break

        # Edit residuals/thresholds in-place
        if text.lower() == "/residuals":
            edit_lut_hparams(transformer)
            continue

        # Teach a new LUT example
        if text.lower() == "/teach":
            teach_example(transformer, tokenizer)
            continue

        # Normal question -> generate
        prompt = f"User: {text}\nAssistant: "
        res, _logits = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
        )

        # Assuming generate returns a list of strings
        answer = res[0] if isinstance(res, (list, tuple)) else res
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    # if you still want an initial manual teach, you can call trainLUT once here:
    # transformer.trainLUT(
    #     tokenizer=tokenizer,
    #     label="Astarus is building continuously trainable LLMs.",
    #     lm_head=None,
    #     label_context="User: What is Astarus\nAssistant: ",
    # )

    chat_loop(transformer, tokenizer, max_tokens=max_tokens)
