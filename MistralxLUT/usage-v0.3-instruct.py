from pathlib import Path

from main import Tokenizer, Transformer, generate

device = "cuda"
model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 25

# --- Load model once ---
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

# --- Configure which blocks have WNN / LUTs ---
# You can add more indices here if you want multiple LUT-enabled blocks
transformer.layers[-1].wnn_block = True
# transformer.layers[-5].wnn_block = True  # example

# --- Default LUT hyperparams ---
for block in transformer.layers:
    block.residual_scale = 0.75

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
