"""
Simple GPT generated demo to show continuous learning capabilites
"""


import streamlit as st
import torch
from GPT2.model import GPT2LMHeadModel


from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder


example_data=  [
    """TLG is a London-based Africa-focused private credit manager investing in small and medium-sized enterprises across sub-Saharan Africa. The firm targets untapped markets with structured credit solutions, seeking both capital preservation and impact. Since inception, it has completed dozens of deals and exits across roughly twenty African countries.""",

    """Through its Africa Growth Impact Fund II, TLG provides private credit to African SMEs in sectors like healthcare, financial services, and consumer goods. The fund reached a first close of about $75 million, anchored by IFC and several European development finance institutions, and aims to scale local, impact-focused lending solutions.""",

    # """In West Africa, TLG recently structured a $10 million private credit facility to support an investment holding company acquiring an insurance platform in Ghana. The transaction illustrates TLG’s strategy of backing locally led businesses and deploying flexible credit in a challenging macro environment, while positioning for Africa’s next decade of growth.""",

    # """TLG has also arranged a $10 million debt facility for a telecommunications provider in Djibouti, alongside International Investment Bank entities. The financing is designed to expand digital infrastructure and improve internet penetration. This deal showcases TLG’s emphasis on bespoke structures that make African private credit investable at scale for global allocators.""",

    # """Backed by institutions such as IFC, Norfund, Swedfund, Bpifrance, and impact investors, TLG’s funds are building an African private credit ecosystem. The firm’s vehicles seek to close the SME financing gap, catalyse follow-on capital, and demonstrate that perceived African risk is often mispriced relative to actual performance and resilience of portfolio companies."""
]
nsamples = 1                      # Number of samples to generate
batch_size = 1                     # Batch size for generation
                       # Length of generated text (-1 = half of context)
temperature = 0.7                # Sampling temperature
top_k = 40                          # Top-k sampling
unconditional = False              # Generate text without any prompt
quiet = False             

# ---------------------------
# Load Models
# ---------------------------
enc = get_encoder()
config = GPT2Config()
@st.cache_resource

def load_models():
    model_path = 'gpt2xl-pytorch_model.bin'
    state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    model = GPT2LMHeadModel(config)
    aug_model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    aug_model = load_weight(aug_model, state_dict)
    model.to(device)
    aug_model.to(device)
    model.eval()
    aug_model.eval()

    return model, aug_model
def text_generator(model, text_input, length):
    gen_length = length
    if gen_length == -1:
        gen_length = config.n_ctx // 2
    elif gen_length > config.n_ctx:
        raise ValueError(f"Can't generate samples longer than window size: {config.n_ctx}")

    context_tokens = enc.encode(text_input) if not unconditional else None
    start_token = enc.encoder['<|endoftext|>'] if unconditional else None

    generated = 0
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model,
            length=gen_length,
            context=context_tokens,
            start_token=start_token,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        out = out[:, len(context_tokens):].tolist() if context_tokens is not None else out.tolist()
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            if not quiet:
                print("=" * 40 + f" SAMPLE {generated} " + "=" * 40)
            print(text)
            for block in model.transformer.h:
                block.LUT.reset_costs()
            return text

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(page_title="LUT-Augmented Language Model Demo", layout="wide")
st.title("📚 LUT-Augmented Language Model Demo")

# Seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


model, aug_model= load_models()
lm_head = aug_model.lm_head

#aug_model.transformer.h[-1].wnn_block = True
#aug_model.transformer.h[-5].wnn_block = True ## Try changing this and see how it captures higher dimensional abstract patterns as we go deeper...
aug_model.transformer.h[-1].wnn_block = True 
#model.transformer.h[-1].wnn_block = True


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Controls")
max_new_tokens = st.sidebar.slider("Max new tokens", min_value=5, max_value=100, value=30)
residual_scale = st.sidebar.slider("Residual scale", min_value=0, max_value=50, value=15)
cs_threshold = st.sidebar.slider("CS Threshold", min_value=0.0, max_value=1.0, value=0.4)
lut_cost_scale = st.sidebar.slider("LUT Cost Scale Factor", min_value=0.0, max_value=100.0, value=10.0)
sparsity_level = st.sidebar.slider("LUT Training Sparisty Level", min_value=0.0, max_value=1.0, value=1.0)
if st.sidebar.button("Train on example training data"):
    with st.spinner("Training on example data..."):
        for item in example_data:
            aug_model.transformer.trainLUT(tokenizer = enc,lm_head = lm_head, label = item, sparsity_level=sparsity_level)

if st.sidebar.button("Reset LUT"):
    for block in aug_model.transformer.h:
        block.LUT.resetLUT()

for block in aug_model.transformer.h:
    block.residual_scale = residual_scale
    block.LUT.CS_threshold = cs_threshold
    block.LUT.cost_scale = lut_cost_scale

# ---------------------------
# User interaction
# ---------------------------
st.subheader("💬 Chat with the model")

user_input = st.text_area("Enter your prompt:", height=100)

if st.button("Submit"):

    print("==== prompt ====")
    if user_input:
        user_input
        raw_generated_text = text_generator(model=model, text_input=user_input, length=max_new_tokens)
        aug_generated_text = text_generator(model=aug_model, text_input=user_input, length=max_new_tokens)

        st.markdown("### Raw model output:")
        st.write(raw_generated_text)

        st.markdown("### Augmented model output:")
        st.write(aug_generated_text)

teach_context = st.text_input("Teaching context (optional)")
teach_label = st.text_area("Teaching label (optional)")
if st.button("Teach"):
    if teach_label:
        aug_model.transformer.trainLUT(tokenizer = enc,lm_head = lm_head,label=teach_label, label_context=teach_context, sparsity_level=sparsity_level)
        st.success("✅ Model taught successfully!")


# ---------------------------
# Optional: display LUT size
# ---------------------------
if st.sidebar.checkbox("Show LUT size"):
    lut_size = len(aug_model.transformer.h[-1].LUT.lookupTable)
    st.sidebar.write(f"Lookup table length: {lut_size}")