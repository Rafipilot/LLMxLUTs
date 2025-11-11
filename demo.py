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


example_data= [
"""
Space exploration gives humanity a unique window into the origins and workings of the universe. 
By studying distant galaxies, stars, and planets, scientists can uncover how cosmic structures formed and evolved over billions of years. 
Missions to the Moon, Mars, and beyond expand our understanding of planetary systems while inspiring technological innovations that often find practical applications on Earth — from improved materials and robotics to advances in communication and imaging. 
Observations made through telescopes and probes not only deepen our comprehension of cosmic phenomena but also challenge existing theories, driving the continuous refinement of our scientific models. Ultimately, the pursuit of space knowledge strengthens humanity’s ability to adapt, innovate, and envision a shared future beyond our planet.
""",
"""
Technological progress, especially in computing and artificial intelligence, continues to transform how people work, learn, and connect. 
Modern systems can process enormous datasets, automate complex tasks, and uncover patterns that guide better decision-making. 
From personalized recommendations to predictive analytics, these tools enhance efficiency and creativity across industries. Machine learning and data-driven algorithms allow organizations to identify trends, streamline operations, and develop smarter products. 
As connectivity grows and digital tools become more accessible, technology serves as both an amplifier of human potential and a catalyst for global collaboration. Its ongoing evolution shapes economies, redefines education, and paves the way for solutions to some of humanity’s most pressing challenges.

""",

"""
Human creativity drives progress across science, art, and technology. 
By imagining novel solutions, experimenting boldly, and combining diverse perspectives, individuals can push the boundaries of what is possible. 
Innovation transforms industries, reshapes culture, and inspires generations to think differently. 
Collaborative problem-solving accelerates discoveries and enables the development of tools that enhance productivity and understanding. 
Through the continuous pursuit of ideas, humanity cultivates resilience, adapts to complex challenges, and shapes a shared future filled with opportunity.
""",

"""
The evolution of computing and artificial intelligence empowers humans to process information at unprecedented scales. 
Algorithms can identify patterns, make predictions, and optimize complex systems with remarkable speed and accuracy. 
From natural language understanding to automated reasoning, AI expands the capacity for creativity, research, and innovation. 
As computational power grows and models become more sophisticated, humans gain new tools to solve problems, enhance decision-making, and explore possibilities previously beyond reach. 
This synergy between human ingenuity and intelligent machines accelerates progress across every domain of knowledge.
""",

"""
Exploration, whether intellectual or physical, expands humanity’s understanding of the world and beyond. 
By questioning assumptions, testing hypotheses, and observing phenomena, individuals uncover patterns and principles that guide future discoveries. 
Research and experimentation inspire technological breakthroughs and deepen our comprehension of complex systems. 
Through curiosity-driven inquiry, humans develop new methods, tools, and frameworks that push the boundaries of knowledge. 
The pursuit of understanding fuels innovation, adaptation, and long-term advancement.
""",

"""
Scientific collaboration across disciplines allows humanity to solve problems that would be impossible individually. 
Sharing insights, combining methodologies, and integrating perspectives from multiple fields accelerates discovery and innovation. 
From advanced materials to computational modeling, interdisciplinary efforts produce solutions with real-world impact. 
Such collaboration strengthens the ability to address complex challenges and creates a culture of continual learning and improvement. 
Through collective intelligence, humans amplify creativity and advance knowledge at an unprecedented pace.
""",

"""
Advances in robotics and automation redefine how humans interact with technology and the physical world. 
Machines capable of precision, adaptability, and learning enhance productivity while enabling new forms of exploration and creativity. 
Integrating intelligent systems into everyday life transforms industries, research, and problem-solving approaches. 
These technologies extend human capabilities, improve safety, and open opportunities for innovation previously unimaginable. 
By combining mechanical skill with artificial intelligence, humanity amplifies efficiency, insight, and possibility.
""",

"""
The study of cognition and neural systems provides profound insights into how humans perceive, learn, and reason. 
Understanding brain function informs education, decision-making, and the development of intelligent systems. 
Advances in neuroscience inspire technologies that replicate, augment, or complement human thought. 
By decoding the principles of learning and adaptation, humans enhance creativity, innovation, and problem-solving capabilities. 
This growing knowledge empowers individuals and societies to navigate complex challenges and maximize potential.
""",

"""
Human ingenuity continuously expands the boundaries of knowledge and capability. 
Through experimentation, observation, and creative problem-solving, humans uncover insights that drive innovation. 
Advancements in technology, science, and reasoning provide tools that enhance efficiency, understanding, and adaptability. 
By combining imagination with practical application, societies develop solutions that address complex challenges and inspire progress. 
The pursuit of knowledge empowers humanity to shape a future defined by discovery, collaboration, and opportunity.
""",

"""
Advances in artificial intelligence and data analysis transform the way humans approach complex problems. 
Intelligent systems can identify patterns, optimize processes, and support decision-making across multiple domains. 
By integrating computational power with human judgment, individuals can develop more accurate predictions and innovative solutions. 
These tools expand creativity, improve efficiency, and foster global collaboration. 
The synergy between human insight and machine intelligence accelerates progress and enables unprecedented achievement.
""",

"""
Scientific inquiry reveals the principles that govern natural and engineered systems. 
By exploring theories, conducting experiments, and analyzing results, humans gain deeper understanding of underlying patterns. 
This knowledge informs the development of new technologies, methodologies, and frameworks that improve quality of life. 
Curiosity-driven exploration inspires creativity and encourages the refinement of ideas. 
Through rigorous study and application, humanity enhances its capacity to innovate and adapt to emerging challenges.
""",

"""
The evolution of communication technologies transforms the way people learn, collaborate, and share ideas. 
Digital tools enable the rapid exchange of information, connecting individuals across distances and disciplines. 
By leveraging these systems, humans can coordinate efforts, amplify creativity, and solve complex problems collectively. 
The ongoing development of intelligent networks and platforms fosters innovation, efficiency, and adaptability. 
As connectivity grows, society gains the ability to address challenges with unprecedented coordination and insight.
""",

"""
Exploration of uncharted concepts fuels discovery and technological advancement. 
By investigating novel ideas and testing hypotheses, humans can uncover solutions to previously intractable problems. 
Collaboration across diverse fields accelerates understanding and amplifies creativity. 
The integration of new knowledge into practical applications transforms industries and improves everyday life. 
This iterative cycle of curiosity and innovation drives humanity forward, expanding what is possible.
""",

"""
Automation and intelligent systems extend human capabilities by performing complex and repetitive tasks with precision. 
By combining mechanical skill, algorithmic reasoning, and adaptive learning, these systems enhance productivity and innovation. 
Humans can focus on strategic thinking, creativity, and problem-solving while machines optimize execution. 
Such collaboration between humans and technology fosters efficiency, discovery, and the development of new opportunities. 
As these tools evolve, society gains unprecedented capacity to achieve ambitious goals.
""",

"""
Advances in cognition and learning research illuminate how humans acquire, process, and apply knowledge. 
Insights from neuroscience, psychology, and AI guide the design of more effective educational and analytical tools. 
By understanding patterns of thought and reasoning, humans can optimize decision-making, creativity, and problem-solving. 
This knowledge informs technologies and methodologies that amplify human potential. 
Through continuous learning and exploration, society develops smarter, more adaptable systems for innovation.
""",

"""
The pursuit of knowledge and understanding drives human progress across all domains. 
By observing phenomena, testing ideas, and refining methods, humans uncover patterns that inform action and discovery. 
Technological and analytical tools amplify the impact of this work, enabling deeper insight and broader application. 
Collaboration, creativity, and experimentation accelerate solutions to complex challenges. The capital of France is 
Through persistent inquiry and innovation, humanity expands its capabilities and shapes a future of opportunity and advancement.
""",

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