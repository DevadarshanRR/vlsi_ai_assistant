import os
import streamlit as st
from groq import Groq
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Gamma Intelligence",
    layout="wide"
)

# ---------------------------------------
# PROFESSIONAL DARK THEME
# ---------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #343541;
    color: white;
}

.stChatMessage {
    background-color: #444654;
    border-radius: 12px;
    padding: 12px;
}

.stTextInput > div > div > input {
    background-color: #40414F;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #202123;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# LOAD API KEY (SECURE)
# ---------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Configure it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("Gamma Intelligence")
    st.markdown("Semiconductor and VLSI Engineering Assistant")
    st.markdown("RAG Architecture with Llama 3")

# ---------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ---------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# ---------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------
def build_index(text):
    chunks = text.split("\n\n")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    index, chunks = build_index(text)

    st.session_state.index = index
    st.session_state.chunks = chunks

    st.success("PDF processed successfully.")

# ---------------------------------------
# SEARCH FUNCTION
# ---------------------------------------
def search(query, k=5):
    if "index" not in st.session_state:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = st.session_state.index.search(
        np.array(query_embedding), k
    )

    return [st.session_state.chunks[i] for i in indices[0]]

# ---------------------------------------
# GENERATE ANSWER
# ---------------------------------------
def generate_answer(query):
    context = ""

    if "index" in st.session_state:
        retrieved_chunks = search(query, k=5)
        context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are Gamma Intelligence, an expert in semiconductor materials and VLSI systems.
    Provide structured, precise, and technically rigorous explanations.

    Use context if available.

    Context:
    {context}

    Question:
    {query}

    Structure the answer as:
    1. Definition
    2. Physical mechanism
    3. Mathematical explanation if applicable
    4. Practical significance
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

# ---------------------------------------
# CHAT INTERFACE
# ---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Gamma Intelligence")
st.subheader("Semiconductor and VLSI Engineering Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about MOSFET, doping, band theory, scaling, or your uploaded PDF.")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    answer = generate_answer(user_input)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )