import os
import streamlit as st
from groq import Groq
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===============================
# PAGE CONFIG (ChatGPT Style)
# ===============================
st.set_page_config(page_title="Semiconductor AI", layout="wide")

# Dark Theme CSS (ChatGPT-like)
st.markdown("""
<style>
body {
    background-color: #343541;
    color: white;
}
.stChatMessage {
    background-color: #444654;
    border-radius: 10px;
    padding: 10px;
}
.stTextInput > div > div > input {
    background-color: #40414F;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOGIN SYSTEM
# ===============================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "vlsi123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ===============================
# LOAD API KEY
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("API key not found. Set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# ===============================
# EMBEDDING MODEL
# ===============================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ===============================
# PDF PROCESSING
# ===============================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def build_index(text):
    chunks = text.split("\n\n")
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
    st.success("PDF processed successfully!")

# ===============================
# SEARCH FUNCTION
# ===============================
def search(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = st.session_state.index.search(
        np.array(query_embedding), k
    )
    return [st.session_state.chunks[i] for i in indices[0]]

# ===============================
# GENERATE ANSWER
# ===============================
def generate_answer(query):
    context = ""
    if "index" in st.session_state:
        retrieved_chunks = search(query, k=5)
        context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a semiconductor and VLSI expert.
    Use only the context if available.

    Context:
    {context}

    Question:
    {query}

    Provide structured technical answer.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content

# ===============================
# CHAT MEMORY
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 Semiconductor & VLSI AI Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about MOSFET, CMOS, or your uploaded PDF...")

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