import os
import streamlit as st
from groq import Groq

# -------------------
# Load API Key
# -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("API key not found. Set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------------------
# UI
# -------------------
st.title("🔬 Semiconductor & VLSI AI Assistant")

user_input = st.text_input("Ask a question:")

if st.button("Generate Answer"):
    if user_input:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        st.write(answer)