import streamlit as st
import openai

# ✅ Retrieve API key securely from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Streamlit UI
st.title("AI Brand Search")
st.write("Enter your query below to find the best brand matches.")

query = st.text_input("Enter your brand search query:")

if st.button("Search"):
    if query:
        st.write(f"Searching for: {query}")
        # Example API call (modify based on your actual function)
        response = client.embeddings.create(input=query, model="text-embedding-ada-002")
        st.write("Results:", response)
    else:
        st.error("Please enter a query.")

