import streamlit as st
import openai
from pinecone import Pinecone

# 🔐 Load OpenAI API Key securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔐 Load Pinecone API Key (use secrets if needed)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index_name = "negosh-matchmaking"
index = pc.Index(index_name)

# Streamlit Interface
st.title("AI Brand Search")
st.write("Enter your query below to find the best brand matches.")

query = st.text_input("Enter your brand search query:")

if st.button("Search"):
    st.write(f"Searching for: **{query}**")

    # Generate embedding for query
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding

    # Query Pinecone
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Display results
    st.subheader("Results:")
    if "matches" in results:
        for match in results['matches']:
            brand_name = match['metadata'].get('name', 'Unknown Brand')
            score = match['score']
            st.write(f"🔹 {brand_name} (Score: {score:.4f})")
    else:
        st.write("No matches found.")

