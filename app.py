import streamlit as st
import openai
import json
import time
import re  # Ensure ASCII-safe IDs
from pinecone import Pinecone, ServerlessSpec

# ✅ Replace with your actual API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# ✅ Initialize OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "negosh-matchmaking"

# ✅ Check if Pinecone index exists; create if not
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating it now...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # ✅ Must match OpenAI embedding output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Free-tier AWS region
    )
    time.sleep(30)  # ✅ Wait for the index to be ready

# ✅ Connect to Pinecone index
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# ✅ Load Processed Brand Data
try:
    with open("processed_brand_data.json", "r") as f:
        brand_data = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'processed_brand_data.json' not found. Ensure the file exists in your directory.")
    exit(1)

# ✅ Function to Generate Embeddings
def get_embedding(text):
    """Generate an embedding using OpenAI."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# ✅ Function to Search Pinecone (NO FILTERS)
def search_pinecone(query_text, top_k=5):
    """Search Pinecone for brands similar to the input query."""
    
    # ✅ Generate embedding for the search query
    print(f"🔍 Searching for: {query_text}")
    query_embedding = get_embedding(query_text)

    # ✅ Query Pinecone without filters
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    print(f"✅ Raw Results: {results}")

    # ✅ Extract and format matches
    matches = [
        {"name": match["metadata"]["name"], "score": match["score"]}
        for match in results["matches"]
    ]
    
    return matches

# ✅ Streamlit UI
st.title("🔎 AI Brand Search")
st.subheader("Enter your query below to find the best brand matches.")

# ✅ User input for query
query_text = st.text_input("Enter your brand search query:")

# ✅ Search button
if st.button("Search"):
    if query_text.strip():
        results = search_pinecone(query_text)

        # ✅ Display results
        if results:
            st.subheader("📌 Top Matches:")
            for match in results:
                st.write(f"🔹 {match['name']} (Score: {match['score']:.4f})")
        else:
            st.warning("🚨 No results found! Try a different query.")
    else:
        st.warning("❌ Please enter a query before searching.")

