import streamlit as st
import openai
import json
import time
import re
from pinecone import Pinecone, ServerlessSpec

# ✅ Retrieve API key securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "negosh-matchmaking"

# ✅ Check if Pinecone index exists; create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(30)  # ✅ Wait for the index to be ready

# ✅ Connect to Pinecone index
index = pc.Index(index_name)

# ✅ Streamlit UI
st.title("🔍 AI Brand Search")
st.write("Enter your query below to find the best brand matches.")

# ✅ User input for search
query_text = st.text_input("Enter your search query", "")

# ✅ Streamlit dropdown filters
category_filter = st.selectbox("📌 Select a Category", ["All", "Toys", "Fashion", "Sports", "Tech", "Entertainment"])
min_followers = st.slider("📊 Minimum Followers", 0, 1000000, 0)
region_filter = st.selectbox("🌍 Select a Region", ["All", "North America", "Europe", "Asia", "Global"])
founded_before = st.slider("📅 Founded Before Year", 1900, 2025, 2025)
price_level = st.selectbox("💰 Select Price Level", ["All", "Budget", "Mid-Range", "Premium", "Luxury"])

# ✅ Function to Ensure ASCII-Safe Brand IDs
def clean_id(text):
    """Removes non-ASCII characters and replaces spaces with underscores."""
    return re.sub(r'[^\x00-\x7F]+', '', text).replace(" ", "_")

# ✅ Function to Search Pinecone with Filters
def search_pinecone(query_text, category=None, min_followers=None, region=None, founded_before=None, price_level=None, top_k=5):
    """Search Pinecone for brands similar to the input query with filters."""
    if not query_text:
        st.warning("⚠️ Please enter a search query.")
        return

    st.write(f"🔍 Searching for: {query_text}")

    # ✅ Generate the embedding for the query
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding

    # ✅ Build filter conditions based on user selections
    filter_conditions = {}
    if category and category != "All":
        filter_conditions["category"] = {"$eq": category}
    if min_followers and min_followers > 0:
        filter_conditions["followers"] = {"$gte": min_followers}
    if region and region != "All":
        filter_conditions["region"] = {"$eq": region}
    if founded_before and founded_before < 2025:
        filter_conditions["founded"] = {"$lte": founded_before}
    if price_level and price_level != "All":
        filter_conditions["price_level"] = {"$eq": price_level}

    # ✅ Query Pinecone with filters
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_conditions if filter_conditions else None
    )

    # ✅ Process results to remove duplicates and improve readability
    unique_matches = {}
    for match in results.get("matches", []):
        brand_name = match['metadata'].get("name", "Unknown Brand")
        category = match['metadata'].get("category", "No Category Info")
        description = match['metadata'].get("description", "No Description Available")

        # ✅ Keep only the highest-scoring match for each brand
        if brand_name not in unique_matches or match["score"] > unique_matches[brand_name]["score"]:
            unique_matches[brand_name] = {
                "score": match["score"],
                "category": category,
                "description": description
            }

    # ✅ Display results in a structured way
    if not unique_matches:
        st.error("🚨 No results found! Try adjusting filters.")
        return

    st.subheader("🔎 **Top Matches**:")
    for brand, data in unique_matches.items():
        st.markdown(f"""
        **🔹 {brand} (Score: {data['score']:.4f})**
        - 📌 **Category:** {data['category']}
        - 📝 **Why this matched?** {data['description']}
        ---
        """)

# ✅ Run search when user submits a query
if st.button("🔎 Search"):
    search_pinecone(query_text, category_filter, min_followers, region_filter, founded_before, price_level)

