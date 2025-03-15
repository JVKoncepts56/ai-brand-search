import streamlit as st
import openai
import json
import time
import re  # ✅ Ensure ASCII-safe IDs
from pinecone import Pinecone, ServerlessSpec

# ✅ Load API Keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# ✅ Initialize OpenAI and Pinecone
client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Define Pinecone Index
index_name = "negosh-matchmaking"
index = pc.Index(index_name)

# ✅ Function to Ensure ASCII-Safe Brand IDs
def clean_id(text):
    """Removes non-ASCII characters and replaces spaces with underscores."""
    return re.sub(r'[^\x00-\x7F]+', '', text).replace(" ", "_")

# ✅ Function to Generate Embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# ✅ Function to Search Pinecone with Filters
def search_pinecone(query_text, category=None, min_followers=None, region=None, brand_age=None, price_level=None):
    """Search Pinecone for brands similar to the input query with filters."""
    
    # ✅ Generate embedding for search query
    query_embedding = get_embedding(query_text)

    # ✅ Apply optional filters
    filter_conditions = {}
    if category:
        filter_conditions["category"] = {"$eq": category}
    if min_followers:
        filter_conditions["followers"] = {"$gte": min_followers}
    if region:
        filter_conditions["region"] = {"$eq": region}
    if brand_age:
        filter_conditions["brand_age"] = {"$lte": brand_age}
    if price_level:
        filter_conditions["price_level"] = {"$eq": price_level}

    # ✅ Query Pinecone with filters
    results = index.query(
        vector=query_embedding,
        top_k=5,  # Return top 5 matches
        include_metadata=True,
        filter=filter_conditions if filter_conditions else None
    )

    # ✅ Format Results
    formatted_results = []
    for match in results["matches"]:
        formatted_results.append({
            "brand": match["metadata"]["name"],
            "score": round(match["score"], 4),
            "category": match["metadata"].get("category", "Unknown"),
            "followers": match["metadata"].get("followers", "N/A"),
            "region": match["metadata"].get("region", "N/A"),
            "brand_age": match["metadata"].get("brand_age", "N/A"),
            "price_level": match["metadata"].get("price_level", "N/A"),
        })

    return formatted_results

# ✅ Streamlit UI
st.title("🔍 AI Brand Search")

query_text = st.text_input("Enter your search query", "")

category = st.selectbox("📌 Select a Category", ["All", "Cartoons", "Fashion", "Toys", "Food", "Sports"])
category = None if category == "All" else category

min_followers = st.number_input("📊 Minimum Followers", min_value=0, step=1000, value=0)

region = st.selectbox("🌍 Select a Region", ["All", "North America", "Europe", "Asia"])
region = None if region == "All" else region

brand_age = st.slider("📅 Founded Before Year", min_value=1900, max_value=2025, value=2025)

price_level = st.selectbox("💰 Select Price Level", ["All", "Luxury", "Mass Market"])
price_level = None if price_level == "All" else price_level

if st.button("🔍 Search Brands"):
    results = search_pinecone(query_text, category, min_followers, region, brand_age, price_level)

    if results:
        st.write("### 🔍 Search Results:")
        for result in results:
            st.write(f"🔹 **{result['brand']}** (Score: {result['score']:.4f})")
            st.write(f"📌 **Category:** {result['category']}, 📊 **Followers:** {result['followers']}, 🌍 **Region:** {result['region']}")
            st.write(f"📅 **Founded Before:** {result['brand_age']}, 💰 **Price Level:** {result['price_level']}")
            st.write("---")
    else:
        st.warning("🚨 No results found! Try adjusting filters.")


