import streamlit as st
import openai
import json
import time
import re  # ✅ Ensure ASCII-safe IDs
from pinecone import Pinecone, ServerlessSpec

# ✅ Retrieve API Key Securely
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

# ✅ Function to Ensure ASCII-Safe Brand IDs
def clean_id(text):
    """Removes non-ASCII characters and replaces spaces with underscores."""
    return re.sub(r'[^\x00-\x7F]+', '', text).replace(" ", "_")
    
# ✅ Function to Generate Embeddings and Upload to Pinecone (Only for New Brands)
def generate_and_upload_embeddings():
    """Ensure all brands are embedded and uploaded to Pinecone, handling errors."""
    print("\n🔍 Checking which brands already exist in Pinecone...")
     
    index_stats = index.describe_index_stats()
    existing_count = index_stats.get("total_vector_count", 0)
    print(f"✅ Pinecone currently contains {existing_count} vectors.")

    vectors = []
    new_brands = 0

    for brand in brand_data:
        brand_id = clean_id(brand["brand_name"])  # ✅ Ensure ASCII-compatible IDs
        
        print(f"🆕 Embedding brand: {brand['brand_name']}...")
        try:
            response = client.embeddings.create(
                input=brand["processed_text"],
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
    
            vectors.append({
                "id": brand_id,
                "values": embedding, 
                "metadata": {
                    "name": brand["brand_name"],
                    "category": brand.get("category", "Unknown"),
                    "description": brand.get("description", "No description available.")
                }
            })
            new_brands += 1
    
        except Exception as e:
            print(f"⚠️ OpenAI embedding failed for {brand['brand_name']}: {e}")
            continue  # ✅ Skip this brand and move to the next
    
    if new_brands == 0:
        print("\n✅ No new brands to embed.")
        return
        
    # ✅ Upload in batches & retry failed uploads
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        print(f"📤 Uploading batch {i // BATCH_SIZE + 1} of {len(vectors) // BATCH_SIZE + 1}...")
                
        for retry in range(3):  # ✅ Retry up to 3 times
            try:
                index.upsert(vectors=batch)
                print(f"✅ Successfully uploaded batch {i // BATCH_SIZE + 1}")
                break  # ✅ If successful, move to the next batch
            except Exception as e:   
                print(f"⚠️ Upload failed (attempt {retry + 1}): {e}")
                time.sleep(5)  # ✅ Wait before retrying
            
    print(f"\n✅ {new_brands} new brand embeddings uploaded successfully!")
        
# ✅ Function to Search Pinecone with Explanations
def search_pinecone(query_text, top_k=5):
    """Search Pinecone for brands similar to the input query and include explanations."""
    print(f"🔍 Generating embedding for query: {query_text}")
        
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding
        
    print(f"🔍 Query embedding generated: {query_embedding[:5]}...")  # ✅ Print first 5 values
                
    print("🔍 Querying Pinecone for best matches...")   
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if not results['matches']:
        return "🚨 No results found! Try adjusting your query."

    # ✅ Format results with explanations
    formatted_results = []
    for match in results["matches"]:
        brand_name = match["metadata"].get("name", "Unknown Brand")
        category = match["metadata"].get("category", "No Category Info")
        description = match["metadata"].get("description", "No Description Available")
        score = round(match["score"], 4)

        explanation = (
            f"🔹 **{brand_name}** (Score: {score})\n"
            f"📌 **Category:** {category}\n"
            f"📝 **Why this matched?** {description}\n"
            "—" * 20
        )

        formatted_results.append(explanation)

    return "\n\n".join(formatted_results)

# ✅ Streamlit UI
st.title("🔍 AI Brand Search")

query_text = st.text_input("Enter your search query:")

if st.button("Search"):
    if query_text:
        st.write(f"**Searching for:** {query_text}")
        results = search_pinecone(query_text)
        st.markdown(results)
    else:
        st.warning("⚠️ Please enter a search query.")

