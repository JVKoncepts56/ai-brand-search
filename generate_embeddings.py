import openai
import pinecone
import pandas as pd

# Set your API keys
openai.api_key = "your_openai_api_key"
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")

# Load CSV file
df = pd.read_csv("brand_licensee_data.csv")

# Function to generate AI embeddings
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Generate embeddings for each description
df["embedding"] = df["Description"].apply(generate_embedding)

# Save updated CSV
df.to_csv("brand_licensee_with_embeddings.csv", index=False)

print("Embeddings successfully generated and saved!")

