import re
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = "http://185.182.187.174:6333/"
QDRANT_API_KEY = "16a3a57d"
OLD_COLLECTION = "companies_jina_final"
NEW_COLLECTION = "companies_jina_final_v2"

# API URL - Confirmed working in previous tests
JINA_API_URL = "https://python-embedding.34.166.92.24.sslip.io/embed"

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_clean_tech_list(raw_tech_list):
    """Extracts clean tech names and returns them as a LIST."""
    if not raw_tech_list:
        return []
        
    full_string = str(raw_tech_list)
    
    # 1. Pull values from the messy dictionary formats
    tech_keywords = re.findall(r"'([^']+)': '([^']+)'", full_string)
    clean_technologies = []
    
    for key, value in tech_keywords:
        if value and value not in ["COMPLETED", "U.S. Server Location"] and not key.endswith("Global Trends"):
            clean_technologies.append(value.strip())
            
    # 2. Pull standalone strings from the array
    if isinstance(raw_tech_list, list):
        for item in raw_tech_list:
            if isinstance(item, str) and ":" not in item and "{" not in item:
                clean_technologies.append(item.strip())
                
    # Return a deduplicated list
    return list(set(clean_technologies))


def get_jina_embedding(text):
    """Fetches the vector from your running Jina v5 API."""
    if not text or text.strip() == "":
        return None 
    
    payload = {"text": text} 
    
    try:
        # 60s timeout to allow for model inference time
        response = requests.post(JINA_API_URL, json=payload, timeout=60)
        response.raise_for_status() 
        data = response.json()
        
        if 'embedding' in data:
            return data['embedding']
        else:
            raise ValueError(f"Unrecognized response format: {data}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API Error: {e}")
        return None


# ==========================================
# 3. PREPARE THE NEW COLLECTION
# ==========================================
print(f"Fetching schema from old collection: {OLD_COLLECTION}...")
old_info = client.get_collection(OLD_COLLECTION)

# Detect size from old "chunk_text" vector
DETECTED_OLD_SIZE = old_info.config.params.vectors["chunk_text"].size
print(f"✅ Old vector size detected: {DETECTED_OLD_SIZE}")

# Prepare named vectors configuration for the new collection
new_vectors_config = {}

# Copy all existing vector parameters from old collection
for name, params in old_info.config.params.vectors.items():
    new_vectors_config[name] = models.VectorParams(
        size=params.size, 
        distance=params.distance
    )

# Determine the size of the new Jina v5 embedding
print("Testing Jina v5 output dimension from server...")
test_vec = get_jina_embedding("test")
if test_vec is None:
    print("❌ Critical: Could not connect to Jina server. Check your URL.")
    exit(1)

JINA_V5_SIZE = len(test_vec)
print(f"✅ Jina v5 output size: {JINA_V5_SIZE}")

# Add the new 'technology' vector configuration
new_vectors_config["technology"] = models.VectorParams(
    size=JINA_V5_SIZE, 
    distance=models.Distance.COSINE
)

print(f"Creating new collection: {NEW_COLLECTION}...")
client.recreate_collection(
    collection_name=NEW_COLLECTION,
    vectors_config=new_vectors_config
)
print("✅ New collection created successfully!")


# ==========================================
# 4. MIGRATE AND EMBED THE DATA
# ==========================================
print("Starting data migration...")
offset = None
total_processed = 0
failed_points = 0

while True:
    # Read from OLD collection
    records, offset = client.scroll(
        collection_name=OLD_COLLECTION,
        limit=50, 
        offset=offset,
        with_payload=True,
        with_vectors=True 
    )
    
    if not records:
        print(f"\n🎉 Migration Complete!")
        print(f"Successfully processed: {total_processed}")
        print(f"Failed embedding attempts: {failed_points}")
        break
    
    points_to_insert = []
    
    for record in records:
        # 1. Get and clean the technologies
        raw_tech = record.payload.get("technology", [])
        clean_tech_list = get_clean_tech_list(raw_tech)
        
        # 2. Join into a string for the Embedding Model
        text_for_embedding = ", ".join(clean_tech_list)
        
        # 3. Generate the new Jina v5 Vector
        tech_vector = get_jina_embedding(text_for_embedding)
        
        if tech_vector is None:
            # Fallback to zero-vector if the API fails for a specific point
            tech_vector = [0.0] * JINA_V5_SIZE
            failed_points += 1
            
        # 4. Clean up the payload
        new_payload = record.payload.copy()
        new_payload["technology"] = clean_tech_list 
        
        # 5. Merge vectors (Old vectors + New 'technology' vector)
        combined_vectors = record.vector.copy() 
        combined_vectors["technology"] = tech_vector
            
        # 6. Add to batch
        points_to_insert.append(
            models.PointStruct(
                id=record.id,
                payload=new_payload,
                vector=combined_vectors 
            )
        )
    
    # Write to NEW collection
    if points_to_insert:
        client.upsert(
            collection_name=NEW_COLLECTION,
            points=points_to_insert
        )
        
    total_processed += len(records)
    print(f"Progress: {total_processed} points moved...")
    
    # Slight delay to be kind to your Jina server
    time.sleep(0.1)
