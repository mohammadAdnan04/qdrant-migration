import re
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from concurrent.futures import ThreadPoolExecutor  # <--- Added for speed

# ==========================================
# 1. CONFIGURATION
# ==========================================
QDRANT_URL = "http://185.182.187.174:6333/"
QDRANT_API_KEY = "16a3a57d"
OLD_COLLECTION = "companies_jina_final"
NEW_COLLECTION = "companies_jina_final_v2"
JINA_API_URL = "https://python-embedding.34.166.92.24.sslip.io/embed"

# Set how many requests to send at once (don't go too high or you'll crash the API)
MAX_WORKERS = 10 

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_clean_tech_list(raw_tech_list):
    if not raw_tech_list: return []
    full_string = str(raw_tech_list)
    tech_keywords = re.findall(r"'([^']+)': '([^']+)'", full_string)
    clean_technologies = [v.strip() for k, v in tech_keywords if v and v not in ["COMPLETED", "U.S. Server Location"]]
    if isinstance(raw_tech_list, list):
        for item in raw_tech_list:
            if isinstance(item, str) and ":" not in item and "{" not in item:
                clean_technologies.append(item.strip())
    return list(set(clean_technologies))

def get_jina_embedding(text):
    """Fetches a single embedding from the API."""
    if not text or text.strip() == "": return None 
    try:
        response = requests.post(JINA_API_URL, json={"text": text}, timeout=60)
        response.raise_for_status() 
        return response.json().get('embedding')
    except Exception as e:
        return None

def process_single_record(record, jina_size):
    """Worker function to process one record in the thread pool."""
    raw_tech = record.payload.get("technology", [])
    clean_tech_list = get_clean_tech_list(raw_tech)
    text_for_embedding = ", ".join(clean_tech_list)
    
    tech_vector = get_jina_embedding(text_for_embedding)
    if tech_vector is None:
        tech_vector = [0.0] * jina_size
        
    new_payload = record.payload.copy()
    new_payload["technology"] = clean_tech_list 
    
    combined_vectors = record.vector.copy() 
    combined_vectors["technology"] = tech_vector
        
    return models.PointStruct(
        id=record.id,
        payload=new_payload,
        vector=combined_vectors 
    )

# ==========================================
# 3. PREPARE COLLECTION
# ==========================================
print("Fetching schema and testing API...")
old_info = client.get_collection(OLD_COLLECTION)
test_vec = get_jina_embedding("test")
if not test_vec:
    raise Exception("API Connection Failed")
JINA_V5_SIZE = len(test_vec)

new_vectors_config = {name: models.VectorParams(size=p.size, distance=p.distance) 
                     for name, p in old_info.config.params.vectors.items()}
new_vectors_config["technology"] = models.VectorParams(size=JINA_V5_SIZE, distance=models.Distance.COSINE)

client.recreate_collection(collection_name=NEW_COLLECTION, vectors_config=new_vectors_config)
print("✅ New collection ready.")

# ==========================================
# 4. MIGRATE (SPEED VERSION)
# ==========================================
print(f"Starting fast migration with {MAX_WORKERS} parallel threads...")
offset = None
total_processed = 0

while True:
    records, offset = client.scroll(
        collection_name=OLD_COLLECTION,
        limit=100,  # Increased batch size to 100
        offset=offset,
        with_payload=True,
        with_vectors=True 
    )
    
    if not records:
        print(f"\n🎉 Migration Complete! Total: {total_processed}")
        break
    
    # --- The Parallel Part ---
    # We send 100 records into the 'thread pool' to be processed simultaneously
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        points_to_insert = list(executor.map(lambda r: process_single_record(r, JINA_V5_SIZE), records))
    
    # Bulk upload the result to Qdrant
    if points_to_insert:
        client.upsert(collection_name=NEW_COLLECTION, points=points_to_insert)
        
    total_processed += len(records)
    print(f"Progress: {total_processed} points moved...")

    # Removed time.sleep for maximum speed
