import chromadb
import numpy as np

# Define dimensions and weights
weights = np.array([2.0, 1.0, 1.0]) # Chest, Waist, Hip
scaled_weights = np.sqrt(weights)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./garment_db") 
collection = client.get_or_create_collection(
    name="garment_measurements",
    metadata={"hnsw:space": "l2"} # Enforce L2 distance for the retrieval stage
)

if collection.count() < 1000000:
    # Generate Mock Data
    num_garments = 1000000
    raw_garments = np.random.uniform(80, 120, (num_garments, 3)).astype('float32')

    # Scale the vectors for insertion
    scaled_garments = raw_garments * scaled_weights

    # Prepare data for ChromaDB ingestion
    ids = [f"G_{i}" for i in range(num_garments)]
    embeddings = scaled_garments.tolist()

    # We store the RAW, unscaled measurements in metadata 
    # so we can use them for the asymmetric penalty in Stage 2
    metadatas = [
        {"chest": float(g[0]), "waist": float(g[1]), "hip": float(g[2])} 
        for g in raw_garments
    ]

    BATCH_SIZE = 5000

    # Insert into ChromaDB
    print("Ingesting garments into ChromaDB...")

    for i in range(0, num_garments, BATCH_SIZE):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE]
        )
    
print(f"Total garments indexed: {collection.count()}")

def query_chromadb_and_rerank(user_measurements, chroma_collection, top_k_candidates=500, final_k=3):
    # Format and scale the user query
    user_vector = np.array(user_measurements, dtype='float32')
    weighted_query = (user_vector * scaled_weights).tolist()
    
    # Broad Retrieval via ChromaDB
    # This fetches the closest matches using standard symmetric L2 distance
    results = chroma_collection.query(
        query_embeddings=[weighted_query],
        n_results=top_k_candidates
    )
    
    # Extract the payloads from Chroma's response structure
    candidate_ids = results['ids'][0]
    candidate_metadatas = results['metadatas'][0]
    
    # Precision Re-Ranking
    HARD_PENALTY = 15.0
    SOFT_PENALTY = 1.0
    
    best_fits = []
    
    for i in range(len(candidate_ids)):
        g_id = candidate_ids[i]
        meta = candidate_metadatas[i]
        
        # Reconstruct the raw garment vector from metadata
        garment_vector = np.array([meta['chest'], meta['waist'], meta['hip']])
        total_penalty = 0.0
        
        # Calculate asymmetric penalty
        deltas = garment_vector - user_vector
        
        for j, delta in enumerate(deltas):
            if delta < 0:
                # Too small
                total_penalty += abs(delta) * HARD_PENALTY * weights[j]
            else:
                # Too large
                total_penalty += delta * SOFT_PENALTY * weights[j]
                
        # Calculate 0-100 score
        confidence_score = max(0.0, 100.0 - total_penalty)
        
        best_fits.append({
            "id": g_id, 
            "score": round(confidence_score, 2), 
            "penalty": round(total_penalty, 2),
            "specs": meta
        })
        
    # Sort by lowest penalty (or highest score)
    best_fits.sort(key=lambda x: x['penalty'])
    
    # Return the exact Top 3
    return best_fits[:final_k]

# Sample Execution
user_input = [100.0, 85.0, 98.0] # Chest, Waist, Hip
top_garments = query_chromadb_and_rerank(user_input, collection)

for rank, garment in enumerate(top_garments):
    print(f"Rank {rank+1} | ID: {garment['id']} | Score: {garment['score']} | Specs: {garment['specs']}")