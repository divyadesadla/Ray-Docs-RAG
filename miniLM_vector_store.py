from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import ray
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def init_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing Pinecone API key. Please set 'PINECONE_API_KEY' in environment variables.")
    pc = Pinecone(api_key=api_key)
    return pc

def create_pinecone_index(index_name="ray-rag-pinecone-index-minilm"):
    """Create Pinecone index or connect if it already exists."""
    pc = init_pinecone()
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric='euclidean',
            spec=spec,
            deletion_protection='disabled',
            tags={
                "environment": "development"
            }
        )
    return pc.Index(index_name)

# Initialize embeddings using SentenceTransformer with MiniLM model
embeddings_model = SentenceTransformer("all-MiniLM-L12-v2")

def embed_batch(batch: dict):
    # Extract 'text' and 'metadata' arrays
    try:
        texts = batch["text"]
    except KeyError as e:
        raise KeyError(f"Missing key: {e} - Make sure 'text' key exists in the batch.")

    try:
        metadatas = batch["metadata"]
    except KeyError as e:
        raise KeyError(f"Missing key: {e} - Make sure 'metadata' key exists in the batch.")

    # Check if lengths of 'texts' and 'metadatas' match
    if len(texts) != len(metadatas):
        raise ValueError(f"Mismatch in lengths: Texts ({len(texts)}) and Metadata ({len(metadatas)}) arrays have different lengths.")
    
    # Print to debug and inspect the batch
    print(f"Embedding batch with {len(texts)} items.")

    try:
        embeddings = embeddings_model.encode(texts)  # Get embeddings for the batch

    except Exception as e:
        raise Exception(f"Error generating embeddings: {e}")

    # Return the final structured list
    return {
        "results": [
            {
                "text": text,
                "metadata": {**metadata, "content": text},
                "embedding": embedding
            }
            for text, metadata, embedding in zip(texts, metadatas, embeddings)
        ]
    }

def truncate_metadata(metadata, max_size=1000):
    if "content" in metadata:
        content = metadata.pop("content")  # Temporarily remove it
    else:
        content = None

    while len(json.dumps(metadata)) > max_size:
        for key in metadata:
            metadata[key] = str(metadata[key])[:max(10, len(str(metadata[key])) - 10)]

    if content:
        metadata["content"] = content  # Reinsert after truncation
    return metadata

def store_embeddings_in_pinecone(embedded_data, pinecone_index, batch_size=100, max_batch_size=4194304):
    # If the data is a list instead of a dictionary, wrap it
    if isinstance(embedded_data, list):
        embedded_data = {"results": embedded_data}

    # Ensure embedded_data is wrapped in a dictionary with key 'results'
    if "results" not in embedded_data:
        raise KeyError("Expected key 'results' in the embedded data.")

    # Extract the actual list of embeddings
    embedded_data = embedded_data["results"]

    # Flatten the nested structure if items are nested under "results"
    embedded_data = [item["results"] if "results" in item else item for item in embedded_data]

    ids = [str(uuid.uuid4()) for _ in embedded_data]
    
    vectors = [
        item.get("embedding").tolist()  # Convert numpy array to list
        for item in embedded_data
        if "embedding" in item
    ]
    
    metadatas = [
        truncate_metadata(item.get("metadata", {}))
        for item in embedded_data
    ]

    # Check lengths
    if len(ids) != len(vectors) or len(vectors) != len(metadatas):
        raise ValueError(f"Mismatch in lengths: ids ({len(ids)}), vectors ({len(vectors)}), metadatas ({len(metadatas)})")

    print(f"Upserting {len(ids)} vectors to Pinecone in batches of {batch_size}")

    try:
        # Split the data into batches and upsert each batch separately
        batch_start = 0
        while batch_start < len(ids):
            # Slice out the current batch
            batch_ids = ids[batch_start:batch_start + batch_size]
            batch_vectors = vectors[batch_start:batch_start + batch_size]
            batch_metadatas = metadatas[batch_start:batch_start + batch_size]

            # Check if the batch exceeds the max allowed size
            batch_size_bytes = sum([len(str(item)) for item in batch_vectors])  # Estimate size in bytes
            if batch_size_bytes > max_batch_size:
                # If the batch exceeds the limit, we store only as much as we can
                print(f"Warning: Batch size exceeds the maximum allowed size. Storing as much as possible in batch {batch_start // batch_size + 1}.")
                end_idx = batch_start + batch_size
                while end_idx > batch_start and batch_size_bytes > max_batch_size:
                    end_idx -= 1
                    batch_vectors = batch_vectors[:end_idx - batch_start]
                    batch_metadatas = batch_metadatas[:end_idx - batch_start]
                    batch_ids = batch_ids[:end_idx - batch_start]
                    batch_size_bytes = sum([len(str(item)) for item in batch_vectors])
                pinecone_index.upsert(vectors=[
                    (id_, vector, metadata)
                    for id_, vector, metadata in zip(batch_ids, batch_vectors, batch_metadatas)
                ])
                print(f"Batch {batch_start // batch_size + 1} upserted with partial data.")
            else:
                # If it's under the max size, we can safely upsert the entire batch
                pinecone_index.upsert(vectors=[
                    (id_, vector, metadata)
                    for id_, vector, metadata in zip(batch_ids, batch_vectors, batch_metadatas)
                ])
                print(f"Batch {batch_start // batch_size + 1} upserted successfully.")

            # Move to the next batch
            batch_start += batch_size
            
    except Exception as e:
        raise Exception(f"Error while upserting to Pinecone: {e}")


def build_pinecone_index(chunked_ds: ray.data.Dataset, index_name="ray-rag-pinecone-index-minilm"):
    init_pinecone()
    pinecone_index = create_pinecone_index(index_name)

    chunked_ds = chunked_ds.map(lambda item: {
        "text": item["text"],
        "metadata": item.get("metadata", {})
    })

    embedded_ds = chunked_ds.map_batches(embed_batch, batch_size=32, batch_format="default")
    embedded_data = embedded_ds.take_all()
    # embedded_data = embedded_ds.take(1)  # Only take one to inspect

    store_embeddings_in_pinecone(embedded_data, pinecone_index)