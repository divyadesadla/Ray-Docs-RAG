from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
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

def create_pinecone_index(index_name="ray-rag-pinecone-index"):
    """Create Pinecone index or connect if it already exists."""
    pc = init_pinecone()
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=spec,
            deletion_protection='disabled',
            tags={
                "environment": "development"
            }
        )
    return pc.Index(index_name)

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

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

    try:
        embeddings = embeddings_model.embed_documents(texts)
    except Exception as e:
        raise Exception(f"Error generating embeddings: {e}")

    # Return the final structured list
    return [
        {
            "text": text,
            "metadata": metadata,
            "embedding": embedding
        }
        for text, metadata, embedding in zip(texts, metadatas, embeddings)
    ]

def truncate_metadata(metadata, max_size=1000):
    while len(json.dumps(metadata)) > max_size:
        for key in metadata:
            metadata[key] = str(metadata[key])[:max(10, len(str(metadata[key])) - 10)]
    return metadata

def store_embeddings_in_pinecone(embedded_data, pinecone_index):
    ids = [str(uuid.uuid4()) for _ in embedded_data]
    vectors = [item["embedding"] for item in embedded_data]
    metadatas = [truncate_metadata(item["metadata"]) for item in embedded_data]

    pinecone_index.upsert(vectors=[
        (id_, vector, metadata)
        for id_, vector, metadata in zip(ids, vectors, metadatas)
    ])

def build_pinecone_index(chunked_ds: ray.data.Dataset, index_name="ray-rag-pinecone-index"):
    init_pinecone()
    pinecone_index = create_pinecone_index(index_name)

    chunked_ds = chunked_ds.map(lambda item: {
        "text": item["text"],
        "metadata": item.get("metadata", {})
    })

    embedded_ds = chunked_ds.map_batches(embed_batch, batch_size=32, batch_format="default")
    embedded_data = embedded_ds.take_all()

    store_embeddings_in_pinecone(embedded_data, pinecone_index)
