from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import torch
import json

# Cache models and tokenizers
MODEL_CACHE = {}
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L12-v2")

# Load environment variables
load_dotenv()

def init_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing Pinecone API key. Please set 'PINECONE_API_KEY' in environment variables.")
    pc = Pinecone(api_key=api_key)
    return pc

def load_model(model_name):
    """Load and cache model and tokenizer."""
    if model_name not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        MODEL_CACHE[model_name] = (tokenizer, model)
    return MODEL_CACHE[model_name]

def get_retrieved_context(index_name, query_text, top_k=5):
    """Retrieve and concatenate top-k relevant snippets from Pinecone."""
    try:
        pc = init_pinecone()
        index = pc.Index(index_name)

        query_vector = EMBEDDING_MODEL.encode([query_text])[0].tolist()

        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        if not query_response['matches']:
            print("No results found in Pinecone for the given query.")
            return "", []

        snippets = [
            match["metadata"].get("content", "No content found")
            for match in query_response["matches"]
        ]

        context = "\n".join(snippets)
        return context, snippets

    except Exception as e:
        raise Exception(f"Error retrieving context from Pinecone: {e}")

def generate_with_model(model_name, query, context):
    """Generate an answer using the specified Hugging Face model (with context)."""
    try:
        tokenizer, model = load_model(model_name)
        input_text = f"Question: {query} Answer based on the context: {context}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        output = model.generate(
            **inputs,
            max_new_tokens=200,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Error generating with {model_name}: {e}"

def generate_llm_response(model_name, query):
    """Generate an LLM-only response (without context)."""
    try:
        tokenizer, model = load_model(model_name)
        input_text = f"Question: {query}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        output = model.generate(
            **inputs,
            max_new_tokens=200,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Error generating LLM-only response with {model_name}: {e}"

def save_results(query, context, combined_outputs, filename="evaluation_data.json"):
    """Save query, context, and generated responses to a JSON file."""
    eval_data = {
        "query": query,
        "context": context,
        "responses": combined_outputs
    }

    with open(filename, "w") as f:
        json.dump(eval_data, f, indent=4)

if __name__ == "__main__":
    query_text = "What is Ray Data?"
    index_name = "ray-rag-pinecone-index-minilm"

    try:
        context, snippets = get_retrieved_context(index_name, query_text)
        if not context:
            print("No relevant content found.")
            exit()

        print(f"\n📚 Top-{len(snippets)} retrieved snippets:")
        for i, snippet in enumerate(snippets, 1):
            print(f"[{i}] {snippet[:300]}...")  # First 300 chars

        # Models to test
        model_list = [
            "t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "facebook/bart-large-cnn"
        ]

        combined_outputs = []

        for model_name in model_list:
            print(f"\n🧠 Model: {model_name}")

            # Generate response with context
            rag_response = generate_with_model(model_name, query_text, context)
            print(f"📝 LLM+RAG Response:\n{rag_response}\n")

            # Generate LLM-only response (without context)
            llm_response = generate_llm_response(model_name, query_text)
            print(f"💬 LLM-only Response:\n{llm_response}\n")

            combined_outputs.append({
                "model": model_name,
                "llm_response": llm_response,
                "llm+rag_response": rag_response
            })

        # Save to file
        save_results(query_text, context, combined_outputs)

    except Exception as e:
        print(f"❌ Error: {e}")
