import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# Initialize necessary components
embeddings_model = SentenceTransformer("all-MiniLM-L12-v2")
rouge = Rouge()

# Directory to save reports
REPORT_DIR = "evaluation_reports"
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# Load the saved evaluation data (query, context, and responses)
with open("evaluation_data.json", "r") as f:
    eval_data = json.load(f)

def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using sentence embeddings.
    """
    embeddings = embeddings_model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def compute_bleu(reference, candidate):
    """
    Compute BLEU score for a response.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return sentence_bleu([reference_tokens], candidate_tokens)

def compute_rouge(reference, candidate):
    """
    Compute ROUGE scores for a response.
    """
    rouge_scores = rouge.get_scores(candidate, reference)[0]
    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"]
    }

def evaluate_response(response, context, query):
    """
    Evaluate a response on multiple metrics:
        - Relevance: Similarity between the response and the query.
        - Faithfulness: Similarity between the response and the context.
        - Completeness: Basic proxy metric comparing response length to context.
        - Fluency: Simple rule-based check on response fluency.
        - BLEU: BLEU score between response and reference.
        - ROUGE: ROUGE-1, ROUGE-2, ROUGE-L.
        - Embedding Similarity: Cosine similarity between response and context/query.
    """
    metrics = {}

    # 1. Relevance
    relevance = compute_similarity(response, query)
    metrics["relevance"] = round(relevance, 4)

    # 2. Faithfulness
    faithfulness = compute_similarity(response, context) if context else 0.0
    metrics["faithfulness"] = round(faithfulness, 4)

    # 3. Completeness (basic proxy)
    metrics["completeness"] = round(min(len(response) / max(len(context), 1), 1.0), 4)

    # 4. Fluency (rule-based)
    metrics["fluency"] = round((response.endswith(".") or response.endswith("!")) * (len(response.split()) > 5), 2)

    # 5. BLEU and 6. ROUGE
    if context:
        bleu = compute_bleu(context, response)
        metrics["BLEU"] = round(bleu, 4)

        rouge_scores = compute_rouge(context, response)
        metrics["ROUGE-1"] = round(rouge_scores["rouge-1"], 4)
        metrics["ROUGE-2"] = round(rouge_scores["rouge-2"], 4)
        metrics["ROUGE-L"] = round(rouge_scores["rouge-l"], 4)
    else:
        metrics["BLEU"] = None
        metrics["ROUGE-1"] = None
        metrics["ROUGE-2"] = None
        metrics["ROUGE-L"] = None

    # 7. Embedding Similarity
    embedding_similarity = compute_similarity(response, context) if context else 0.0
    metrics["Embedding_Similarity"] = round(embedding_similarity, 4)

    return metrics

def generate_llm_response(model, query):
    """
    Mock function to simulate LLM-only response generation. Replace with actual model interaction.
    """
    # Replace with actual logic for LLM generation based on the query.
    return f"Simulated response for LLM model {model} to the query: {query}"

def evaluate_rag_vs_llm(query, context, model_responses):
    """
    Evaluate both RAG and LLM model responses for a given query and context.
    Now uses saved LLM-only responses instead of generating mock ones.
    """
    evals = {}

    for model_data in model_responses:
        model = model_data["model"]
        rag_response = model_data["llm+rag_response"]
        llm_response = model_data["llm_response"]

        # Evaluate RAG response (uses context)
        evals[f"{model} - RAG"] = {
            "response": rag_response,
            "metrics": evaluate_response(rag_response, context, query)
        }

        # Evaluate LLM-only response (does not use context)
        evals[f"{model} - LLM (no context)"] = {
            "response": llm_response,
            "metrics": evaluate_response(llm_response, "", query)
        }

    return evals


def save_eval_report(report_data, query):
    """
    Save the evaluation metrics to a JSON file with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{timestamp}.json"
    path = os.path.join(REPORT_DIR, filename)

    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy types to native Python types
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    full_report = {
        "query": query,
        "evaluations": convert_numpy_types(report_data)
    }

    with open(path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n✅ Evaluation saved to {path}")
    return path

if __name__ == "__main__":
    # Initialize evaluation results
    evaluation_results = {}

    # Load the saved evaluation data
    query = eval_data["query"]
    context = eval_data["context"]
    model_responses = eval_data["responses"]

    # Evaluate
    evals = evaluate_rag_vs_llm(query, context, model_responses)

    # Collect metrics for leaderboard
    for model_name, eval_data in evals.items():
        evaluation_results[model_name] = eval_data["metrics"]

    # Leaderboard based on a chosen metric (Embedding Similarity)
    leaderboard = sorted(evaluation_results.items(), key=lambda item: item[1]["Embedding_Similarity"], reverse=True)

    # Save the evaluation report
    save_eval_report(evaluation_results, query)

    # Print the leaderboard
    print("\nLeaderboard (based on Embedding Similarity):")
    for idx, (model_name, metrics) in enumerate(leaderboard, start=1):
        print(f"{idx}. Model: {model_name}")
        for metric_name, score in metrics.items():
            print(f"   {metric_name}: {score}")
        print("\n")
