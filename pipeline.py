from extraction import process_html_file, get_html_paths, DOCS_DIR
from chunking import chunk_sections_ray
# from openai_vector_store import build_pinecone_index
from miniLM_vector_store import build_pinecone_index
import ray
from dotenv import load_dotenv

def safe_process(item):
    try:
        result = process_html_file(item)
        return result
    except Exception as e:
        print(f"[ERROR] Failed to process {item}: {e}")
        return {"sections": []}

def run_pipeline():
    load_dotenv()

    # Initialize Ray
    try:
        ray.init(ignore_reinit_error=True) #, runtime_env={"env_vars": {"RAY_DEBUG": "1"},}
    except Exception as e:
        print(f"Error initializing Ray: {e}")

    html_files = get_html_paths(DOCS_DIR)
    # html_files = get_html_paths(DOCS_DIR)[:2]

    print(f"[INFO] Found {len(html_files)} HTML files.")
    if not html_files:
        print("[ERROR] No HTML files found. Check DOCS_DIR or file paths.")
        return
    ds = ray.data.from_items(html_files)
    sections_ds = ds.map(safe_process).flat_map(lambda x: x["sections"])
    print(f"[INFO] Sections count: {sections_ds.count()}")
    chunked_ds = chunk_sections_ray(sections_ds)
    print(f"[INFO] Chunk count: {chunked_ds.count()}")
    build_pinecone_index(chunked_ds, index_name="ray-rag-pinecone-index-minilm") # ray-rag-pinecone-index

if __name__ == "__main__":
    run_pipeline()
