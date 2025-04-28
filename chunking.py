from langchain.text_splitter import RecursiveCharacterTextSplitter
import ray
import time

def chunk_section(section: dict):

    print(f"[chunk_section] Start processing section_id={section.get('section_id')}, url={section.get('url')}")

    start = time.time()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    try:
        text = section.get("text", "")
        if not text:
            print(f"[chunk_section] Empty text for section_id={section.get('section_id')}")
            return []

        chunks = text_splitter.split_text(text)

        print(f"[chunk_section] Finished splitting section_id={section.get('section_id')} into {len(chunks)} chunks in {time.time() - start:.2f}s")

        return [{
            "text": chunk,
            "metadata": {
                "url": section["url"],
                "section_id": section["section_id"],
                "chunk_id": i
            }
        } for i, chunk in enumerate(chunks)]

    except Exception as e:
        print(f"[chunk_section] ERROR section_id={section.get('section_id')}: {e}")
        return []

def chunk_sections_ray(sections_ds: ray.data.Dataset) -> ray.data.Dataset:
    """
    Chunk sections using Ray Dataset's flat_map for parallel scalability.
    Each output item is a dictionary with chunked text and associated metadata.
    """
    print("[chunk_sections_ray] About to chunk sections with Ray flat_map")
    return sections_ds.flat_map(chunk_section)