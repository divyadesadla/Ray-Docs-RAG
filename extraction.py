import os
import ray
from pathlib import Path
from bs4 import BeautifulSoup
import ray

#Facebook AI Similarity Search (Stores and indexes high-dimensional vectors to efficiently perform similarity searches.)
from langchain_community.vectorstores import FAISS

"""
After applying the flat map, can print to see 5 samples:
for section in flat_sections.materialize().take(5):  # Take and print the first 5 sections
    print(section)
"""

# Directory containing the HTML files
DOCS_DIR = Path("/efs/ray_docs/docs.ray.io/en/master/")

def get_html_paths(directory: Path):
    """Return a list of dictionaries with HTML file paths under the given directory."""
    return [{"path": path} for path in directory.rglob("*.html") if path.is_file()]

def read_html(file_path: Path) -> str:
    """Read and return the content of an HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return ""

def extract_sections_from_html(html_content: str, url: str):
    """
    Extract content sections from HTML using BeautifulSoup.
    Each section is returned as a dictionary with text, section ID, and source URL.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    extracted_data = []

    # Search for any tag with an 'id' attribute — typically sections or divs
    for section in soup.find_all(["section", "div"], id=True):
        section_id = section.get("id")
        section_text = " ".join(p.get_text(strip=True) for p in section.find_all("p"))
        
        if section_text:  # Avoid empty entries
            extracted_data.append({
                "text": section_text,
                "url": url,
                "section_id": section_id
            })

    return extracted_data

def process_html_file(file_info: dict):
    """
    Process a single HTML file dictionary to extract meaningful content sections.
    """
    file_path = file_info["path"]
    rel_path = file_path.relative_to(DOCS_DIR)
    url = f"https://docs.ray.io/en/master/ray-core/{rel_path}"

    html_content = read_html(file_path)
    if not html_content:
        return {'sections': []}  # Wrap the result in a dictionary with a 'sections' key
    
    extracted_data = extract_sections_from_html(html_content, url)
    return {'sections': extracted_data}  # Wrap the result in a dictionary with a 'sections' key






