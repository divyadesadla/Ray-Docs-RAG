"""
This Python script uses the wget command (a Linux tool) to download a full copy of the Ray documentation website and save it to a directory (like /efs/ray_docs).

Folder Structure in File System: When wget scrapes a website and preserves the domain structure, it creates directories that represent the structure of the URLs. 
Since docs.ray.io is part of the URL (https://docs.ray.io/), it creates a folder with that name in the destination path (/efs/ray_docs/docs.ray.io).

The Folder Contains Files: Inside the docs.ray.io folder, wget saves all the scraped HTML files, images, stylesheets, and other resources that make up the web pages.
For example:
/efs/ray_docs/docs.ray.io/en/master/ray-core/starting-ray.html
/efs/ray_docs/docs.ray.io/en/master/ray-core/tips-for-first-time.html

# Current code might not be able to scrape all data, does: Converted links in 3652 files in 144 seconds
Saves the data in /efs/ray_docs folder.
"""
import subprocess
import os

def scrape_ray_docs(output_dir="/efs/ray_docs"):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the wget command
    cmd = [
        "wget",
        "-e", "robots=off",
        "--recursive",
        "--no-clobber",
        "--page-requisites",
        "--html-extension",
        "--convert-links",
        "--restrict-file-names=windows",
        "--domains", "docs.ray.io",
        "--no-parent",
        "--accept=html",
        "-P", output_dir,
        "https://docs.ray.io/en/master/"
    ]

    print(f"Starting wget to scrape Ray docs into: {output_dir}\n")
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Scraping completed successfully!")
    except subprocess.CalledProcessError as e:
        print("\n❌ Scraping failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    scrape_ray_docs()
