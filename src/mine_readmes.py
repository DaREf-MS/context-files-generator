import os
import pandas as pd
import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from logging_config import get_logger

load_dotenv(override=True)


logger = get_logger(__name__)


# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

INPUT_FILE = DATA_DIR / "extracted_repo_metrics.csv"
OUTPUT_DIR = DATA_DIR / "readmes"
PROGRESS_FILE = DATA_DIR / "readme_download_progress.json"

# Retrieve the GitHub Token from the specified environment variable
GITHUB_TOKEN = os.getenv("TOKEN_1_JAPAN_PROJECT")

def load_progress():
    """Loads the set of already processed repositories from the progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()

def save_progress(processed_repos):
    """Saves the set of processed repositories to the progress file."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_repos), f)

def fetch_readme_content(repo, commit):
    """Fetches the README.md content from GitHub at a specific commit using Auth headers."""
    variations = ["README.md", "readme.md", "README.MD"]
    
    # Configure headers if the token is available
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        headers["Accept"] = "application/vnd.github.v3.raw"  # Best practice for GitHub API raw content
    
    for file_name in variations:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_name}"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                # Active backing off if explicitly rate-limited
                print("\n[Rate Limit] HTTP 429 detected. Sleeping for 30 seconds...")
                time.sleep(30)
                return fetch_readme_content(repo, commit)
            elif response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
                # Check if 403 is actually a rate limit exhaustion
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                    sleep_duration = max(reset_time - int(time.time()), 10)
                    print(f"\n[Rate Limit] GitHub secondary rate limit hit. Sleeping for {sleep_duration} seconds...")
                    time.sleep(sleep_duration)
                    return fetch_readme_content(repo, commit)
                    
        except requests.exceptions.RequestException as e:
            print(f"\n[Network Error] Connection issue for {repo}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            return fetch_readme_content(repo, commit)
            
    return ""

def main():
    # Verify token presence early to alert the user
    if not GITHUB_TOKEN:
        print("Warning: 'TOKEN_1_JAPAN_PROJECT' environment variable not found. Proceeding unauthenticated (strict rate limits apply).")
    else:
        print("Successfully loaded GitHub Token from 'TOKEN_1_JAPAN_PROJECT'.")

    # 1. Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(str(OUTPUT_DIR))
    
    # 2. Load the data
    print(f"Loading dataset from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return
        
    if 'repo' not in df.columns or 'resolved_commit' not in df.columns:
        print("Error: The input CSV must contain 'repo' and 'resolved_commit' columns.")
        return

    # Deduplicate based on repo and commit pairing
    df_unique = df.drop_duplicates(subset=['repo', 'resolved_commit'])
    
    # 3. Load progress state
    processed_repos = load_progress()
    df_to_process = df_unique[~df_unique['repo'].isin(processed_repos)]
    
    print(f"Total unique records: {len(df_unique)}")
    print(f"Already processed: {len(processed_repos)}")
    print(f"Remaining to process: {len(df_to_process)}")
    
    if df_to_process.empty:
        print("All repositories have been processed.")
        return

    # 4. Iterate and download with a progress bar
    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Downloading READMEs"):
        repo = row['repo']
        commit = row['resolved_commit']
        
        content = fetch_readme_content(repo, commit)
        
        logger.info(content)
        
        # Format filename replacing "/" with "§"
        safe_repo_name = str(repo).replace("/", "§")
        output_filepath = OUTPUT_DIR / f"{safe_repo_name}.md"
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        processed_repos.add(repo)
        save_progress(processed_repos)
        
        # Micro-sleep to keep standard API behavior smooth
        time.sleep(0.05)

if __name__ == "__main__":
    main()