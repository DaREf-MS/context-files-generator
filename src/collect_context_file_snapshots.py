import os
import pandas as pd
import requests
import base64
import logging
import time
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "context_files_subset.csv"
OUTPUT_BASE_DIR = DATA_DIR / "repo_snapshot"
METADATA_OUTPUT_CSV = DATA_DIR / "context_files_commits_metadata.csv" 

# Progress file tracker path
PROGRESS_TRACKER_TXT = DATA_DIR / "processed_instances.txt"

GITHUB_TOKEN = os.getenv("TOKEN_2_JAPAN_PROJECT")
REQUEST_DELAY = 0.8 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("collection_progress.log"), logging.StreamHandler()]
)

def get_repo_full_name(github_link):
    path = urlparse(github_link).path
    parts = path.strip("/").split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None

def safe_request(url, headers, params=None):
    """Wraps requests.get with rate limit monitoring and automatic recovery."""
    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_time = response.headers.get("X-RateLimit-Reset")
            
            if remaining is not None and int(remaining) <= 3:
                if reset_time:
                    sleep_duration = max(int(reset_time) - time.time(), 0) + 5
                    logging.warning(f"Proactive Cooldown: Core limit low ({remaining}). Sleeping {sleep_duration:.1f}s...")
                    time.sleep(sleep_duration)
                    continue
            
            if response.status_code in [403, 429]:
                response_json = response.json() if response.text else {}
                error_msg = response_json.get("message", "").lower()
                
                if "rate limit" in error_msg or "secondary rate limit" in error_msg:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        sleep_duration = int(retry_after) + 5
                        logging.warning(f"Secondary Limit Hit. Sleeping {sleep_duration}s...")
                    elif reset_time:
                        sleep_duration = max(int(reset_time) - time.time(), 0) + 5
                        logging.warning(f"Primary Limit Hit. Sleeping {sleep_duration:.1f}s...")
                    else:
                        sleep_duration = 60
                        logging.warning(f"Rate limit hit. Backing off {sleep_duration}s...")
                        
                    time.sleep(sleep_duration)
                    continue
            
            return response

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}. Retrying in 10s...")
            time.sleep(10)

def extract_commit_metadata(commit_json, file_path, repo_name):
    """Safely extracts comprehensive metadata fields including additions, deletions, and patches."""
    commit_obj = commit_json.get('commit', {})
    author_profile = commit_json.get('author') or {}
    committer_profile = commit_json.get('committer') or {}
    
    meta = {
        "repo_name": repo_name,
        "context_file": file_path,
        "commit_hash": commit_json.get('sha'),
        "commit_message": commit_obj.get('message'),
        "git_author_name": commit_obj.get('author', {}).get('name'),
        "git_author_email": commit_obj.get('author', {}).get('email'),
        "git_author_date": commit_obj.get('author', {}).get('date'),
        "git_committer_name": commit_obj.get('committer', {}).get('name'),
        "git_committer_email": commit_obj.get('committer', {}).get('email'),
        "git_committer_date": commit_obj.get('committer', {}).get('date'),
        "github_author_user": author_profile.get('login'),
        "github_author_id": author_profile.get('id'),
        "github_author_type": author_profile.get('type'),
        "github_committer_user": committer_profile.get('login'),
        "github_committer_type": committer_profile.get('type'),
        "file_additions": None,
        "file_deletions": None,
        "file_patch": None
    }
    
    if 'files' in commit_json:
        for f in commit_json['files']:
            if f.get('filename') == file_path:
                meta["file_additions"] = f.get('additions')
                meta["file_deletions"] = f.get('deletions')
                meta["file_patch"] = f.get('patch')
                break
                
    return meta

def get_single_commit_details(repo_full_name, sha, file_path, repo_name):
    url = f"https://api.github.com/repos/{repo_full_name}/commits/{sha}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    response = safe_request(url, headers=headers)
    time.sleep(REQUEST_DELAY)
    
    if response and response.status_code == 200:
        return extract_commit_metadata(response.json(), file_path, repo_name)
    return None

def get_all_file_shas_up_to_date(repo_full_name, file_path, first_sha=None):
    """Fetches all commits for a file up to a specific date, handling API pagination."""
    url = f"https://api.github.com/repos/{repo_full_name}/commits"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    # Set the cutoff date specifically to June 11, 2026 (End of day UTC)
    cutoff_iso = "2026-06-11T23:59:59Z"
    
    params = {
        "path": file_path, 
        "until": cutoff_iso,
        "per_page": 100  # Maximize items per page to reduce API calls
    }
    
    shas = []
    
    while url:
        response = safe_request(url, headers=headers, params=params)
        time.sleep(REQUEST_DELAY)
        
        if response and response.status_code == 200:
            all_commits = response.json()
            for c in all_commits:
                sha = c['sha']
                shas.append(sha)
                
                # Stop if we reach the earliest known commit for this file
                if first_sha and sha == first_sha:
                    return shas
            
            # Handle Pagination: check if there's a next page of commits
            if 'next' in response.links:
                url = response.links['next']['url']
                params = None  # Clear params as they are embedded in the pagination URL
            else:
                url = None
        else:
            break
            
    return shas

def fetch_and_save_snapshot(repo_full_name, file_path, meta, repo_name):
    sha = meta["commit_hash"]
    timestamp = meta["git_committer_date"] or meta["git_author_date"] or "unknown_timestamp"
    
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    clean_ts = timestamp.replace(":", "-").replace("Z", "").replace("T", "_")
    target_dir = OUTPUT_BASE_DIR / repo_name.replace("/", "§") / file_path.replace("/", "§")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_save_path = target_dir / f"{clean_ts}.md"

    if file_save_path.exists():
        return

    response = safe_request(url, headers=headers, params={"ref": sha})
    time.sleep(REQUEST_DELAY)
    
    if response and response.status_code == 200:
        data = response.json()
        if data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode('utf-8', errors='replace')
            with open(file_save_path, "w", encoding="utf-8") as f:
                f.write(content)

def main():
    if not GITHUB_TOKEN:
        logging.error("Missing TOKEN_1_JAPAN_PROJECT environment variable.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Load progress tracking snapshot
    processed_instances = set()
    if PROGRESS_TRACKER_TXT.exists():
        with open(PROGRESS_TRACKER_TXT, "r", encoding="utf-8") as f:
            processed_instances = set(line.strip() for line in f if line.strip())
        logging.info(f"Resuming pipeline. Read {len(processed_instances)} previously completed instances.")

    # 2. Outer Progress Bar setup via tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Repository File Targets"):
        repo_full_name = get_repo_full_name(row['github_link'])
        file_path = row['context_file']
        repo_name = row['repo_name']
        
        # We optionally keep first_sha to bound how far back we go.
        first_sha = row.get('first_commit_sha') 

        if not repo_full_name:
            continue

        # Unique state tracking token key
        instance_key = f"{repo_name}||{file_path}"
        if instance_key in processed_instances:
            continue # Safe resume skip

        # Fetch all SHAs backward from the June 11, 2026 cutoff date
        shas_to_process = get_all_file_shas_up_to_date(repo_full_name, file_path, first_sha)

        instance_metadata_rows = []
        
        # 3. Process commit targets for the instance row
        for sha in shas_to_process:
            meta = get_single_commit_details(repo_full_name, sha, file_path, repo_name)
            if meta:
                fetch_and_save_snapshot(repo_full_name, file_path, meta, repo_name)
                instance_metadata_rows.append(meta)

        # 4. Save progress incrementally for this specific instance
        if instance_metadata_rows:
            instance_df = pd.DataFrame(instance_metadata_rows)
            file_exists = METADATA_OUTPUT_CSV.exists()
            
            METADATA_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            # Appends data row chunks dynamically while managing headers safely
            instance_df.to_csv(METADATA_OUTPUT_CSV, mode='a', index=False, header=not file_exists, encoding="utf-8")

        # Write successful state completion entry to tracker file
        PROGRESS_TRACKER_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_TRACKER_TXT, "a", encoding="utf-8") as f:
            f.write(f"{instance_key}\n")
            
        processed_instances.add(instance_key)

    logging.info("Pipeline tracking processing run finished successfully.")

if __name__ == "__main__":
    main()