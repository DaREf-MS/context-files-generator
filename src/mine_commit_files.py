import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'

# --- Configuration ---
INPUT_FILE = DATA_DIR / "context_files_commits_metadata.csv"
OUTPUT_FILE = DATA_DIR / "commit_files_results.csv"
PROGRESS_FILE = DATA_DIR / "progress.txt"

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Fetch token from environment variable for security
GITHUB_TOKEN = os.getenv("TOKEN_1_JAPAN_PROJECT")
if not GITHUB_TOKEN:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


def handle_rate_limit(response):
    """Checks GitHub API headers and pauses execution if the rate limit is reached."""
    if 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers['X-RateLimit-Remaining'])
        if remaining <= 1:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_duration = max(
                reset_time - int(time.time()), 0) + 5  # Add 5s buffer
            print(
                f"\n[!] Rate limit reached. Sleeping for {sleep_duration} seconds (until {datetime.fromtimestamp(reset_time).strftime('%H:%M:%S')})...")
            time.sleep(sleep_duration)


def load_progress():
    """Loads already processed (repo_name, commit_hash) pairs."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        # Format in progress file: repo_name,commit_hash
        return set(line.strip() for line in f if line.strip())


def append_to_output(results):
    """Appends a batch of results to the output CSV."""
    df = pd.DataFrame(results, columns=["repo_name", "commit_hash", "file_name", "status",
                      "additions", "deletions", "changes", "patch", "previous_filename", "raw_url"])
    write_header = not os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header)


def append_to_progress(repo_name, commit_hash):
    """Records a processed commit to the progress file."""
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{repo_name},{commit_hash}\n")


def fetch_commit_files(repo_name, commit_hash):
    """Fetches file changes for a specific commit."""
    url = f"https://api.github.com/repos/{repo_name}/commits/{commit_hash}"

    while True:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            handle_rate_limit(response)

            if response.status_code == 200:
                data = response.json()
                files_data = []
                for file_info in data.get('files', []):
                    files_data.append({
                        "repo_name": repo_name,
                        "commit_hash": commit_hash,
                        "file_name": file_info.get('filename'),
                        "status": file_info.get('status'),
                        "additions": file_info.get('additions', 0),
                        "deletions": file_info.get('deletions', 0),
                        "changes": file_info.get('changes', 0),
                        "patch": file_info.get('patch', ''),  # The actual diff
                        "previous_filename": file_info.get('previous_filename', ''),
                        "raw_url": file_info.get('raw_url', '')
                    })
                return files_data

            elif response.status_code == 404:
                # Commit or repo not found (e.g., force pushed, deleted, or private)
                return []
            elif response.status_code == 403 or response.status_code == 429:
                # Secondary rate limit hit
                retry_after = int(response.headers.get('Retry-After', 60))
                print(
                    f"\n[!] Secondary rate limit hit. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            else:
                print(
                    f"\n[!] Unexpected status {response.status_code} for {repo_name} {commit_hash}")
                return []

        except requests.exceptions.RequestException as e:
            print(
                f"\n[!] Network error for {repo_name} {commit_hash}: {e}. Retrying in 5 seconds...")
            time.sleep(5)


def main():
    # 1. Load input data
    print(f"Loading metadata from {INPUT_FILE}...")
    try:
        df_input = pd.read_csv(INPUT_FILE)
        df_input = df_input.drop_duplicates(
            subset=['repo_name', 'commit_hash'])
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Load progress
    processed = load_progress()
    print(f"Found {len(processed)} previously processed commits.")

    # 3. Process commits
    total_rows = len(df_input)

    with tqdm(total=total_rows, desc="Mining Commits", unit="commit") as pbar:
        for _, row in df_input.iterrows():
            repo_name = row['repo_name']
            commit_hash = row['commit_hash']
            progress_key = f"{repo_name},{commit_hash}"

            if progress_key in processed:
                pbar.update(1)
                continue

            # Fetch data
            file_results = fetch_commit_files(repo_name, commit_hash)

            # Save results and update progress if data was found
            if file_results:
                append_to_output(file_results)

            # Always mark as processed to avoid infinite loops on 404s
            append_to_progress(repo_name, commit_hash)
            processed.add(progress_key)

            pbar.update(1)

    print("\nData collection complete. Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
