import json
import csv
import random
import re
from pathlib import Path

# File paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
INPUT_FILE = DATA_DIR / "assignment_corrected_sample.jsonl"
METADATA_FILE = DATA_DIR / "context_files_commits_metadata.csv"
OUTPUT_FILE = DATA_DIR / "acf_random_sample.csv"
SAMPLE_SIZE = 100

def extract_topics(responses_text):
    """
    Extracts the topic names from the 'responses' text.
    Matches lines like '[1] Topic Name: Explanation' and captures 'Topic Name'.
    """
    if not responses_text:
        return ""
    
    topics = re.findall(r'\[\d+\]\s*(.*?):', responses_text)
    return "&SEP&".join(topics)

def normalize_date(date_str):
    """
    Converts both date formats into a standard 'YYYY-MM-DDTHH:MM:SS' format.
    - JSONL format: '2025-12-17_16-23-18.md'
    - CSV format:   '2026-04-28T01:05:52Z'
    """
    date_str = date_str.strip()
    
    # Handle JSONL version format
    if date_str.endswith('.md'):
        date_str = date_str[:-3]  # Remove '.md'
        if '_' in date_str:
            date_part, time_part = date_str.split('_', 1)
            time_part = time_part.replace('-', ':')  # Convert HH-MM-SS to HH:MM:SS
            return f"{date_part}T{time_part}"
            
    # Handle CSV git_committer_date format
    elif date_str.endswith('Z') and 'T' in date_str:
        return date_str[:-1]  # Remove trailing 'Z'
        
    return date_str

def normalize_key(repo, file, version_or_date):
    """
    Normalizes the join keys to ensure a match between the JSONL and CSV.
    - Standardizes separators (replacing '/' with '§')
    - Standardizes the timestamp formatting.
    """
    norm_repo = repo.replace('/', '§')
    norm_file = file.replace('/', '§')
    norm_date = normalize_date(version_or_date)
    return (norm_repo, norm_file, norm_date)

def load_metadata():
    """
    Loads the metadata CSV into a dictionary for fast joining.
    Key: (repo_name, context_file, git_committer_date)
    Value: commit_hash
    """
    metadata = {}
    print(f"Loading metadata from {METADATA_FILE}...")
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = normalize_key(row['repo_name'], row['context_file'], row['git_committer_date'])
                metadata[key] = row['commit_hash']
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {METADATA_FILE}. Join will produce empty hashes.")
    
    return metadata

def main():
    # 1. Load the metadata for the join
    metadata = load_metadata()

    # 2. Read the JSONL file, group by (repo, file), and exclude the first version
    repo_file_dict = {}
    print(f"Reading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                repo = data.get("repo")
                file_name = data.get("file")
                
                if repo and file_name:
                    key = (repo, file_name)
                    if key not in repo_file_dict:
                        repo_file_dict[key] = []
                    repo_file_dict[key].append(data)
    except FileNotFoundError:
        print(f"Error: The file {INPUT_FILE} was not found.")
        return

    # Sort each file's versions chronologically, drop the first, and build repos_dict
    repos_dict = {}
    for (repo, file_name), entries in repo_file_dict.items():
        # Sort using the normalize_date function to ensure correct chronological order
        entries.sort(key=lambda x: normalize_date(x.get("version", "")))
        
        # Skip the first (oldest) version, keep the rest
        subsequent_versions = entries[1:]
        
        if subsequent_versions:
            if repo not in repos_dict:
                repos_dict[repo] = []
            repos_dict[repo].extend(subsequent_versions)

    unique_repos = list(repos_dict.keys())
    print(f"Found {len(unique_repos)} unique repositories.")

    # 3. Randomly sample 100 different repositories
    if len(unique_repos) < SAMPLE_SIZE:
        print(f"Warning: Only found {len(unique_repos)} repositories, which is less than the requested {SAMPLE_SIZE}. Sampling all available.")
        sampled_repos = unique_repos
    else:
        sampled_repos = random.sample(unique_repos, SAMPLE_SIZE)

    # 4. Pick one random file from each sampled repository and perform the join
    sampled_data = []
    for repo in sampled_repos:
        chosen_entry = random.choice(repos_dict[repo])
        
        repo_name = chosen_entry.get("repo", "")
        file_name = chosen_entry.get("file", "")
        version = chosen_entry.get("version", "")
        responses = chosen_entry.get("responses", "")
        
        topics = extract_topics(responses)
        
        # Perform the join using the normalized key
        join_key = normalize_key(repo_name, file_name, version)
        commit_hash = metadata.get(join_key, "")
        
        # Construct GitHub URL 
        # (Converting 'Owner§Repo' back to 'Owner/Repo' for a valid URL)
        github_url = ""
        if commit_hash:
            standard_repo_name = repo_name.replace('§', '/')
            github_url = f"https://github.com/{standard_repo_name}/commit/{commit_hash}"
        
        sampled_data.append([repo_name, file_name, version, topics, commit_hash, github_url])

    # 5. Write to CSV
    print(f"Writing {len(sampled_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['repo', 'file', 'version', 'topics', 'commit_hash', 'github_url'])
        # Write rows
        writer.writerows(sampled_data)
        
    print("Done!")

if __name__ == "__main__":
    main()