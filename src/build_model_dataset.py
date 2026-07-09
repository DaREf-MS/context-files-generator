import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
README_DIR = ROOT_DIR / "readmes"

# Input Files
REPO_METRICS_FILE = DATA_DIR / "pre_acf_model_features.csv"
RAW_PRS_FILE = DATA_DIR / "prs_features.csv"
ADVANCED_REPO_METRICS_FILE = DATA_DIR / "extracted_repo_metrics.csv"
TOPICS_JSONL_FILE = DATA_DIR / "assignment_corrected_sample.jsonl"
REFINEMENT_FILE = DATA_DIR / "refinement.md"


# Output Files
OUTPUT_DATASET_FILE = DATA_DIR / "final_model_dataset.parquet"
OUTPUT_CSV_FILE = DATA_DIR / "final_model_dataset.csv"

SEPARATOR = '&SEP&'

def read_refinement_topics():
    topics = []
    with open(REFINEMENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                group_pattern = r"\]\s(.*?)\s\(Count"
                match = re.search(group_pattern, line)
                if match:
                    topic = match.group(1)
                    topics.append(topic)
    return topics

def extract_latest_acf_topics():
    """
    Parses the JSONL to extract the target variable: the topics present in the 
    LAST version of the ACF file, unioned across all ACFs for a given repository.
    """
    print("Extracting dependent variables (topics) from the latest ACFs...")
    data = []
    
    refinement_topics = read_refinement_topics()
    
    with open(TOPICS_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_id = row.get("id", "")
            raw_responses = row.get("responses", "")
            
            if SEPARATOR in raw_id:
                repo_part, file_part = raw_id.split(SEPARATOR, 1)
                repo_name = repo_part.replace("§", "/")
                context_file = file_part.split(SEPARATOR)[0]
                timestamp_str = file_part.split(SEPARATOR)[1].replace(".md", "").replace('_', ' ')
            else:
                continue
                
            topics = re.findall(r"\[\d+\]\s*(.*?):", raw_responses)
            cleaned_topics = [str(t).strip() for t in topics if t if t.strip() in refinement_topics]
            
            data.append({
                "repo": repo_name,
                "context_file": context_file,
                # Explicit format to avoid timezone offset parsing errors
                "timestamp": pd.to_datetime(timestamp_str, format="%Y-%m-%d %H-%M-%S", errors="coerce", utc=True),
                "topics": cleaned_topics
            })
            
    df_topics = pd.DataFrame(data)
    if df_topics.empty:
        return pd.DataFrame(columns=['repo', 'final_topics'])

    # Sort by time, keep the absolute latest version per context_file
    df_topics = df_topics.sort_values(by=["repo", "context_file", "timestamp"])
    latest_versions = df_topics.drop_duplicates(subset=["repo", "context_file"], keep="last")
    
    # Union all topics across different latest context files for a single repo
    repo_topics = latest_versions.groupby("repo")["topics"].agg(
        lambda x: list(set(topic for sublist in x for topic in sublist))
    ).reset_index()
    
    repo_topics.rename(columns={"topics": "final_topics"}, inplace=True)
    return repo_topics


def load_readmes_to_dataframe(readme_dir):
    """
    Reads all README files from the specified directory and loads them into a DataFrame.
    """
    data = []
    
    # Verify the directory exists
    if not readme_dir.exists():
        print(f"Error: Directory '{readme_dir}' not found.")
        return pd.DataFrame(columns=['repo', 'readme_content'])

    print(f"Reading files from {readme_dir}...")
    
    # Iterate through all files in the directory
    for file_path in readme_dir.iterdir():
        if file_path.is_file():
            # Reconstruct the original repository name
            repo_name = file_path.name.replace("§", "/")
            
            # Read the file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {file_path.name} - {e}")
                content = ""
                
            data.append({
                "repo": repo_name,
                "readme_content": content
            })

    # Create and return the DataFrame
    df = pd.DataFrame(data)
    return df


def aggregate_pr_metrics(df_raw_prs, df_repo):
    """
    Filters raw PRs/commits strictly before the first ACF date and computes averages.
    """
    print("Calculating historical PR/commit averages strictly before the first ACF...")
    
    # Standardize dates for accurate filtering
    date_col = 'created_at' if 'created_at' in df_raw_prs.columns else 'date'
    df_raw_prs[date_col] = pd.to_datetime(df_raw_prs[date_col], utc=True, errors='coerce')
    df_repo['first_acf_date'] = pd.to_datetime(df_repo['first_acf_date'], format="mixed", utc=True, errors='coerce')
    
    print(f"HIHIHI: {df_raw_prs.shape[0]} PR records.")

    # Merge to align PRs with their repo's cutoff date
    df_merged = pd.merge(df_raw_prs, df_repo[['repo', 'first_acf_date']], on='repo', how='inner')
    
    
    # STRICT FILTER: Keep only records created BEFORE the first ACF
    df_filtered = df_merged[df_merged[date_col] < df_merged['first_acf_date']].copy()
    
    if df_filtered.empty:
        print("Warning: No PRs found before the ACF dates.")
        return pd.DataFrame(columns=['repo'])

    # Define the exact metrics to average
    target_metrics = [
        'merge_time',
        'closure_time',
        'num_files',
        'code_churn',
        'num_commits',
        'comments',
        'description_length'
    ]

    # Ensure columns exist and are numeric
    existing_metrics = [col for col in target_metrics if col in df_filtered.columns]
    for col in existing_metrics:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Group by repo and compute the mean
    df_pr_averages = df_filtered.groupby('repo')[existing_metrics].mean().reset_index()

    # Rename columns to clearly indicate they are pre-ACF averages
    rename_dict = {col: f"avg_{col}_before_acf" for col in existing_metrics}
    df_pr_averages.rename(columns=rename_dict, inplace=True)

    return df_pr_averages


def generate_readme_embeddings(df, text_column='readme_content'):
    """
    Generates embedding vectors for the project readme contents.
    """
    print(f"Generating sentence embeddings for `{text_column}`...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    descriptions = df[text_column].fillna("").astype(str).tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    df['readme_embedding'] = list(embeddings)
    return df


def main():
    # 1. Load Repo Metrics (Features collected prior to ACF)
    try:
        df_repo = pd.read_csv(REPO_METRICS_FILE)
        readmes_df = load_readmes_to_dataframe()
        df_repo = df_repo.merge(readmes_df, on='repo', how='left')
    
        # Select the requested static and pre-calculated repo metrics
        repo_columns_to_keep = [
            'repo', #'stargazerCount', 'forkCount', 'watchers_count', 'total_commits_all_time',
            'age_at_first_acf_days', 'pre_acf_unique_contributors', 'pre_acf_total_issues', 
            'pre_acf_total_prs', 'readme_content', 'first_acf_date'
        ]
        
        df_repo = df_repo[[c for c in repo_columns_to_keep if c in df_repo.columns]]
    except FileNotFoundError:
        print(f"Error: Could not find {REPO_METRICS_FILE}")
        return

    # 2. Load Raw PRs and Compute Averages
    try:
        df_raw_prs = pd.read_csv(RAW_PRS_FILE)
        df_pr_metrics = aggregate_pr_metrics(df_raw_prs, df_repo)
    except FileNotFoundError:
        print(f"Warning: {RAW_PRS_FILE} not found. Skipping PR aggregation.")
        df_pr_metrics = pd.DataFrame(columns=['repo'])

    # 3. Load Repo Advanced metrics
    try:
        df_repo_adva_metrics = pd.read_csv(ADVANCED_REPO_METRICS_FILE)
        df_repo_adva_metrics = df_repo_adva_metrics.drop(columns=['first_acf_date', 'resolved_commit'])
    except FileNotFoundError:
        print(f"Warning: {ADVANCED_REPO_METRICS_FILE} not found. Skipping PR aggregation.")
        raise FileNotFoundError(f"Required file {ADVANCED_REPO_METRICS_FILE} not found. Please ensure it exists.")

    # 4. Extract the Dependent Variables (Topics)
    df_topics = extract_latest_acf_topics()

    # 5. Merge DataFrames
    print("Merging features and labels...")
    df_final = df_repo.merge(df_pr_metrics, on='repo', how='left')
    df_final = df_final.merge(df_topics, on='repo', how='inner') # Drops repos without ACFs
    df_final = df_final.merge(df_repo_adva_metrics, on='repo', how='left')

    # Fill NaNs with 0 for PR metrics where a repo had no historical PRs
    # (Leaves text columns and embedding targets alone)
    pr_fill_cols = [c for c in df_final.columns if c.startswith('avg_') and c.endswith('_before_acf')]
    for c in pr_fill_cols:
        df_final[c] = df_final[c].fillna(0)

    # 6. Generate Embeddings for Project Descriptions
    df_final = generate_readme_embeddings(df_final)

    # 7. One-Hot Encode the Dependent Variable
    print("One-Hot Encoding target topics...")
    topic_dummies = pd.get_dummies(df_final['final_topics'].explode(), prefix='topic').groupby(level=0).max()
    df_final = pd.concat([df_final, topic_dummies], axis=1)

    # Optional: drop columns only used for processing
    if 'first_acf_date' in df_final.columns:
        df_final.drop(columns=['first_acf_date'], inplace=True)
    if 'final_topics' in df_final.columns:
        df_final.drop(columns=['final_topics'], inplace=True)

    # 8. Export the final dataset
    df_final.to_parquet(OUTPUT_DATASET_FILE, index=False)
    
    # Save CSV version without the embedding array for easy visual inspection
    df_final_csv = df_final.drop(columns=['readme_embedding'])
    df_final_csv.to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"\nSuccess! Dataset generated with {len(df_final)} repositories.")
    print(f"Parquet file (contains embeddings) saved to: {OUTPUT_DATASET_FILE}")
    print(f"CSV file (no embeddings) saved to: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()