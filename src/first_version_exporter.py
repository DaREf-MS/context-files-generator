import json
import logging
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "repo_snapshot"
OUTPUT_FILE = DATA_DIR / "context_files_all_versions.jsonl"
UNIQUE_CHAR = "&SEP&"
COUNT_PR_THRESHOLD = 20

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_markdown(text):
    """
    Applies pre-processing techniques to the context file content:
    1. Truncates long paragraphs to 5 words.
    2. Keeps markdown headers AS IS.
    3. Removes code snippets.
    4. Preserves import statements and file paths.
    """
    lines = text.split('\n')
    processed_lines = []
    in_code_block = False
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            words = para_text.split()
            if len(words) > 5:
                # Truncate to first 5 words
                processed_lines.append(' '.join(words[:5]) + '...')
            else:
                processed_lines.append(para_text)
            current_paragraph.clear()

    for line in lines:
        stripped = line.strip()

        # Rule 3: Handle code block toggling
        if stripped.startswith('```'):
            flush_paragraph()
            in_code_block = not in_code_block
            continue

        if in_code_block:
            # Rule 4: Keep import statements and paths even inside code blocks
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or 
                '/' in stripped or 
                '\\' in stripped):
                processed_lines.append(line)
            continue

        # Rule 2: Keep all level headers AS IS
        if re.match(r'^#{1,6}\s+', stripped):
            flush_paragraph()
            processed_lines.append(line)
            continue

        # Rule 4 (outside code blocks): Bypass truncation for imports and paths
        if (stripped.startswith('import ') or 
            stripped.startswith('from ') or 
            '/' in stripped or 
            '\\' in stripped):
            flush_paragraph()
            processed_lines.append(line)
            continue

        # Empty lines break paragraphs
        if not stripped:
            flush_paragraph()
            processed_lines.append('')
            continue

        # Rule 1: Accumulate standard text for paragraph truncation
        current_paragraph.append(stripped)

    # Flush any remaining paragraph at the end of the file
    flush_paragraph()
    
    return '\n'.join(processed_lines).strip()

def collect_all_versions():
    dataset = []
    
    if not SNAPSHOT_DIR.exists():
        logging.error(f"Snapshot directory {SNAPSHOT_DIR} does not exist.")
        return
    
    # Load and filter repositories based on metrics
    try:
        df = pd.read_csv(DATA_DIR / "repo_agent_metrics_comparison_mode_1.csv")
        df = df[
            # (df['pr_count_before'] >= COUNT_PR_THRESHOLD) &
            (df['pr_count_after'] >= COUNT_PR_THRESHOLD)
        ]
        repos_to_consider = df['repo'].tolist()
        repos_to_consider = [r.replace("/", "§") for r in repos_to_consider]
    except Exception as e:
        logging.error(f"Failed to read metrics CSV: {e}")
        return

    # Get a list of all repo folders
    repos = [d for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and d.name in repos_to_consider]
    
    logging.info(f"Scanning {len(repos)} repositories...")

    for repo_path in tqdm(repos, desc="Processing Repositories"):
        repo_name = repo_path.name
        # if :
            # continue
        
        # Each repo has folders named after the context files
        context_file_folders = [d for d in repo_path.iterdir() if d.is_dir()]
        
        for file_folder in context_file_folders:
            context_file_name = file_folder.name
            
            # List all .md files (versions) in this folder
            versions = sorted([f for f in file_folder.glob("*.md")])
            
            for version_path in versions:
                try:
                    with open(version_path, 'r', encoding='utf-8') as f:
                        raw_content = f.read()
                    
                    # --- MODIFICATION: Apply preprocessing rules ---
                    processed_content = preprocess_markdown(raw_content)
                    
                    row = {
                        "id": f"{repo_name}{UNIQUE_CHAR}{context_file_name}{UNIQUE_CHAR}{version_path.name}",
                        "repo": repo_name,
                        "file": context_file_name,
                        "version": version_path.name,
                        "text": processed_content
                    }
                    dataset.append(row)
                    
                except Exception as e:
                    logging.error(f"Could not read {version_path}: {e}")

    # Write to JSONL
    logging.info(f"Writing {len(dataset)} records to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logging.info("Successfully created JSONL dataset.")
    except Exception as e:
        logging.error(f"Failed to write JSONL file: {e}")

if __name__ == "__main__":
    collect_all_versions()