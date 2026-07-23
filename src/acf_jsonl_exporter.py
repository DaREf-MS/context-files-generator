import json
import logging
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import tiktoken

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "repo_snapshot"
OUTPUT_FILE = DATA_DIR / "context_files_all_versions.jsonl"
UNIQUE_CHAR = "&SEP&"
COUNT_PR_THRESHOLD = 20

# --- Prompt File Paths ---
DOCS_DIR = ROOT_DIR / "docs" / "prompts"
GENERATION_PROMPT_PATH = DOCS_DIR / "generation_1.txt"
ASSIGNMENT_PROMPT_PATH = DOCS_DIR / "assignment.txt"
SEED_PATH = DOCS_DIR / "seed_1.md"

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
    total_prompt_tokens = 0
    
    if not SNAPSHOT_DIR.exists():
        logging.error(f"Snapshot directory {SNAPSHOT_DIR} does not exist.")
        return

    # Initialize tiktoken encoder
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logging.error(f"Failed to initialize tiktoken: {e}")
        return

    # Load templates and seed content once
    try:
        with open(GENERATION_PROMPT_PATH, 'r', encoding='utf-8') as f:
            gen_prompt_template = f.read()
        with open(ASSIGNMENT_PROMPT_PATH, 'r', encoding='utf-8') as f:
            assignment_prompt_template = f.read()
        with open(SEED_PATH, 'r', encoding='utf-8') as f:
            seed_content = f.read()
    except Exception as e:
        logging.error(f"Failed to read prompt or seed files: {e}")
        return
    
    # Load and filter repositories based on metrics
    try:
        df = pd.read_csv(DATA_DIR / "repo_agent_metrics_comparison_mode_1.csv")
        df = df[df['pr_count_after'] >= COUNT_PR_THRESHOLD]
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
                    
                    # Apply preprocessing rules
                    # processed_content = preprocess_markdown(raw_content)
                    processed_content = raw_content
                    
                    # --- Token Counting Logic ---
                    # Populate generation prompt
                    populated_gen_prompt = gen_prompt_template.replace("{Document}", processed_content).replace("{Topics}", seed_content)
                    # Populate assignment prompt
                    populated_assignment_prompt = assignment_prompt_template.replace("{Document}", processed_content).replace("{Tree}", seed_content)
                    
                    # Calculate tokens for both
                    tokens_gen = len(encoding.encode(populated_gen_prompt))
                    tokens_assignment = len(encoding.encode(populated_assignment_prompt))
                    
                    # Accumulate total tokens
                    total_prompt_tokens += (tokens_gen + tokens_assignment)

                    row = {
                        "id": f"{repo_name}{UNIQUE_CHAR}{context_file_name}{UNIQUE_CHAR}{version_path.name}",
                        "repo": repo_name,
                        "file": context_file_name,
                        "version": version_path.name,
                        "text": processed_content,
                        # Adding these to the JSONL is optional but highly recommended for analysis
                        # "tokens_gen_prompt": tokens_gen,
                        # "tokens_assignment_prompt": tokens_assignment
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
        
        gpt_54_mini_total_input_cost = (total_prompt_tokens / 1000000) * 0.75
        gpt_51_total_input_cost = (total_prompt_tokens / 1000000) * 1.25

        total_output_tokens = 412 * len(dataset) * 2
        gpt_54_mini_total_output_cost = (total_output_tokens / 1000000) * 4.5
        gpt_51_total_output_cost = (total_output_tokens / 1000000) * 10
        
        # Final output of total tokens
        logging.info(f"- TOTAL PROMPT TOKENS ACCUMULATED: {total_prompt_tokens:,}, TOTAL OUPUT TOKENS ACCUMULATED: {total_output_tokens:,}.")
        logging.info(f"- Prompt cost (gpt-5.4-mini): {gpt_54_mini_total_input_cost:.2f}, Output cost: {gpt_54_mini_total_output_cost:.2f}, Total cost: {(gpt_54_mini_total_input_cost+gpt_54_mini_total_output_cost):.2f} USD.")
        logging.info(f"- Prompt cost (gpt-5.1): {gpt_51_total_input_cost:.2f}, Output cost: {gpt_51_total_output_cost:.2f}, Total cost: {(gpt_51_total_input_cost+gpt_51_total_output_cost):.2f} USD.")
        
    except Exception as e:
        logging.error(f"Failed to write JSONL file: {e}")

if __name__ == "__main__":
    collect_all_versions()