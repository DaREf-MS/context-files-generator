import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_markdown_metrics():
    # Setup Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    
    # Specific input/output paths
    csv_input_path = DATA_DIR / "context_files.csv"
    base_dir = DATA_DIR / "repos"
    output_path = DATA_DIR / "context_file_metrics.csv"
    
    if not csv_input_path.exists():
        print(f"Error: CSV file not found at {csv_input_path}")
        return

    # Load the context data
    df_context = pd.read_csv(csv_input_path)
    
    # Regex Patterns
    word_pattern = re.compile(r'\w+')
    header_patterns = {
        'H1': re.compile(r'^#\s', re.MULTILINE),
        'H2': re.compile(r'^##\s', re.MULTILINE),
        'H3': re.compile(r'^###\s', re.MULTILINE),
        'H4': re.compile(r'^####\s', re.MULTILINE),
        'H5': re.compile(r'^#####\s', re.MULTILINE),
        'H6': re.compile(r'^######\s', re.MULTILINE),
    }

    results = []

    # Group by repo_name to aggregate metrics per repository
    repo_groups = df_context.groupby('repo_name')

    print(f"Processing {len(repo_groups)} repositories from CSV...")

    for repo_name, group in tqdm(repo_groups, desc="Quantifying Metrics"):
        total_words = 0
        content_length = 0
        headers = {f'H{i}': 0 for i in range(1, 7)}

        for _, row in group.iterrows():
            # Construct path: data/downloaded_agent_configs/repo_name/context_file
            file_path = base_dir / repo_name.replace("/", "§") / row['context_file']
            
            if not file_path.exists():
                # Optional: print(f"Warning: File {file_path} not found.")
                continue
                
            try:
                content = None
                # List of encodings to try in order of likelihood
                encodings = ['utf-8-sig', 'utf-16', 'utf-16le', 'utf-16be', 'latin-1']
                
                for enc in encodings:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read()
                        break # Success! Exit the encoding loop
                    except (UnicodeDecodeError, UnicodeError):
                        continue # Try the next one
                
                if content is None:
                    print(f"Skipping {file_path}: Could not determine encoding.")
                    continue

                # Quantify Words and Length
                total_words += len(word_pattern.findall(content))
                content_length += len(content)
                
                # Quantify H-Metrics
                for tag, pattern in header_patterns.items():
                    headers[tag] += len(pattern.findall(content))
                        
            except Exception as e:
                print(f"Skipping {file_path} due to unexpected error: {e}")

        # Store aggregated metrics for this repo
        results.append({
            'repo': repo_name,
            'word_count': total_words,
            'content_length': content_length,
            **headers
        })

    # Save to CSV
    df_results = pd.DataFrame(results, )
    df_results.to_csv(output_path, index=False)
    print(f"\nDone! Metrics quantified for {len(df_results)} repos.")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    extract_markdown_metrics()