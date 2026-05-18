import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import ast
from tqdm import tqdm

# 1. Environment Setup & Directory Patching
load_dotenv()
GITHUB_TOKEN = os.getenv("TOKEN_1_JAPAN_PROJECT")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "configured_repos.csv"
RAW_OUTPUT_PATH = DATA_DIR / "processed_repos_raw.csv"  # Saves row-by-row
OUTPUT_PATH = DATA_DIR / "dependency_summary.csv"       # Final aggregated metrics

# 2. Define Supported Languages Configuration
TARGET_LANGUAGES = {
    'typescript': ['package.json'],
    'javascript': ['package.json'],
    'python': ['requirements.txt', 'pyproject.toml'],
    'go': ['go.mod'],
    'java': ['pom.xml', 'build.gradle'],
    'c#': ['packages.config'], 
    'rust': ['Cargo.toml'],
    'c++': ['vcpkg.json', 'CMakeLists.txt'],
    'c': ['CMakeLists.txt'],
    'php': ['composer.json']
}

def extract_owner_repo(github_link):
    """Extracts owner and repo name from a GitHub URL."""
    if pd.isna(github_link):
        return None, None
    parts = github_link.rstrip('/').replace('.git', '').split('/')
    if len(parts) >= 5:
        return parts[-2], parts[-1]
    return None, None

def fetch_github_file(owner, repo, branch, filename):
    """Fetches a manifest file via GitHub API with built-in rate limit handling."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}?ref={branch}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            
            # Extract Rate Limit Headers
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_time = response.headers.get("X-RateLimit-Reset")
            
            # 1. Handle Primary Rate Limit Exhaustion
            if response.status_code == 403 and remaining == "0":
                if reset_time:
                    sleep_time = max(int(reset_time) - int(time.time()), 0) + 2
                    print(f"\n[Rate Limit] Primary limit hit. Sleeping for {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
            
            # 2. Handle Secondary Rate Limit / Abuse Detection
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"\n[Rate Limit] Secondary limit hit. Sleeping for {retry_after}s...")
                time.sleep(retry_after)
                continue

            if response.status_code == 200:
                # Small proactive delay to avoid hitting secondary limits
                time.sleep(0.1)
                return response.text
                
            # For other statuses (e.g., 404 Not Found), break out
            break
            
        except requests.exceptions.RequestException:
            time.sleep(2)  # Network hiccup, back off slightly
            break
            
    return None

def count_dependencies(content, filename=""):
    """Parses manifest text contents to count declared third-party dependencies."""
    if not content:
        return 0
    
    try:
        # 1. JSON-based package managers (NodeJS, PHP, C++ vcpkg)
        if filename in ['package.json', 'composer.json', 'vcpkg.json']:
            data = json.loads(content)
            deps = len(data.get('dependencies', {})) + len(data.get('devDependencies', {}))
            deps += len(data.get('require', {})) + len(data.get('require-dev', {}))
            return deps

        # 2. Python requirements.txt
        if filename == 'requirements.txt':
            lines = content.splitlines()
            return sum(1 for l in lines if l.strip() and not l.strip().startswith(('#', '-')))

        # 3. Python pyproject.toml (Handles PEP 621 standard arrays & Poetry style tables)
        if filename == 'pyproject.toml':
            lines = content.splitlines()
            count = 0
            in_deps_block = False
            in_deps_array = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Table Headers (e.g., [tool.poetry.dependencies] or [project.optional-dependencies])
                if line.startswith('[') and line.endswith(']'):
                    block_name = line.strip('[]')
                    in_deps_block = 'dependencies' in block_name
                    continue
                
                # PEP 621 Array Style: dependencies = [ ... ]
                if (line.startswith('dependencies') or 'dependencies =' in line) and '=' in line:
                    if '[' in line and ']' not in line:
                        in_deps_array = True
                        continue
                    elif '[' in line and ']' in line:
                        items = line.split('[')[1].split(']')[0].split(',')
                        count += sum(1 for item in items if item.strip().strip('"\''))
                        continue

                if in_deps_array:
                    if ']' in line:
                        in_deps_array = False
                        item = line.split(']')[0].strip().strip(',').strip('"\'')
                        if item: count += 1
                    else:
                        item = line.strip().strip(',').strip('"\'')
                        if item: count += 1
                    continue

                # Key-Value Style (Poetry / Hatch) inside dependency blocks
                if in_deps_block and '=' in line:
                    key = line.split('=')[0].strip()
                    if key.lower() != 'python':  # Skips the python engine requirement rule
                        count += 1
            return count

        # 4. Go modules
        if filename == 'go.mod':
            lines = content.splitlines()
            count = 0
            in_block = False
            for line in lines:
                line = line.strip()
                if line.startswith('require ('):
                    in_block = True
                    continue
                if in_block and line == ')':
                    in_block = False
                    continue
                if (in_block and line) or line.startswith('require '):
                    count += 1
            return count

        # 5. Rust Cargo
        if filename == 'Cargo.toml':
            lines = content.splitlines()
            count = 0
            in_deps = False
            for line in lines:
                line = line.strip()
                if line.startswith('[') and 'dependencies' in line:
                    in_deps = True
                    continue
                elif line.startswith('['):
                    in_deps = False
                if in_deps and line and not line.startswith('#') and '=' in line:
                    count += 1
            return count

        # 6. Java Maven (pom.xml)
        if filename == 'pom.xml':
            return content.count('<dependency>')

        # 7. Java Gradle (build.gradle)
        if filename == 'build.gradle':
            lines = content.splitlines()
            count = 0
            # Standard Gradle dependency scopes
            keywords = ('implementation', 'api', 'compile', 'testImplementation', 'runtimeOnly', 'compileOnly', 'testRuntimeOnly', 'classpath')
            for line in lines:
                line = line.strip()
                if line.startswith('//') or line.startswith('*') or line.startswith('/*'):
                    continue
                if any(line.startswith(kw) for kw in keywords):
                    if "'" in line or '"' in line or "project(" in line:
                        count += 1
            return count

        # 8. C# Manifests
        if '.csproj' in filename or filename == 'packages.config':
            return content.count('<PackageReference') + content.count('<package id=')
            
        # 9. C/C++ CMake
        if filename == 'CMakeLists.txt':
            return content.count('find_package(') + content.count('FetchContent_Declare(')

    except Exception:
        pass
    return 0

def process_repository(row):
    """Orchestrates the discovery and parsing of a repository's dependency manifest."""
    if not row.get('languages') or not isinstance(row.get('languages'), dict):
        return 0
        
    languages = {k.lower() for k in row['languages'].keys() if k.lower() in TARGET_LANGUAGES}
    if len(languages) == 0:
        return 0
    
    try:
        owner, repo = row.get('repo_name').split("/")
    except ValueError:
        return 0
        
    branch = row.get('default_branch') if pd.notna(row.get('default_branch')) else "main"
    
    manifests = []
    for lang in languages:
        manifests.extend(TARGET_LANGUAGES[lang])
    
    # Safe handling for unpredictable .csproj naming
    if 'c#' in languages:
        manifests.extend([f"{repo}.csproj", "packages.config"])

    for manifest in manifests:
        content = fetch_github_file(owner, repo, branch, manifest)
        if content:
            return count_dependencies(content, manifest)
            
    return 0

def main():
    if not CSV_PATH.exists():
        print(f"Error: Could not find configuration file at {CSV_PATH}")
        return

    # Load Source File
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded metadata for {len(df)} repositories.")
    df['languages'] = df['languages'].map(ast.literal_eval)
    
    # Bug Fix: Ensure this column is prepared before processing loop or final group operations
    df['mainLanguage_clean'] = df['mainLanguage'].astype(str).str.strip()

    # --- Resume & Incremental Saving Logic ---
    if RAW_OUTPUT_PATH.exists():
        try:
            print("Found existing progress file. Loading completed records...")
            raw_df = pd.read_csv(RAW_OUTPUT_PATH)
            if 'repo_name' in raw_df.columns and 'dependency_count' in raw_df.columns:
                # Map completed counts back onto our driving dataframe
                completed_map = raw_df.set_index('repo_name')['dependency_count'].to_dict()
                df['dependency_count'] = df['repo_name'].map(completed_map)
        except Exception as e:
            print(f"Warning: Could not read progress file ({e}). Starting fresh.")
            
    if 'dependency_count' not in df.columns:
        df['dependency_count'] = None

    # --- Process via Loops paired with Progress Bars ---
    print("Scanning remote repositories via GitHub API...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scanning Repositories"):
        # Skip if already computed in a previous run
        if pd.notna(row['dependency_count']):
            continue
            
        # Process and write update back to main tracking dataframe
        dep_count = process_repository(row)
        df.at[index, 'dependency_count'] = dep_count
        
        # Incremental Save: Slice out single-row DataFrame to append safely to file
        row_slice = pd.DataFrame([df.loc[index]])
        row_slice['languages'] = row_slice['languages'].astype(str) # Stringify dict for smooth CSV writing
        
        file_exists = RAW_OUTPUT_PATH.exists()
        row_slice.to_csv(RAW_OUTPUT_PATH, mode='a', header=not file_exists, index=False)

    # Convert counts back to numeric values for aggregation math
    df['dependency_count'] = pd.to_numeric(df['dependency_count'], errors='coerce').fillna(0)

    # Filter and group metrics
    summary = df.groupby('mainLanguage_clean')['dependency_count'].agg(
        total_dependencies='sum',
        average_dependencies='mean',
        max_dependencies='max',
        scanned_projects='count'
    ).reset_index()

    summary['average_dependencies'] = summary['average_dependencies'].round(2)
    summary = summary.sort_values(by='scanned_projects', ascending=False)

    # Print results and cache file back to the specified Data Directory
    print("\n=== Dependency Metrics Matrix ===")
    print(summary.to_string(index=False))
    
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAggregated dataset exported successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()