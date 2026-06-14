import os
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("instruction_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_CSV = DATA_DIR / "plus_twenty_merge_rate_context_files.csv"
OUTPUT_CSV = DATA_DIR / "plus_twenty_merge_rate_context_complexity_metrics.csv"
REPOS_BASE_DIR = DATA_DIR / "repos" 

# OpenAI Setup
MODEL_NAME = "gpt-4o-mini" # Note: Updated to gpt-4o-mini as gpt-5 is not public
OPENAI_TOKEN = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_TOKEN)

SYSTEM_PROMPT = """
You are an expert Software Engineering Code Analyzer. Your task is to analyze developer-authored instruction files (context files) designed to guide autonomous AI coding agents. 

Your goal is to extract key characteristics that demonstrate how the developer manages project complexity, constraints the agent's behavior, and handle different external resources (i.e., APIs). 

Analyze the provided instruction file and extract the following dimensions. Output your analysis STRICTLY as a JSON object matching the schema below. 

### Extraction Dimensions:

1. Actionable Execution (Testing & Tooling):
   - Does the file provide explicit, runnable CLI commands for the agent to execute (e.g., linting, building, testing)?
   - Are there explicit instructions to run tests before completing a task?

2. Negative Constraints (Guardrails):
   - Developers manage complexity by telling agents what NOT to do. Count and extract the explicit negative constraints (look for words like "NEVER", "DO NOT", "AVOID", "INSTEAD OF").

3. Architectural & Structural Navigation:
   - Does the file provide a mental map of the project (e.g., explaining where specific logic lives, port numbers, or database stack)?
   - Are there multi-step navigation rules (e.g., "If you change X, you must also update Y and Z")?

4. Strictness & Enforcement:
   - Rate the overall tone and enforcement level of the document from 1 (loose guidelines) to 10 (highly rigid, systemic constraints).

### JSON Output Schema:
{
  "execution_and_tooling": {
    "has_executable_commands": boolean,
    "requires_test_execution": boolean,
    "extracted_commands": [string],
    "testing_frameworks_mentioned": [string]
  },
  "guardrails": {
    "negative_constraint_count": integer,
    "key_negative_constraints": [string]
  },
  "structural_navigation": {
    "provides_architecture_mapping": boolean,
    "has_multi_step_workflows": boolean,
    "multi_step_examples": [string]
  },
  "strictness_score": integer,
  "complexity_summary": "A 2-sentence summary of how this file attempts to control the agent's behavior."
}
"""

def extract_features_from_llm(repo_name, file_name, file_content):
    """Calls the OpenAI API and returns the parsed JSON."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this instruction file:\n\n{file_content}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0,
            max_tokens=2000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"API Error for {repo_name}/{file_name}: {str(e)}")
        return {"error": str(e)}

def flatten_json_to_row(repo_name, context_file, llm_data):
    """Flattens the nested JSON response into a flat dictionary for CSV rows."""
    if "error" in llm_data:
        return {
            "repo_name": repo_name,
            "context_file": context_file,
            "error_msg": llm_data["error"]
        }
        
    return {
        "repo_name": repo_name,
        "context_file": context_file,
        "has_executable_commands": llm_data.get("execution_and_tooling", {}).get("has_executable_commands", False),
        "requires_test_execution": llm_data.get("execution_and_tooling", {}).get("requires_test_execution", False),
        "cmd_count": len(llm_data.get("execution_and_tooling", {}).get("extracted_commands", [])),
        "frameworks": " | ".join(llm_data.get("execution_and_tooling", {}).get("testing_frameworks_mentioned", [])),
        "negative_constraint_count": llm_data.get("guardrails", {}).get("negative_constraint_count", 0),
        "key_constraints_preview": " | ".join(llm_data.get("guardrails", {}).get("key_negative_constraints", [])[:3]),
        "provides_architecture_mapping": llm_data.get("structural_navigation", {}).get("provides_architecture_mapping", False),
        "has_multi_step_workflows": llm_data.get("structural_navigation", {}).get("has_multi_step_workflows", False),
        "strictness_score": llm_data.get("strictness_score", 0),
        "complexity_summary": llm_data.get("complexity_summary", ""),
        "error_msg": ""
    }

def main():
    if not INPUT_CSV.exists():
        logger.critical(f"Input file not found: {INPUT_CSV}")
        return

    logger.info(f"Loading data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV).iloc[:10]
    
    if 'repo_name' not in df.columns or 'context_file' not in df.columns:
        logger.error("Missing required columns: repo_name or context_file")
        return

    results = []
    pbar = tqdm(total=len(df), desc="Analyzing Context Files")

    try:
        for _, row in df.iterrows():
            repo_name = row['repo_name'].replace('/', '§')
            file_name = row['context_file']
            
            pbar.set_postfix({'repo': repo_name[:15]})
            file_path = REPOS_BASE_DIR / repo_name / file_name
            
            if not file_path.exists():
                logger.warning(f"File skipped (not found): {repo_name}/{file_name}")
                results.append({"repo_name": repo_name, "context_file": file_name, "error_msg": "File not found"})
                pbar.update(1)
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Read error on {file_path}: {str(e)}")
                results.append({"repo_name": repo_name, "context_file": file_name, "error_msg": f"Read error: {str(e)}"})
                pbar.update(1)
                continue
                
            llm_response = extract_features_from_llm(repo_name, file_name, file_content)
            results.append(flatten_json_to_row(repo_name, file_name, llm_response))
            pbar.update(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted. Cleaning up...")
    finally:
        pbar.close()
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Analysis complete. Results saved to {OUTPUT_CSV}")
        else:
            logger.warning("No results to save.")

if __name__ == "__main__":
    main()