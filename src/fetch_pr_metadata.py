import requests
import time
import pandas as pd
import json
import os
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "configured_repos.csv"

# Updated Date Range
START_DATE = "2026-03-30"
END_DATE = "2026-06-11"

AGENT_CONFIG = {
    # Start dates here are kept for historical reference, but the script now uses the global START_DATE
    'claude':  {'query': '"Co-Authored-By: Claude" OR "Generated with Claude Code"', 'start_date': '2025-02-24'},
    'codex':   {'query': 'head:codex/', 'start_date': '2025-05-16'},
    'copilot': {'query': 'head:copilot/', 'start_date': '2025-01-01'},
    'cursor':  {'query': 'head:cursor/', 'start_date': '2025-01-01'},
    'gemini':  {'query': '"Co-Authored-By: Gemini"', 'start_date': '2025-01-01'} 
}

# In-memory caches to reduce redundant API calls and duplicate rows
repo_cache = {}
user_cache = {}
issue_cache = set() # Stores issue IDs we've already written to issues.jsonl

# --- I/O Writer ---
class JSONLWriter:
    """Manages appending multiple relational data streams to separate JSONL files."""
    def __init__(self, output_dir, part):
        self.output_dir = output_dir
        self.part = part
        self.handles = {}

    def get_handle(self, table_name):
        if table_name not in self.handles:
            path = self.output_dir / f"{table_name}_part{self.part}.jsonl"
            self.handles[table_name] = open(path, 'a', encoding='utf-8')
        return self.handles[table_name]

    def write(self, table_name, record):
        if record:
            f = self.get_handle(table_name)
            f.write(json.dumps(record) + '\n')
            f.flush()  # Flush to ensure data safety on interruption

    def close_all(self):
        for f in self.handles.values():
            f.close()

# --- Network & Rate Limit Handling ---
def handle_rate_limit(response, logger):
    try:
        remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
        if remaining <= 1:
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait_seconds = max(reset_time - time.time() + 5, 0)
            logger.warning(f"⏳ Rate limit hit. Sleeping {wait_seconds:.0f}s...")
            time.sleep(wait_seconds)
            return wait_seconds
    except Exception as e:
        logger.error(f"Error parsing rate limits: {e}")
    return 60 

def make_request(url, headers, logger, params=None):
    while True:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code in [403, 429] and 'rate limit' in response.text.lower():
                handle_rate_limit(response, logger)
                continue
            
            if response.status_code != 200:
                if response.status_code != 404:
                    logger.error(f"❌ HTTP Error {response.status_code}: {url}")
                return None
            
            return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"⚠️ Network error: {e}. Retrying in 5s...")
            time.sleep(5)

def fetch_all_pages(url, headers, logger, params=None):
    if params is None: params = {}
    if 'per_page' not in params: params['per_page'] = 100
    
    while url:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code in [403, 429] and 'rate limit' in response.text.lower():
                handle_rate_limit(response, logger)
                continue
            
            if response.status_code != 200:
                break

            data = response.json()
            items = data.get('items', data) if isinstance(data, dict) else data
            if not items: break

            for item in items:
                yield item

            if 'next' in response.links:
                url = response.links['next']['url']
                params = {} 
            else:
                url = None
        except Exception as e:
            logger.error(f"Pagination error: {e}")
            break

# --- Schema Mapping Functions ---

def fetch_user_data(username, headers, logger, writer):
    if not username: return None
    if username in user_cache: return user_cache[username]
    
    data = make_request(f"https://api.github.com/users/{username}", headers, logger)
    if not data: return None
    
    user_record = {
        "id": data.get("id"),
        "login": data.get("login"),
        "type": data.get("type"),
        "name": data.get("name"),
        "email": data.get("email"),
        "company": data.get("company"),
        "location": data.get("location"),
        "followers": data.get("followers"),
        "following": data.get("following"),
        "public_repos": data.get("public_repos"),
        "created_at": data.get("created_at")
    }
    user_cache[username] = user_record
    
    writer.write('user', user_record)
    return user_record

def fetch_repo_data(owner, repo, headers, logger, writer):
    repo_full_name = f"{owner}/{repo}"
    if repo_full_name in repo_cache: return repo_cache[repo_full_name]
    
    data = make_request(f"https://api.github.com/repos/{owner}/{repo}", headers, logger)
    if not data: return None
    
    repo_record = {
        "id": data.get("id"),
        "url": data.get("html_url"),
        "full_name": data.get("full_name"),
        "language": data.get("language"),
        "forks": data.get("forks_count"),
        "stars": data.get("stargazers_count")
    }
    repo_cache[repo_full_name] = repo_record
    
    writer.write('repository', repo_record)
    return repo_record

def process_agent_prs(repo_name, agent_name, config, headers, logger, writer):
    owner, repo = repo_name.split('/')
    repo_data = fetch_repo_data(owner, repo, headers, logger, writer)
    
    pr_count = 0
    
    # Date Chunking Implementation using global START_DATE and END_DATE
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    current_dt = start_dt
    
    while current_dt <= end_dt:
        # Create a 30-day chunk window
        chunk_end_dt = min(current_dt + timedelta(days=30), end_dt)
        chunk_start_str = current_dt.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end_dt.strftime("%Y-%m-%d")
        
        search_query = f"repo:{repo_name} is:pr {config['query']} created:{chunk_start_str}..{chunk_end_str}"
        search_url = "https://api.github.com/search/issues"
        search_params = {"q": search_query, "sort": "created", "order": "desc"}
        
        logger.info(f"     -> Querying timeframe: {chunk_start_str} to {chunk_end_str}")

        for item in fetch_all_pages(search_url, headers, logger, search_params):
            pr_number = item['number']
            pr_id = item['id']
            pr_count += 1
            
            # PR Base
            pull_request = {
                "id": pr_id,
                "number": pr_number,
                "title": item['title'],
                "body": item.get('body'),
                "user_id": item['user']['id'] if item.get('user') else None,
                "user": item['user']['login'] if item.get('user') else None,
                "state": item['state'],
                "created_at": item['created_at'],
                "closed_at": item['closed_at'],
                "merged_at": item.get('pull_request', {}).get('merged_at'),
                "repo_id": repo_data['id'] if repo_data else None,
                "repo_url": repo_data['url'] if repo_data else None,
                "html_url": item['html_url']
            }
            
            writer.write('pull_request', pull_request)

            # Ensure user exists in users table
            fetch_user_data(pull_request['user'], headers, logger, writer)

            # Commits & Details
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits"
            for c in fetch_all_pages(commits_url, headers, logger):
                sha = c['sha']
                
                author_login = c.get('author', {}).get('login') if c.get('author') else None
                committer_login = c.get('committer', {}).get('login') if c.get('committer') else None
                
                commit_record = {
                    "sha": sha,
                    "pr_id": pr_id,
                    "author": author_login,
                    "committer": committer_login,
                    "message": c['commit']['message']
                }
                writer.write('pr_commits', commit_record)
                
                commit_detail = make_request(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}", headers, logger)
                if commit_detail and 'files' in commit_detail:
                    stats = commit_detail.get('stats', {})
                    for f in commit_detail['files']:
                        detail_record = {
                            "sha": sha,
                            "pr_id": pr_id,
                            "author": author_login,
                            "committer": committer_login,
                            "message": c['commit']['message'],
                            "commit_stats_total": stats.get('total'),
                            "commit_stats_additions": stats.get('additions'),
                            "commit_stats_deletions": stats.get('deletions'),
                            "filename": f['filename'],
                            "status": f['status'],
                            "additions": f['additions'],
                            "deletions": f['deletions'],
                            "changes": f['changes'],
                            "patch": f.get('patch')
                        }
                        writer.write('pr_commit_details', detail_record)

            # Comments
            comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
            for ic in fetch_all_pages(comments_url, headers, logger):
                comment_record = {
                    "id": ic['id'],
                    "pr_id": pr_id,
                    "user": ic['user']['login'] if ic.get('user') else None,
                    "user_id": ic['user']['id'] if ic.get('user') else None,
                    "created_at": ic['created_at'],
                    "body": ic['body']
                }
                writer.write('pr_comments', comment_record)

            # Reviews & Timeline
            seen_issue_ids_for_pr = set()
            timeline_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/timeline"
            for event in fetch_all_pages(timeline_url, headers, logger):
                actor = event.get('actor', {}).get('login') or event.get('user', {}).get('login')
                assignee = event.get('assignee', {}).get('login')
                label = event.get('label', {}).get('name')
                
                timeline_record = {
                    "pr_id": pr_id,
                    "event": event['event'],
                    "commit_id": event.get('commit_id'),
                    "created_at": event.get('created_at') or event.get('submitted_at'),
                    "actor": actor,
                    "assignee": assignee,
                    "label": label,
                    "message": event.get('message')
                }
                writer.write('pr_timeline', timeline_record)
                
                # Reviews Table Mapping
                if event['event'] == 'reviewed':
                    review_record = {
                        "id": event.get('id'),
                        "pr_id": pr_id,
                        "user": actor,
                        "state": event.get('state'),
                        "submitted_at": event.get('submitted_at'),
                        "body": event.get('body')
                    }
                    writer.write('pr_reviews', review_record)

                # Issue & Related Issues Mapping
                if event['event'] == 'cross-referenced' and event.get('source', {}).get('type') == 'issue':
                    issue_obj = event['source']['issue']
                    issue_id = issue_obj.get('id')
                    
                    if issue_id and issue_id not in seen_issue_ids_for_pr:
                        seen_issue_ids_for_pr.add(issue_id)
                        
                        # 1. Write the junction table record
                        related_issue_record = {
                            "pr_id": pr_id,
                            "issue_id": issue_id,
                            "source": issue_obj.get('html_url')
                        }
                        writer.write('related_issue', related_issue_record)
                        
                        # 2. Write the actual issue record (if not already cached)
                        if issue_id not in issue_cache:
                            issue_cache.add(issue_id)
                            
                            issue_record = {
                                "id": issue_id,
                                "number": issue_obj.get('number'),
                                "title": issue_obj.get('title'),
                                "body": issue_obj.get('body'),
                                "user": issue_obj.get('user', {}).get('login') if issue_obj.get('user') else None,
                                "state": issue_obj.get('state'),
                                "created_at": issue_obj.get('created_at'),
                                "closed_at": issue_obj.get('closed_at')
                            }
                            writer.write('issue', issue_record)
        
        # Advance to the next chunk
        current_dt = chunk_end_dt + timedelta(days=1)
                    
    return pr_count

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Fetch PR metadata split across tokens.")
    parser.add_argument('--part', type=int, choices=[1, 2], required=True, help="Which half of the CSV to process (1 or 2).")
    args = parser.parse_args()

    log_file = ROOT_DIR / f"collection_part{args.part}.log"
    tracker_file = DATA_DIR / f"progress_checker_part{args.part}.txt"
    
    token_env = 'TOKEN_1_JAPAN_PROJECT' if args.part == 1 else 'TOKEN_2_JAPAN_PROJECT'
    github_token = os.getenv(token_env)
    
    if not github_token:
        raise ValueError(f"Missing {token_env} in environment variables.")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    if not INPUT_FILE.exists():
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    if 'repo_name' not in df.columns:
        logger.error("❌ CSV must contain a 'repo_name' column.")
        return

    # Split the DataFrame
    midpoint = len(df) // 2
    if args.part == 1:
        df = df.iloc[:midpoint]
        logger.info(f"🚀 Running Part 1: Processing repositories 0 to {midpoint-1}")
    else:
        df = df.iloc[midpoint:]
        logger.info(f"🚀 Running Part 2: Processing repositories {midpoint} to {len(df)-1}")

    processed_tasks = set()
    if tracker_file.exists():
        with open(tracker_file, 'r', encoding='utf-8') as f:
            processed_tasks = set(line.strip() for line in f if line.strip())
    
    logger.info(f"Loaded {len(processed_tasks)} already processed repo-agent pairs.")

    # Initialize the relational Data Writer
    writer = JSONLWriter(DATA_DIR, args.part)

    try:
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Scanning Repos (Part {args.part})"):
            repo_name = row['repo_name']
            
            active_agents = [agent for agent in AGENT_CONFIG.keys() if agent in row and row[agent] == True]
            
            for agent in active_agents:
                task_id = f"{repo_name}_{agent}"
                
                if task_id in processed_tasks:
                    continue
                
                logger.info(f"🔍 Mining {agent} PRs for {repo_name}...")
                
                try:
                    pr_count = process_agent_prs(repo_name, agent, AGENT_CONFIG[agent], headers, logger, writer)
                    
                    logger.info(f"   ✅ Saved {pr_count} PRs and associated relational data.")
                    
                    with open(tracker_file, 'a', encoding='utf-8') as f_track:
                        f_track.write(f"{task_id}\n")
                        
                except Exception as e:
                    logger.error(f"❌ Unhandled error on {task_id}: {e}")
                    continue

        logger.info(f"🎉 Pipeline execution finished. Metadata saved to {DATA_DIR}")
    finally:
        # Guarantee safe closure of all file pointers
        writer.close_all()

if __name__ == "__main__":
    main()