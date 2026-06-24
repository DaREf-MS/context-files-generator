import os
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TOPICS_JSONL_FILE = DATA_DIR / "assignment_corrected_sample.jsonl"
OUTPUT_METRICS_FILE = DATA_DIR / "pre_acf_model_features.csv"
PROCESSED_TRACKER_FILE = DATA_DIR / "processed_repos_tracker.txt"

GITHUB_TOKEN = os.getenv("TOKEN_1_JAPAN_PROJECT")
if not GITHUB_TOKEN:
    raise EnvironmentError("Please set the GITHUB_TOKEN environment variable.")

HEADERS = {
    "Authorization": f"bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}

GRAPHQL_URL = "https://api.github.com/graphql"
SEPARATOR = '&SEP&'

# --- GraphQL Queries ---
REPO_METADATA_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    name
    nameWithOwner
    description
    url
    homepageUrl
    isPrivate
    isArchived
    isTemplate
    isDisabled
    createdAt
    updatedAt
    pushedAt
    isFork
    primaryLanguage {
      name
    }
    licenseInfo {
      name
      spdxId
      url
    }
    owner {
      __typename
      ... on User {
        login
        name
      }
      ... on Organization {
        login
        name
      }
    }
    stargazerCount
    forkCount
    watchers {
      totalCount
    }
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(first: 1) {
            totalCount
            nodes {
              committedDate
            }
          }
        }
      }
    }
    repositoryTopics(first: 10) {
      nodes {
        topic {
          name
        }
      }
    }
  }
}
"""

# Query specifically designed to paginate through history and extract authors
PRE_ACF_CONTRIBUTORS_QUERY = """
query($owner: String!, $name: String!, $until: GitTimestamp!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      target {
        ... on Commit {
          history(until: $until, first: 100, after: $cursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              author {
                user {
                  login
                }
                email
                name
              }
            }
          }
        }
      }
    }
  }
}
"""

def get_first_acf_dates():
    """Parses the JSONL to determine the absolute earliest ACF timestamp for every repository."""
    print("Parsing JSONL to determine the first ACF date for each repository...")
    repo_dates = {}

    with open(TOPICS_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_id = row.get("id", "")
            
            if SEPARATOR in raw_id:
                repo_part, file_part = raw_id.split(SEPARATOR, 1)
                repo_name = repo_part.replace("§", "/")
                timestamp_str = file_part.split(SEPARATOR)[1].replace(".md", "").replace('_', ' ')
            else:
                continue

            try:
                dt = pd.to_datetime(timestamp_str, format="mixed", utc=True)
                if repo_name not in repo_dates:
                    repo_dates[repo_name] = dt
                elif dt < repo_dates[repo_name]:
                    repo_dates[repo_name] = dt
            except Exception:
                pass

    return repo_dates


def handle_rate_limit(response, is_search_api=False):
    """Monitors GitHub API rate limits and pauses execution if exhausted."""
    if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
        remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
        if remaining == 0:
            reset_timestamp = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_time = max(reset_timestamp - time.time() + 5, 5)
            api_type = "Search" if is_search_api else "GraphQL"
            print(f"\n[!] {api_type} API Rate limit reached. Sleeping for {sleep_time:.0f} seconds...")
            time.sleep(sleep_time)
            return True
    return False


def get_search_count(query):
    """Executes a REST Search API query for historical (time-filtered) counts."""
    url = f"https://api.github.com/search/issues?q={query}"
    
    while True:
        response = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"})
        if handle_rate_limit(response, is_search_api=True):
            continue
            
        if response.status_code == 200:
            return response.json().get("total_count", 0)
        elif response.status_code == 422:
            return 0
        else:
            return None


def run_graphql_query(query, variables):
    """Executes a generic GraphQL query."""
    while True:
        response = requests.post(GRAPHQL_URL, json={'query': query, 'variables': variables}, headers=HEADERS)
        if handle_rate_limit(response):
            continue
            
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                return None
            return result.get("data", {})
        else:
            return None


def get_pre_acf_contributors(owner, name, date_str):
    """
    Paginates through the commit history up to the first ACF date to extract
    a unique set of historical contributors.
    """
    unique_contributors = set()
    cursor = None
    has_next_page = True
    page_count = 0
    MAX_PAGES = 50 # Safeguard: Cap at 5,000 commits to prevent infinite loops on massive repos
    
    while has_next_page and page_count < MAX_PAGES:
        variables = {"owner": owner, "name": name, "until": date_str, "cursor": cursor}
        data = run_graphql_query(PRE_ACF_CONTRIBUTORS_QUERY, variables)
        
        if not data:
            break
            
        repo_data = data.get("repository")
        if not repo_data:
            break
            
        target = repo_data.get("defaultBranchRef", {}).get("target", {}) if repo_data.get("defaultBranchRef") else {}
        history = target.get("history", {})
        
        for node in history.get("nodes", []):
            author_info = node.get("author")
            if not author_info:
                continue
            
            # Prioritize GitHub login, fallback to email, then name
            user = author_info.get("user")
            if user and user.get("login"):
                unique_contributors.add(user.get("login"))
            elif author_info.get("email"):
                unique_contributors.add(author_info.get("email"))
            elif author_info.get("name"):
                unique_contributors.add(author_info.get("name"))

        page_info = history.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")
        page_count += 1
        
    return len(unique_contributors)


def fetch_repo_metadata(repo, first_acf_date):
    """Fetches GraphQL metadata, historical PR/Issue counts, and historical contributors."""
    try:
        owner, name = repo.split("/")
    except ValueError:
        return None

    date_str = first_acf_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 1. Fetch current comprehensive metadata
    data = run_graphql_query(REPO_METADATA_QUERY, {"owner": owner, "name": name})
    repo_data = data.get("repository") if data else None
    
    if not repo_data:
        return None

    metrics = {
        "repo": repo,
        "first_acf_date": date_str,
        "graphql_id": repo_data.get("id"),
        "description": repo_data.get("description"),
        "url": repo_data.get("url"),
        "homepageUrl": repo_data.get("homepageUrl"),
        "isPrivate": repo_data.get("isPrivate"),
        "isArchived": repo_data.get("isArchived"),
        "isTemplate": repo_data.get("isTemplate"),
        "isDisabled": repo_data.get("isDisabled"),
        "isFork": repo_data.get("isFork"),
        "createdAt": repo_data.get("createdAt"),
        "updatedAt": repo_data.get("updatedAt"),
        "pushedAt": repo_data.get("pushedAt"),
        "stargazerCount": repo_data.get("stargazerCount"),
        "forkCount": repo_data.get("forkCount"),
    }

    metrics["primaryLanguage"] = repo_data.get("primaryLanguage", {}).get("name") if repo_data.get("primaryLanguage") else None
    
    license_info = repo_data.get("licenseInfo") or {}
    metrics["license_name"] = license_info.get("name")
    metrics["license_spdxId"] = license_info.get("spdxId")
    
    owner_info = repo_data.get("owner") or {}
    metrics["owner_type"] = owner_info.get("__typename")
    metrics["owner_login"] = owner_info.get("login")

    metrics["watchers_count"] = repo_data.get("watchers", {}).get("totalCount")

    default_branch = repo_data.get("defaultBranchRef") or {}
    metrics["default_branch_name"] = default_branch.get("name")
    
    target = default_branch.get("target") or {}
    history = target.get("history") or {}
    metrics["total_commits_all_time"] = history.get("totalCount") 

    topics_list = [node.get("topic", {}).get("name") for node in repo_data.get("repositoryTopics", {}).get("nodes", [])]
    metrics["topics"] = ", ".join(filter(None, topics_list))

    if metrics["createdAt"]:
        created_dt = pd.to_datetime(metrics["createdAt"], utc=True)
        metrics["age_at_first_acf_days"] = max((first_acf_date - created_dt).days, 0)
    else:
        metrics["age_at_first_acf_days"] = pd.NA

    # 2. Fetch Historical Contributors (Pre-ACF)
    metrics["pre_acf_unique_contributors"] = get_pre_acf_contributors(owner, name, date_str)

    # 3. Fetch Historical Metrics using the REST Search API (Pre-ACF)
    time.sleep(2.5) 
    issues_query = f"repo:{repo} is:issue created:<{date_str}"
    metrics["pre_acf_total_issues"] = get_search_count(issues_query)
    
    time.sleep(2.5)
    prs_query = f"repo:{repo} is:pr created:<{date_str}"
    metrics["pre_acf_total_prs"] = get_search_count(prs_query)

    return metrics


def load_processed_repos():
    if not PROCESSED_TRACKER_FILE.exists():
        return set()
    with open(PROCESSED_TRACKER_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_processed_repo(repo):
    with open(PROCESSED_TRACKER_FILE, "a") as f:
        f.write(f"{repo}\n")


def append_to_csv(data_dict, file_path):
    df = pd.DataFrame([data_dict])
    write_header = not file_path.exists()
    df.to_csv(file_path, mode='a', header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Collect pre-ACF repository metrics via GraphQL.")
    parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    first_acf_dates = get_first_acf_dates()
    total_repos = len(first_acf_dates)
    print(f"Identified baseline ACF dates for {total_repos} repositories.")

    processed_repos = load_processed_repos()
    print(f"Found {len(processed_repos)} already processed. Resuming...\n")

    repos_to_process = {r: d for r, d in first_acf_dates.items() if r not in processed_repos}

    for repo, acf_date in tqdm(repos_to_process.items(), desc="Fetching Metrics", unit="repo"):
        try:
            metrics = fetch_repo_metadata(repo, acf_date)
            
            if metrics:
                append_to_csv(metrics, OUTPUT_METRICS_FILE)
            else:
                append_to_csv({"repo": repo, "first_acf_date": "NOT_FOUND_OR_ERROR"}, OUTPUT_METRICS_FILE)
            
            save_processed_repo(repo)
            
        except Exception as e:
            print(f"\n[!] Unexpected error processing {repo}: {e}")
            continue

    print(f"\nMetric collection complete. Data saved to {OUTPUT_METRICS_FILE}")


if __name__ == "__main__":
    main()