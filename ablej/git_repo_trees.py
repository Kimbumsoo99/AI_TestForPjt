
import os
import requests
import json

# GitHub 정보 설정
github_user = os.getenv("GITHUB_USER", "your_github_username")
github_repo = os.getenv("GITHUB_REPO", "your_github_repo")
github_branch = os.getenv("GITHUB_BRANCH", "main")
github_token = os.getenv("GITHUB_TOKEN", "your_github_token")

# GitHub API를 통해 리포지토리 구조 한 번에 가져오기
def get_repo_structure_recursive(user, repo, branch="main", token=None):
    url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # 전체 응답 데이터를 JSON 파일로 저장하여 직접 확인
    repo_data = response.json()
    structure = {}

    for item in repo_data.get("tree", []):
        path_parts = item["path"].split("/")
        current = structure
        for part in path_parts[:-1]:
            current = current.setdefault(part, {})
        current[path_parts[-1]] = None if item["type"] == "blob" else {}
        
    return structure

# 압축된 형태로 폴더 구조를 한 줄로 표현
def compress_structure_inline(structure):
    compressed = []

    def recurse(struct, prefix=""):
        for key, value in struct.items():
            if isinstance(value, dict) and value:
                recurse(value, f"{prefix}{key}/")
            elif isinstance(value, dict):
                compressed.append(f"{prefix}{key}/[...]")
            else:
                compressed.append(f"{prefix}{key}")
    
    recurse(structure)
    return ", ".join(compressed)

# 실행: 전체 구조 가져오기
structure = get_repo_structure_recursive(github_user, github_repo, github_branch, github_token)

# 압축된 구조로 출력 및 저장
compressed_structure = compress_structure_inline(structure)

with open(f"{github_repo}_compressed_structure_inline.txt", "w", encoding="utf-8") as f:
    f.write(compressed_structure)

print("Compressed Repository Structure (Inline):")
print(compressed_structure)
