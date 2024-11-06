import os
import requests
import json
import base64
import re
import torch
from transformers import AutoTokenizer, pipeline

# 환경 변수로부터 사용자, 리포지토리 이름, 브랜치 이름 가져오기
user = os.getenv("GITHUB_USER", "default_user")
repo = os.getenv("GITHUB_REPO", "default_repo")
branch = os.getenv("GITHUB_BRANCH", "develop")
print(f"User: {user}, Repository: {repo}, Branch: {branch}")

# Hugging Face 및 GitHub 토큰 설정
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token")
gh_token = os.getenv("GITHUB_TOKEN", "your_github_token")

# 모델 로드 및 pipeline 설정
model_name = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad_token이 없으면 [PAD] 추가
pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=1
)

# GitHub 리포지토리 파일 목록 가져오기
def get_repo_files(user, repo, token=None, branch="main"):
    url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"
    print(f"Fetching file list from URL: {url}")
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json().get("tree", [])
    
    code_files = [file['path'] for file in files if file['path'].endswith((".py", ".java", ".js", ".ts", ".md"))]
    dependency_files = [file['path'] for file in files if file['path'].endswith(("requirements.txt", "package.json", "build.gradle", "pom.xml"))]
    return code_files, dependency_files

# GitHub에서 파일 내용 가져오기 및 디코딩
def get_file_content(user, repo, file_path, token=None):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    file_content = response.json().get("content", "")
    try:
        content_bytes = base64.b64decode(file_content + '==')  
        content_text = content_bytes.decode("utf-8")
    except (base64.binascii.Error, UnicodeDecodeError) as e:
        print(f"Failed to decode {file_path}: {e}")
        return None
    return content_text

# 주요 코드 조각 추출
def extract_key_sections(content):
    key_sections = []
    imports = re.findall(r"(import\s+[^\n]+|from\s+[^\n]+\s+import\s+[^\n]+)", content)
    key_sections.extend(imports)
    definitions = re.findall(r"(class\s+[A-Za-z_][A-Za-z0-9_]*|def\s+[A-Za-z_][A-Za-z0-9_]*)", content)
    key_sections.extend(definitions)
    comments = re.findall(r"(#.*|//.*|/\*[\s\S]*?\*/)", content)
    key_sections.extend(comments)
    return "\n".join(key_sections)

# 긴 텍스트를 최대 길이로 분할하여 Code Llama 모델이 처리할 수 있게 함
def split_text(text, max_length=1024):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Code Llama 모델을 통해 코드 요약 생성
def analyze_code_content(content, file_path):
    # core_content = extract_key_sections(content)
    if not content:
        print("No key sections found; skipping file.")
        return None

    text_chunks = split_text(content, max_length=1024)
    summaries = []

    for chunk in text_chunks:
        prompt = (
            "The following is content for summarizing a code file. Please briefly summarize the main purpose, "
            "key features, and primary libraries used in the code, and explain how this code could be beneficial "
            "for writing a resume:\n" + chunk
        )

        # Pipeline을 사용하여 텍스트 생성
        generated = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=150
        )
        
        summaries.append(generated[0]['generated_text'])

    return " ".join(summaries)

# 의존성 파일 요약
def analyze_dependencies(user, repo, dependency_files, token=None):
    dependencies_summary = {}
    for dep_file in dependency_files:
        print(f"Analyzing dependency file: {dep_file}...")
        content = get_file_content(user, repo, dep_file, token)
        if content:
            dependencies_summary[dep_file] = content
    return dependencies_summary

# 리포지토리 분석 실행
def analyze_repository(user, repo, token=None, branch="main"):
    print(f"Starting analysis for repository: {repo} on branch: {branch}")
    code_files, dependency_files = get_repo_files(user, repo, token, branch)
    repo_summary = {}
    
    for file_path in code_files:
        try:
            print(f"Analyzing {file_path}...")
            content = get_file_content(user, repo, file_path, token)
            if content is None:
                print(f"Skipping {file_path} due to decoding issues.")
                continue
            summary = analyze_code_content(content, file_path)
            repo_summary[file_path] = summary
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")

    dependencies_summary = analyze_dependencies(user, repo, dependency_files, token)
    repo_summary['dependencies'] = dependencies_summary

    summary_filename = f"{repo}_{branch}_summary.json"
    with open(summary_filename, "w", encoding="utf-8") as json_file:
        json.dump(repo_summary, json_file, indent=4)
    print(f"Analysis complete. Summary saved to {summary_filename}")

# 실행 예시
analyze_repository(user, repo, gh_token, branch)
