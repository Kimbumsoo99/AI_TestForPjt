import os
import requests
import openai
import json
import tiktoken

# 사용할 모델에 맞는 토크나이저 선택
encoding = tiktoken.encoding_for_model("gpt-4")  # 또는 gpt-3.5-turbo 등

# 환경 변수로부터 OpenAI API 키와 GitHub 정보 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
user = os.getenv("GITHUB_USER", "default_user")
repo = os.getenv("GITHUB_REPO", "default_repo")
branch = os.getenv("GITHUB_BRANCH", "develop")

# OpenAI API 설정
openai.api_key = openai_api_key

# OpenAI를 통한 요약 생성 함수
def generate_openai_summary(content, directory_structure, example_summary):
    # 프롬프트 텍스트 정의
    prompt = (
        f"다음 프로젝트의 내용을 기반으로 이력서 작성에 도움이 되는 프로젝트 요약을 작성해줘.\n\n"
        f"1. 프로젝트 요약\n2. 사용 기술\n3. 핵심 기능과 서비스의 강점\n\n"
        f"디렉터리 구조:\n{directory_structure}\n\n"
        f"프로젝트 파일 요약:\n{example_summary}\n\n"
        f"참고:\n{content}\n\n"
    )
    print(f"prompt: {prompt}")

    tokens = encoding.encode(prompt)
    token_count = len(tokens)
    
    print("토큰 수:", token_count)

    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "응답은 한글로 작성해 주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5,
        n=1,
        stop=None
    )

    print(response)
    
    # 결과에서 요약 텍스트 추출
    summary_text = response.choices[0]['message']['content'].strip()
    return summary_text

# 디렉터리 구조와 요약을 JSON 파일에서 로드
def load_project_data(summary_file, directory_file):
    with open(summary_file, "r", encoding="utf-8") as f:
        example_summary = json.load(f)
    with open(directory_file, "r", encoding="utf-8") as f:
        directory_structure = f.read()
    return example_summary, directory_structure

# 파일별 요약 생성 실행
def analyze_project_for_resume(summary_file, directory_file, file_name):
    example_summary, directory_structure = load_project_data(summary_file, directory_file)
    
    # 프로젝트 요약 생성
    resume_summary = generate_openai_summary(
        content="",
        directory_structure=directory_structure,
        example_summary=json.dumps(example_summary, indent=4, ensure_ascii=False)
    )

    # 결과 저장
    with open(f"{file_name}_resume_korean.json", "w", encoding="utf-8") as f:
        json.dump({"resume_summary": resume_summary}, f, ensure_ascii=False, indent=4)

    print("Final Resume Summary:", resume_summary)

# 실행 예시
# 'summary.json'과 'directory_structure.txt' 파일은 사전 준비된 예시 요약 및 디렉터리 구조를 담고 있는 파일입니다.
analyze_project_for_resume(f"{repo}_{branch}_summary.json", f"{repo}_compressed_structure_inline.txt", f"{user}_{repo}")