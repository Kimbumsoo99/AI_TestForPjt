from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

# Hugging Face 사용자 정보와 모델 로드
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token).to("cuda")
user = os.getenv("GITHUB_USER", "default_user")
repo = os.getenv("GITHUB_REPO", "default_repo")
branch = os.getenv("GITHUB_BRANCH", "develop")
file_name = f"{repo}_{branch}"

# 파일별 요약 정보를 읽어옴
with open(f"{file_name}_summary.json", "r", encoding="utf-8") as f:
    file_summaries = json.load(f)

# 모든 요약을 결합하여 이력서에 필요한 요소 추출
def extract_resume_elements_combined(file_summaries):
    # 파일 요약들을 하나의 문자열로 결합
    combined_summary_text = " ".join(summary for summary in file_summaries.values() if summary)

    # 이력서에 필요한 핵심 요소 추출
    prompt = (
        "The following is a combined summary of multiple code files. Extract key details relevant for a resume, "
        "focusing on purpose, main functions, technologies used, and any optimization or efficiency "
        "improvements across all files.\n\nCombined Summary:\n" + combined_summary_text + 
        "\n\nResponse:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    summary_ids = model.generate(inputs.input_ids, max_new_tokens=300)
    resume_elements = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return resume_elements

# 최종 요약 생성 함수
def generate_final_resume_summary(file_summaries):
    # 원래 요약 정보 결합
    combined_original_summary = " ".join(summary for summary in file_summaries.values() if summary)
    
    # 이력서용 핵심 요소 추출
    resume_elements = extract_resume_elements_combined(file_summaries)

    # 기존 요약 및 이력서 요소를 결합하여 최종 요약 생성
    prompt = (
        "Combine the following code summaries and resume-focused elements into a concise resume summary, "
        "emphasizing key features, technologies, and optimizations that would be relevant for a resume.\n\n"
        "Original Summaries:\n" + combined_original_summary + "\n\n" +
        "Resume Elements:\n" + resume_elements
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    final_summary_ids = model.generate(inputs.input_ids, max_new_tokens=300)
    final_resume_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
    
    return final_resume_summary

# 실행
final_resume_summary = generate_final_resume_summary(file_summaries)

# 최종 이력서 요약 저장
with open(f"{file_name}_Final_Resume_Summary.json", "w", encoding="utf-8") as f:
    json.dump({"resume_summary": final_resume_summary}, f, indent=4)

print("Final Resume Summary:", final_resume_summary)
