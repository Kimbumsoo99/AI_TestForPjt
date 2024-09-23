from datasets import load_dataset

# 데이터셋을 로컬 디렉토리에 다운로드할 경로를 지정합니다.
local_cache_dir = "./huggingface_datasets"  # 원하는 로컬 디렉토리 경로로 변경 가능

# Hugging Face 데이터셋을 로컬에 다운로드
dataset = load_dataset("gzguevara/mr_potato_head_masked", cache_dir=local_cache_dir)

# 데이터셋 확인
print(dataset)