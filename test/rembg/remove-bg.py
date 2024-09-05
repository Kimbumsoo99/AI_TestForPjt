import os
from transformers import pipeline
from PIL import Image
from io import BytesIO
import torch
import time

# 입력 및 출력 디렉토리 설정
input_dir = "C:/DefectStudio/testing_docs/remove_bg/origin/"  # 실제 입력 이미지 경로로 변경하세요
output_dir = "C:/DefectStudio/testing_docs/remove_bg/test/"  # 실제 출력 이미지 경로로 변경하세요

# 출력 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(output_dir, exist_ok=True)

# GPU가 사용 가능한지 확인하고 디바이스 설정
device = 0 if torch.cuda.is_available() else -1

# 이미지 세그멘테이션 파이프라인 생성 (GPU 사용 설정)
rmbg_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)

print("Background removal script is ready")

# 입력 디렉토리의 모든 파일에 대해 배경 제거 수행
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # JPG와 PNG 둘 다 지원하도록 확장자 검사 수정
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")  # PNG 형식으로 저장

        try:
            # 이미지 로드
            input_image = Image.open(input_image_path).convert("RGB")
            print(f"Processing image: {input_image_path}")

            start_time = time.time()  # 시작 시간 기록

            # 배경 제거 수행
            output_image = rmbg_pipeline(input_image)

            # 결과 저장 (PNG 형식)
            output_image.save(output_image_path, format="PNG")
            print(f"Output image saved at: {output_image_path}")

            end_time = time.time()  # 종료 시간 기록
            print(f"Background removal for {filename} took: {end_time - start_time} seconds")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("All images processed.")