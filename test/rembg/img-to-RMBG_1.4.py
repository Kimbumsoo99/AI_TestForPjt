import os
from transformers import pipeline
from PIL import Image
import torch
import time

# 입력 및 출력 디렉토리 설정
input_dir = "./real"  # 실제 입력 이미지 경로로 변경하세요
output_dir = "./mask_RMBG"  # 실제 출력 이미지 경로로 변경하세요

# 출력 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(output_dir, exist_ok=True)

# 타임 로그 파일 경로 설정
log_file_path = "./time_log.txt"

# GPU가 사용 가능한지 확인하고 디바이스 설정
device = 0 if torch.cuda.is_available() else -1

# 전체 작업 시간 기록 시작
overall_start_time = time.time()

# 모델 로드 전 시간 기록
model_load_start_time = time.time()

# 이미지 세그멘테이션 파이프라인 생성 (GPU 사용 설정)
rmbg_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)

# 모델 로드 후 시간 기록
model_load_end_time = time.time()

print("Background removal script is ready")

# 로그 파일에 기록 시작
with open(log_file_path, "w") as log_file:
    log_file.write(f"Model loaded in {model_load_end_time - model_load_start_time:.2f} seconds\n")

    # 입력 디렉토리의 모든 파일에 대해 배경 제거 수행
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # JPG와 PNG 둘 다 지원하도록 확장자 검사
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")  # PNG 형식으로 저장

            try:
                # 이미지 로드
                input_image = Image.open(input_image_path).convert("RGB")
                print(f"Processing image: {input_image_path}")

                # 이미지 처리 시간 기록 시작
                image_start_time = time.time()

                # GPU 사용량 기록 (처리 전)
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB 단위로 변환
                    gpu_memory_reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)  # MB 단위
                else:
                    gpu_memory_before = gpu_memory_reserved_before = 0

                # 배경 제거 수행
                output_image = rmbg_pipeline(input_image)

                # GPU 사용량 기록 (처리 후)
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB 단위로 변환
                    gpu_memory_reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)  # MB 단위
                else:
                    gpu_memory_after = gpu_memory_reserved_after = 0

                # 결과 저장 (PNG 형식)
                output_image.save(output_image_path, format="PNG")
                print(f"Output image saved at: {output_image_path}")

                # 이미지 처리 시간 기록 종료
                image_end_time = time.time()

                # 개별 이미지 처리 시간 및 GPU 사용량 로그에 기록
                log_file.write(f"Image {filename} processed in {image_end_time - image_start_time:.2f} seconds\n")
                log_file.write(f"GPU memory used before: {gpu_memory_before:.2f} MB, reserved before: {gpu_memory_reserved_before:.2f} MB\n")
                log_file.write(f"GPU memory used after: {gpu_memory_after:.2f} MB, reserved after: {gpu_memory_reserved_after:.2f} MB\n")
                log_file.flush()  # 즉시 파일에 기록

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                log_file.write(f"Error processing {filename}: {e}\n")
                log_file.flush()  # 즉시 파일에 기록

    # 전체 작업 시간 기록 종료
    overall_end_time = time.time()
    log_file.write(f"Overall processing time: {overall_end_time - overall_start_time:.2f} seconds\n")

print("All images processed.")
