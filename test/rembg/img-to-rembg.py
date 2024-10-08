import os
import time
from rembg import remove
from PIL import Image

# 입력 및 출력 디렉토리 설정
input_dir = './real'
output_dir = './mask_rembg_lib'

# 출력 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(output_dir, exist_ok=True)

# 타임 로그 파일 경로 설정
log_file_path = './time_log_rembg.txt'

# 전체 작업 시간 기록 시작
overall_start_time = time.time()

# 로그 파일에 기록 시작
with open(log_file_path, 'w') as log_file:
    log_file.write("RMBG Background Removal Process\n")

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

                # 배경 제거 수행
                output_image = remove(input_image)

                # 결과 저장 (PNG 형식)
                output_image.save(output_image_path, format="PNG")
                print(f"Output image saved at: {output_image_path}")

                # 이미지 처리 시간 기록 종료
                image_end_time = time.time()

                # 개별 이미지 처리 시간 로그에 기록
                log_file.write(f"Image {filename} processed in {image_end_time - image_start_time:.2f} seconds\n")
                log_file.flush()  # 즉시 파일에 기록

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                log_file.write(f"Error processing {filename}: {e}\n")
                log_file.flush()  # 즉시 파일에 기록

    # 전체 작업 시간 기록 종료
    overall_end_time = time.time()
    log_file.write(f"Overall processing time: {overall_end_time - overall_start_time:.2f} seconds\n")

print("All images processed.")
