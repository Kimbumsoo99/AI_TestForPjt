import os
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
import csv

# 이진화 함수 (thresholding)
def binarize_image(image, threshold=128):
    """이진화: 임계값을 기준으로 이미지를 이진화합니다."""
    binarized_image = (np.array(image) > threshold).astype(np.uint8)
    # 이진화된 이미지 확인용 로그 출력
    print(f"Binarized image: {np.sum(binarized_image == 1)} white pixels, {np.sum(binarized_image == 0)} black pixels")
    return binarized_image

# F1 스코어 계산 함수
def calculate_f1(mask_image, generated_image):
    """F1 스코어를 계산합니다."""
    # 이미지 이진화
    mask_image = binarize_image(mask_image)
    generated_image = binarize_image(generated_image)

    # Flatten the images for pixel-wise comparison
    mask_image_flat = mask_image.flatten()
    generated_image_flat = generated_image.flatten()

    # 빈 픽셀 체크
    if np.sum(mask_image_flat) == 0 and np.sum(generated_image_flat) == 0:
        print(f"No positive pixels in both images, skipping F1-score calculation.")
        return None  # F1-score 계산을 건너뜀

    # F1-score 계산
    return f1_score(mask_image_flat, generated_image_flat, average='binary')

# 이미지 비교 함수
def compare_images_and_save_f1(generated_dir, compare_dirs, csv_file_path):
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith(".png")]

    # CSV 파일 작성 시작
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image Name', 'Compare Dir', 'F1 Score'])  # CSV 헤더

        for compare_dir in compare_dirs:
            compare_files = [f for f in os.listdir(compare_dir) if f.endswith(".png")]

            # 이미지 크기 및 포맷 비교
            for generated_file, compare_file in zip(generated_files, compare_files):
                generated_path = os.path.join(generated_dir, generated_file)
                compare_path = os.path.join(compare_dir, compare_file)

                try:
                    # 이미지 로드 및 포맷 확인
                    generated_image = Image.open(generated_path)
                    compare_image = Image.open(compare_path)

                    print(f"Comparing {generated_file} with {compare_file} from {compare_dir}...")

                    # 이미지 크기 확인
                    if generated_image.size != compare_image.size:
                        print(f"Size mismatch: {generated_file} ({generated_image.size}) vs {compare_file} ({compare_image.size})")
                        continue

                    # F1 스코어 계산
                    f1 = calculate_f1(compare_image.convert("L"), generated_image.convert("L"))
                    if f1 is not None:
                        print(f"F1 Score for {generated_file} with {compare_dir}: {f1:.4f}")
                        csv_writer.writerow([generated_file, compare_dir, f1])  # CSV에 기록
                    else:
                        print(f"Skipped F1 Score calculation for {generated_file}")

                except Exception as e:
                    print(f"Error processing {generated_file} and {compare_file}: {e}")

# 경로 설정
generated_dir = "./generated"  # 원본 경로
compare_dirs = ["./mask_RMBG", "./mask_rembg_lib"]  # 비교할 디렉토리 목록
csv_file_path = "./f1_scores.csv"  # 결과 CSV 파일 경로

# 비교 수행 및 CSV 저장
compare_images_and_save_f1(generated_dir, compare_dirs, csv_file_path)
