import os
import numpy as np
from PIL import Image
import csv

# 입력 디렉토리 설정
mask_dir = './generated'  # 원본 이미지 디렉토리
rmbg_dir = './mask_RMBG'  # 생성된 이미지 디렉토리 1
rembg_lib_dir = './mask_rembg_lib'  # 생성된 이미지 디렉토리 2

# CSV 파일 경로 설정
csv_file_path = './mae_evaluation.csv'


# MAE 계산 함수
def calculate_mae(image1, image2):
    image1_np = np.array(image1).astype(np.float32)
    image2_np = np.array(image2).astype(np.float32)

    # 두 이미지 크기가 같아야 하므로 resize 진행
    if image1_np.shape != image2_np.shape:
        image2_np = np.resize(image2_np, image1_np.shape)

    mae = np.mean(np.abs(image1_np - image2_np))
    return mae


# PNG 파일만 가져오는 함수
def get_png_files(directory):
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(".png")])


# MAE 평가를 진행하고 결과를 CSV에 기록하는 함수
def evaluate_and_save_mae(mask_dir, generated_dir1, generated_dir2, csv_file_path):
    mask_files = get_png_files(mask_dir)
    generated_files1 = get_png_files(generated_dir1)
    generated_files2 = get_png_files(generated_dir2)

    if len(mask_files) == 0 or len(generated_files1) == 0 or len(generated_files2) == 0:
        print("Error: One or more directories are empty or contain no PNG files.")
        return

    total_mae_1 = 0
    total_mae_2 = 0
    num_images = len(mask_files)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # CSV 파일에 헤더 작성
        csv_writer.writerow(['Image', 'MAE (mask_RMBG)', 'MAE (mask_rembg_lib)'])

        for i, mask_file in enumerate(mask_files):
            mask_image_path = os.path.join(mask_dir, mask_file)
            gen_image1_path = os.path.join(generated_dir1, generated_files1[i])
            gen_image2_path = os.path.join(generated_dir2, generated_files2[i])

            # 이미지 열기
            mask_image = Image.open(mask_image_path).convert('L')
            gen_image1 = Image.open(gen_image1_path).convert('L')
            gen_image2 = Image.open(gen_image2_path).convert('L')

            # MAE 계산
            mae_1 = calculate_mae(mask_image, gen_image1)
            mae_2 = calculate_mae(mask_image, gen_image2)

            # 결과를 CSV 파일에 기록
            csv_writer.writerow([mask_file, mae_1, mae_2])

            total_mae_1 += mae_1
            total_mae_2 += mae_2

            # 10장마다 MAE 평균 출력
            if (i + 1) % 10 == 0:
                avg_mae_1 = total_mae_1 / (i + 1)
                avg_mae_2 = total_mae_2 / (i + 1)
                print(f"Average MAE for {i + 1} images (mask_RMBG): {avg_mae_1:.4f}")
                print(f"Average MAE for {i + 1} images (mask_rembg_lib): {avg_mae_2:.4f}")

        # 전체 MAE 평균 계산 및 출력
        overall_avg_mae_1 = total_mae_1 / num_images
        overall_avg_mae_2 = total_mae_2 / num_images

        csv_writer.writerow([])
        csv_writer.writerow(['Overall Average MAE', overall_avg_mae_1, overall_avg_mae_2])

        print(f"Overall Average MAE (mask_RMBG): {overall_avg_mae_1:.4f}")
        print(f"Overall Average MAE (mask_rembg_lib): {overall_avg_mae_2:.4f}")


# MAE 평가 실행
evaluate_and_save_mae(mask_dir, rmbg_dir, rembg_lib_dir, csv_file_path)
