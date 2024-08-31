from PIL import Image
import numpy as np
import os

def convert_rgb_to_binary_grayscale(mask_image_path):
    # RGB 마스크 이미지 로드
    mask_image = Image.open(mask_image_path).convert("RGB")

    # NumPy 배열로 변환
    mask_np = np.array(mask_image)

    # 마스크 색상 변환: 검정색이 아닌 모든 픽셀을 흰색(255)으로 변경
    binary_mask_np = np.where(
        (mask_np != [0, 0, 0]).any(axis=2),  # 검정색이 아닌 모든 픽셀을 선택
        255,  # 흰색으로 변경
        0  # 검정색 그대로 유지
    ).astype(np.uint8)

    # 단일 채널 흑백 이미지로 변환
    binary_mask_image = Image.fromarray(binary_mask_np, mode='L')

    # 같은 경로에 덮어씌우기 위해 기존 파일 삭제
    if os.path.exists(mask_image_path):
        os.remove(mask_image_path)

    # 변환된 이미지를 원래 경로에 저장
    binary_mask_image.save(mask_image_path)

    print(f"Converted and saved binary grayscale mask image at {mask_image_path}")

# 마스크 이미지가 있는 폴더 경로 지정
folder_path = "C:/uploads/Test/cable"

# 폴더 내 모든 파일에 대해 작업 수행
for filename in os.listdir(folder_path):
    if filename.endswith('_rgb_mask.png'):  # '_rgb_mask.png'로 끝나는 파일만 선택
        mask_image_path = os.path.join(folder_path, filename)
        convert_rgb_to_binary_grayscale(mask_image_path)
