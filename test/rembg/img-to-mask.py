import os
from PIL import Image

# 경로 설정
real_dir = './real'
mask_dir = './mask'
output_dir = './generated'

# output_dir이 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 파일 리스트 가져오기
real_images = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])
mask_images = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# 이미지 처리
for real_img_name, mask_img_name in zip(real_images, mask_images):
    real_img_path = os.path.join(real_dir, real_img_name)
    mask_img_path = os.path.join(mask_dir, mask_img_name)

    # 이미지 열기
    real_img = Image.open(real_img_path).convert("RGBA")  # 원본 이미지를 RGBA로 변환
    mask_img = Image.open(mask_img_path).convert("L")  # 마스크 이미지는 흑백 모드로 열기

    # 새로운 빈 이미지를 원본 이미지 크기로 생성 (투명한 배경)
    result_img = Image.new("RGBA", real_img.size, (0, 0, 0, 0))

    # 픽셀별로 마스크 적용 (흰색인 부분만 남김)
    for x in range(real_img.width):
        for y in range(real_img.height):
            if mask_img.getpixel((x, y)) > 128:  # 흰색 영역: 값이 128보다 큰 경우
                result_img.putpixel((x, y), real_img.getpixel((x, y)))  # 원본 이미지의 해당 픽셀을 유지

    # 결과 이미지 저장
    output_img_path = os.path.join(output_dir, real_img_name)
    result_img.save(output_img_path)

    print(f"Saved {output_img_path}")