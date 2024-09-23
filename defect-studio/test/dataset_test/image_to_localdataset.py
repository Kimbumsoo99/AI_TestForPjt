import os
from PIL import Image
import numpy as np
from datasets import Dataset, Features, Value, Array3D, Array2D

# 이미지와 마스크를 불러오는 함수
def load_images_and_masks(instance_dir, mask_dir):
    instance_images = []
    mask_images = []

    instance_image_files = sorted(os.listdir(instance_dir))
    mask_image_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(instance_image_files, mask_image_files):
        instance_image = Image.open(os.path.join(instance_dir, img_file)).convert("RGB")
        mask_image = Image.open(os.path.join(mask_dir, mask_file)).convert("L")

        instance_images.append(instance_image)
        mask_images.append(mask_image)

    return instance_images, mask_images

# 프롬프트를 불러오는 함수
def load_prompts(prompt_file):
    with open(prompt_file, "r") as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts]

# 로컬에 저장된 이미지, 마스크, 프롬프트 파일 경로
instance_image_dir = "output/instance_image"
mask_image_dir = "output/mask_image"
prompt_file = "output/instance_prompt.txt"

# 이미지, 마스크, 프롬프트 불러오기
instance_images, mask_images = load_images_and_masks(instance_image_dir, mask_image_dir)
prompts = load_prompts(prompt_file)

# PIL 이미지를 배열로 변환하는 함수
def image_to_array(image, is_mask=False):
    if is_mask:
        return np.array(image, dtype=np.int64)  # 마스크는 단일 채널
    else:
        return np.array(image, dtype=np.int64)  # RGB 이미지는 3 채널

# 배열로 변환된 이미지와 마스크, 프롬프트를 사용해 Dataset 생성
data = {
    "prompt": prompts,
    "image": [image_to_array(img) for img in instance_images],
    "mask": [image_to_array(mask, is_mask=True) for mask in mask_images],
}

# 데이터셋의 각 열에 대한 특징 정의
features = Features({
    "prompt": Value("string"),
    "image": Array3D(dtype="int64", shape=(512, 512, 3)),  # 512x512 RGB 이미지
    "mask": Array2D(dtype="int64", shape=(512, 512)),  # 512x512 단일 채널 마스크
})

# Dataset 객체 생성
dataset = Dataset.from_dict(data, features=features)

# 데이터셋 확인
print(dataset)

# 데이터셋을 로컬에 저장
dataset.save_to_disk("./output/combined_dataset")

print("데이터셋이 성공적으로 생성 및 저장되었습니다.")
