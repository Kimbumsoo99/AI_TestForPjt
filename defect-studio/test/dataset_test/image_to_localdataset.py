import os
from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Array3D, Array2D

# 이미지와 마스크를 불러오는 함수
def load_images_and_masks(instance_dir, mask_dir, prompt_file):
    instance_images = []
    mask_images = []
    prompts = []

    instance_image_files = sorted(os.listdir(instance_dir))
    mask_image_files = sorted(os.listdir(mask_dir))

    # 프롬프트를 불러오기
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    for img_file, mask_file, prompt in zip(instance_image_files, mask_image_files, prompts):
        instance_image = Image.open(os.path.join(instance_dir, img_file)).convert("RGB")
        mask_image = Image.open(os.path.join(mask_dir, mask_file)).convert("L")

        instance_images.append(instance_image)
        mask_images.append(mask_image)

    return instance_images, mask_images, prompts

# PIL 이미지를 배열로 변환하는 함수
def image_to_array(image, is_mask=False):
    if is_mask:
        return np.array(image, dtype=np.int64)  # 마스크는 단일 채널
    else:
        return np.array(image, dtype=np.int64)  # RGB 이미지는 3 채널

# 로컬에 저장된 이미지, 마스크, 프롬프트 파일 경로
train_instance_image_dir = "output/instance_image/train"
train_mask_image_dir = "output/mask_image/train"
train_prompt_file = "output/instance_prompt_train.txt"

test_instance_image_dir = "output/instance_image/test"
test_mask_image_dir = "output/mask_image/test"
test_prompt_file = "output/instance_prompt_test.txt"

# 이미지, 마스크, 프롬프트 불러오기
train_instance_images, train_mask_images, train_prompts = load_images_and_masks(
    train_instance_image_dir, train_mask_image_dir, train_prompt_file)
test_instance_images, test_mask_images, test_prompts = load_images_and_masks(
    test_instance_image_dir, test_mask_image_dir, test_prompt_file)

# 배열로 변환된 이미지와 마스크, 프롬프트를 사용해 Dataset 생성
train_data = {
    "prompt": train_prompts,
    "image": [image_to_array(img) for img in train_instance_images],
    "mask": [image_to_array(mask, is_mask=True) for mask in train_mask_images],
}

test_data = {
    "prompt": test_prompts,
    "image": [image_to_array(img) for img in test_instance_images],
    "mask": [image_to_array(mask, is_mask=True) for mask in test_mask_images],
}

# 데이터셋의 각 열에 대한 특징 정의
features = Features({
    "prompt": Value("string"),
    "image": Array3D(dtype="int64", shape=(512, 512, 3)),  # 512x512 RGB 이미지
    "mask": Array2D(dtype="int64", shape=(512, 512)),  # 512x512 단일 채널 마스크
})

# train, test Dataset 객체 생성
train_dataset = Dataset.from_dict(train_data, features=features)
test_dataset = Dataset.from_dict(test_data, features=features)

# DatasetDict로 train과 test를 결합
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 데이터셋 확인
print(dataset_dict)

# 데이터셋을 로컬에 저장
dataset_dict.save_to_disk("./output/combined_dataset")

print("train과 test 데이터셋이 성공적으로 생성 및 저장되었습니다.")
