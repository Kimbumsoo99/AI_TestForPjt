import os
from PIL import Image


# 디렉토리 생성 함수
def create_directories():
    os.makedirs("output/instance_image/train", exist_ok=True)
    os.makedirs("output/instance_image/test", exist_ok=True)
    os.makedirs("output/mask_image/train", exist_ok=True)
    os.makedirs("output/mask_image/test", exist_ok=True)


# 이미지 저장 함수
def save_images_to_disk(dataset, output_dir, split="train"):
    instance_image_dir = os.path.join(output_dir, "instance_image", split)
    mask_image_dir = os.path.join(output_dir, "mask_image", split)

    # 디렉토리 생성
    create_directories()

    # 프롬프트를 저장할 리스트
    prompts = []

    for i, example in enumerate(dataset[split]):
        # 이미지 및 마스크는 이미 PIL 이미지 객체이므로 직접 저장 가능
        instance_image = example['image']  # 이미 PIL 이미지 객체임
        mask_image = example['mask']  # 이미 PIL 이미지 객체임

        # 이미지와 마스크 저장
        instance_image.save(os.path.join(instance_image_dir, f"instance_{split}_{i + 1}.png"))
        mask_image.save(os.path.join(mask_image_dir, f"mask_{split}_{i + 1}.png"))

        # 프롬프트 저장
        prompts.append(example['prompt'])

    return prompts


# 프롬프트 저장 함수
# 프롬프트 저장 함수 (train, test 분리)
def save_prompts_to_txt(prompts, split, output_dir="output"):
    file_path = os.path.join(output_dir, f"instance_prompt_{split}.txt")
    with open(file_path, "w") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")


# 로컬에 저장된 데이터셋 불러오기
from datasets import load_from_disk

mr_potato_head_masked_local = load_from_disk("./output/mr_potato_head_masked_local")

# 이미지와 마스크를 저장하고 프롬프트 리스트 생성 (train 데이터)
train_prompts = save_images_to_disk(mr_potato_head_masked_local, "output", split="train")
save_prompts_to_txt(train_prompts, split="train")

# 이미지와 마스크를 저장하고 프롬프트 리스트 생성 (test 데이터)
test_prompts = save_images_to_disk(mr_potato_head_masked_local, "output", split="test")
save_prompts_to_txt(test_prompts, split="test")

print("Train, test 이미지와 마스크, 프롬프트가 성공적으로 저장되었습니다.")
