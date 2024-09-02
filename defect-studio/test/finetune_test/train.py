from diffusers import StableDiffusionInpaintPipeline
import torch
from torch.utils.data import DataLoader
import os
from dataset import DefectDataset  # dataset.py에서 DefectDataset 클래스 가져오기
from torchvision import transforms

def main():
    # 데이터셋 준비
    image_dir = "C:/uploads/Test/cable/images"
    mask_dir = "C:/uploads/Test/cable/masks"


    # 모델 학습시에는 이미지 크기가 고정돼야합니다! 그래서 이렇게 리사이즈 시키는 것입니다.
    # 또한 텐서로 변환하는 것은 이미지를 수치 형식으로 표현해야 딥러닝이 가능하기 때문에 그런것입니다.
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 이미지와 마스크를 512x512로 리사이즈
        transforms.ToTensor()           # 이미지를 텐서로 변환
    ])

    dataset = DefectDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print(f"dataset size: {len(dataset)}, dataloader size: {len(dataloader)}")

    # 모델 로드
    model_id = "runwayml/stable-diffusion-inpainting"  # Stable Diffusion 인페인팅 모델 ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline.to(device)

    # 이미지 생성 루프
    output_dir = "C:/uploads/Test/output"
    os.makedirs(output_dir, exist_ok=True)

    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            # 모델을 사용하여 이미지 생성
            output = pipeline(prompt="Add a scratch to the screen", image=images, mask_image=masks).images

        # 생성된 이미지를 저장
        for j, img in enumerate(output):
            img.save(os.path.join(output_dir, f"output_{i * dataloader.batch_size + j:03d}.png"))

        print(f"Processed batch {i + 1}/{len(dataloader)}")

    print("Image generation completed.")

if __name__ == "__main__":
    main()
