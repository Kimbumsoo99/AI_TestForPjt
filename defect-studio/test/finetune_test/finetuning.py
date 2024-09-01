from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

class DefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace('.png', '_mask.png')) # 마스크 파일 경로

        # 원본 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 마스크 이미지 로드
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 데이터셋 준비
image_dir = "C:/uploads/Test/capsule/images"
mask_dir = "C:/uploads/Test/capsule/masks"
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
