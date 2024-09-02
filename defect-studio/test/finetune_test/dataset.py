from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class DefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # python 문법: .png로 끝나는 파일만 필터링하여 리스트 생성 후 정렬
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace('.png', '_mask.png'))  # 마스크 파일 경로

        # 원본 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 마스크 이미지 로드
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
