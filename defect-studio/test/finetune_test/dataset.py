import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DefectDataset(Dataset):
    def __init__(self, good_image_dir, defect_image_dir, mask_image_dir, prompt_list, transform=None):
        self.good_image_dir = good_image_dir
        self.defect_image_dir = defect_image_dir
        self.mask_image_dir = mask_image_dir
        self.prompt_list = prompt_list
        self.transform = transform

        # good 이미지와 defect/mask 이미지들의 파일 목록을 가져옴
        self.image_files = sorted([f for f in os.listdir(good_image_dir) if f.endswith('.png')])
        self.defect_image_files = sorted([f for f in os.listdir(defect_image_dir) if f.endswith('.png')])
        self.mask_image_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith('_mask.png')])

        # 가장 작은 이미지 수를 기준으로 데이터셋 크기 설정
        self.dataset_size = min(len(self.image_files), len(self.defect_image_files), len(self.mask_image_files))

    def __len__(self):
        # 데이터셋 크기를 반환하도록 수정
        return self.dataset_size

    def __getitem__(self, idx):
        # 데이터셋 크기에 맞게 인덱스를 설정
        good_image_path = os.path.join(self.good_image_dir, self.image_files[idx % len(self.image_files)])
        defect_image_path = os.path.join(self.defect_image_dir, self.defect_image_files[idx])
        mask_image_path = os.path.join(self.mask_image_dir, self.mask_image_files[idx])

        # 이미지 로드
        good_image = Image.open(good_image_path).convert("RGB")
        defect_image = Image.open(defect_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")
        prompt = self.prompt_list[idx % len(self.prompt_list)]

        # 변환 적용
        if self.transform:
            good_image = self.transform(good_image)
            defect_image = self.transform(defect_image)
            mask_image = self.transform(mask_image)

        return good_image, defect_image, mask_image, prompt
