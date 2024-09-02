from dataset import DefectDataset
from diffusers import StableDiffusionInpaintPipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

# Transform 정의 -> torch dataset의 이미지 형식 맞춤
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 데이터 경로 정의
# good image
good_image_dir = "C:/DefectStudio/test_cable/test/good"
# defect(불량 데이터) image
defect_image_dir = "C:/DefectStudio/test_cable/test/cut_outer_insulation"
# mask area image
mask_image_dir = "C:/DefectStudio/test_cable/ground_truth/cut_outer_insulation"
prompt_list = ["cable cut_outer_insulation"] * 9  # 이 예에서는 프롬프트가 고정되어 있다고 가정합니다.

dataset = DefectDataset(good_image_dir, defect_image_dir, mask_image_dir, prompt_list, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 모델 학습
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stabilityai/stable-diffusion-2-inpainting"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(pipeline.unet.parameters(), lr=1e-5)

# 학습 루프
num_epochs = 4
output_dir = "C:/uploads/train/output/v3"
# 폴더 생성 exist_ok=True는 경로가 존재하면 넘어가는 옵션
os.makedirs(output_dir, exist_ok=True)

# PIL 이미지를 텐서로 변환하기 위한 transform
pil_to_tensor = transforms.ToTensor()

# 학습 시작
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (good_image, defect_image, mask_image, prompt) in enumerate(dataloader):
        print(f"Epoch {epoch + 1}, Step {i + 1}")
        batch_size = good_image.size(0)

        # 튜플 형태로 나온 프롬프트를 리스트로 변환
        prompt_list_batch = list(prompt)

        for j in range(batch_size):
            good_image_sample = good_image[j].unsqueeze(0).to(device)
            defect_image_sample = defect_image[j].unsqueeze(0).to(device)
            mask_image_sample = mask_image[j].unsqueeze(0).to(device)
            prompt_sample = prompt_list_batch[j]  # 문자열 형태로 처리된 프롬프트

            optimizer.zero_grad()

            generated_images = pipeline(prompt=prompt_sample, image=good_image_sample,
                                        mask_image=mask_image_sample).images

            # PIL 이미지를 텐서로 변환하고 requires_grad=True 설정
            output_tensor = pil_to_tensor(generated_images[0]).unsqueeze(0).to(device)
            output_tensor.requires_grad_(True)

            # 손실 계산
            loss = criterion(output_tensor, defect_image_sample)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}] Completed, Average Loss: {epoch_loss / len(dataloader):.4f}")

    # 모델 체크포인트 저장
    checkpoint_path = os.path.join(output_dir, f"v3_stable_diffusion_checkpoint_epoch_{epoch + 1}.pt")
    torch.save(pipeline.unet.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

print("Training completed!")

# 테스트 이미지 로드
test_good_image = Image.open("C:/uploads/train/cable/001.png").convert("RGB")
test_mask_image = Image.open("C:/uploads/train/cable/001_mask.png").convert("L")
test_prompt = "cable cut_outer_insulation"

# 이미지 변환
test_good_image = transform(test_good_image).unsqueeze(0).to(device)
test_mask_image = transform(test_mask_image).unsqueeze(0).to(device)

# 이미지 생성
with torch.no_grad():
    generated_image = pipeline(prompt=test_prompt, image=test_good_image, mask_image=test_mask_image).images[0]

# 생성된 이미지 저장
generated_image.save("C:/uploads/train/output/001_cable_output.png")
print("Generated image saved at C:/uploads/train/output/001_cable_output.png")
