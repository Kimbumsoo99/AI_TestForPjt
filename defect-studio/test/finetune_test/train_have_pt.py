from diffusers import StableDiffusionInpaintPipeline
import torch
from torchvision import transforms
from PIL import Image
import os

# Transform 정의 -> torch dataset의 이미지 형식 맞춤
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 모델 로드
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stabilityai/stable-diffusion-2-inpainting"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to(device)

# 가장 마지막 체크포인트 파일 로드
checkpoint_dir = "C:/uploads/train/output/v3"
checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])

# 마지막 체크포인트 파일을 불러옵니다.
last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
pipeline.unet.load_state_dict(torch.load(last_checkpoint_path))

# 평가 모드 설정 (이 코드는 StableDiffusionInpaintPipelineLegacy에는 필요 없지만, 다른 파이프라인에서는 사용할 수 있습니다)
pipeline.unet.eval()

# 테스트 이미지 로드
test_good_image = Image.open("C:/uploads/train/cable/001.png").convert("RGB")
test_mask_image = Image.open("C:/uploads/train/cable/001_mask.png").convert("L")

# 학습 시 사용한 프롬프트와 동일한 프롬프트 사용
test_prompt = "cable cut_outer_insulation"

# 이미지 변환
test_good_image = transform(test_good_image).unsqueeze(0).to(device)
test_mask_image = transform(test_mask_image).unsqueeze(0).to(device)

# 이미지 생성
with torch.no_grad():
    # 학습 시 사용한 프롬프트를 동일하게 사용
    generated_image = pipeline(prompt=test_prompt, image=test_good_image, mask_image=test_mask_image).images[0]

# 생성된 이미지 저장
output_path = "C:/uploads/train/output/001_cable_output.png"
generated_image.save(output_path)
print(f"Generated image saved at {output_path}")
