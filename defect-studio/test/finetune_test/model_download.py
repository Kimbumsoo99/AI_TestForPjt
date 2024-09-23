from diffusers import StableDiffusionPipeline

# 모델 ID와 로컬에 저장할 경로 지정
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
local_model_path = "stable-diffusion-v1-5"

# 모델 다운로드 및 로컬 저장
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline.save_pretrained(local_model_path)

print(f"Model saved to: {local_model_path}")