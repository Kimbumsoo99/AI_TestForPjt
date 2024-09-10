from diffusers import StableDiffusionPipeline

# 모델 ID와 로컬에 저장할 경로 지정
model_id = "stabilityai/stable-diffusion-2"
local_model_path = "/home/j-j11s001/project/bumsoo/dataset/dreambooth_output/stable-diffusion-2"

# 모델 다운로드 및 로컬 저장
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline.save_pretrained(local_model_path)

print(f"Model saved to: {local_model_path}")