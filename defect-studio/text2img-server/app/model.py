# model.py
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "CompVis/stable-diffusion-v1-4"

# Text2Img 파이프라인
t2i_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print(f"Using device: {device}")

# t2i 함수
def generate_image(prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    image = t2i_pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image
