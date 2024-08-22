# model.py
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "CompVis/stable-diffusion-v1-4"

# Inpaint 파이프라인
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print(f"Using device: {device}")

def generate_inpaint(init_image: Image, mask_image: Image, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    try:
        # 이미지를 올바른 형태로 변환
        generated_image = inpaint_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        return generated_image
    except Exception as e:
        print(f"Error during inpainting generation: {e}")
        return None
