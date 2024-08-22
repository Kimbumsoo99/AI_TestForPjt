# model.py
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
from PIL import Image


# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "CompVis/stable-diffusion-v1-4"

# Text2Img 파이프라인
t2i_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Img2Img 파이프라인
i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Inpaint 파이프라인
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print(f"Using device: {device}")

# t2i 함수
def generate_image(prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    image = t2i_pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

# i2i 함수
def generate_img2img(init_image: Image, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    try:
        generated_image = i2i_pipe(prompt=prompt, image=init_image, strength=0.75, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return generated_image
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None


def generate_inpaint(init_image: Image, mask_image: Image, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    try:
        generated_image = inpaint_pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return generated_image
    except Exception as e:
        print(f"Error during inpainting generation: {e}")
        return None

