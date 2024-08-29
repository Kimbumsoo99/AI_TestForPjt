# model.py
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image


# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "runwayml/stable-diffusion-v1-5"

# Img2Img 파이프라인
i2i_pipe = (StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16))
i2i_pipe.to(device)

print(f"Using device: {device}")

# i2i 함수
def generate_img2img(init_image: Image, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    print(f"Generating {num_inference_steps} inference steps...")
    try:
        generated_image = i2i_pipe(prompt=prompt, image=init_image, strength=0.7, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return generated_image
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None
