# model.py
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
import os
from PIL import Image

# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id = "runwayml/stable-diffusion-v1-5"


# 폴더가 존재하지 않는 경우 생성
output_dir = "C:/uploads/defect"
os.makedirs(output_dir, exist_ok=True)

# Inpaint 파이프라인
# inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
# inpaint_pipe.safety_checker = lambda images, **kwargs: (images, False)  # 안전 검사 비활성화 (NSFW 콘텐츠)

# inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
# )
inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)

inpaint_pipe = inpaint_pipe.to("cuda")
# inpaint_pipe.safety_checker = lambda images, **kwargs: (images, False)  # 안전 검사 비활성화 (NSFW 콘텐츠)

print(f"Using device: {device}")

def generate_inpaint(init_image: Image, mask_image: Image, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    print(f"generate_inpaint init_image: {init_image}, mask_image: {mask_image}, prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}")
    try:
        # 이미지를 올바른 형태로 변환
        temp_pipe_image = inpaint_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        # 위에 부분에서 오류가 나타나는 것 같아.
        print(f"temp_pipe_image_done")
        print(f"temp_pipe_image: {temp_pipe_image}")
        print(f"temp_pipe_image.images: {temp_pipe_image.images}")
        generated_image = temp_pipe_image.images[0]


        print(f"generated_image: {generated_image}")

        # 파일명을 생성하여 이미지 저장
        output_file = os.path.join(output_dir, "generated_image.png")
        generated_image.save(output_file)
        print(f"Image saved to: {output_file}")

        return generated_image
    except Exception as e:
        print(f"Error during inpainting generation: {e}")
        return None