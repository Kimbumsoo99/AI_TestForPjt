from diffusers import StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image
import numpy as np

# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 폴더가 존재하지 않는 경우 생성
output_dir = "C:/uploads/defect"
os.makedirs(output_dir, exist_ok=True)

# Inpaint 파이프라인
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to(device)

print(f"Using device: {device}")

def generate_cleanup(init_image: Image, mask_image: Image, prompt: str, num_inference_steps: int = 50,
                     guidance_scale: float = 7.5):
    print(
        f"generate_cleanup init_image: {init_image}, mask_image: {mask_image}, prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}")
    try:
        # inpainting 작업 수행
        temp_pipe_image = inpaint_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        print(f"temp_pipe_image_done")

        generated_image = temp_pipe_image.images[0]
        print(f"generated_image: {generated_image}")

        # 원본 이미지와 생성된 이미지를 NumPy 배열로 변환
        init_np = np.array(init_image)
        generated_np = np.array(generated_image)
        mask_np = np.array(mask_image.convert("L"))  # 마스크 이미지를 그레이스케일로 변환

        # 마스크의 흰색 영역(255)은 생성된 이미지로, 검은색 영역(0)은 원본 이미지로 남김
        combined_np = generated_np * (mask_np[..., None] > 127) + init_np * (mask_np[..., None] <= 127)

        # NumPy 배열을 다시 PIL 이미지로 변환
        final_image = Image.fromarray(combined_np.astype(np.uint8))

        # 파일명을 생성하여 이미지 저장
        output_file = os.path.join(output_dir, "cleanup_image.png")
        final_image.save(output_file)
        print(f"Image saved to: {output_file}")

        return final_image
    except Exception as e:
        print(f"Error during cleanup generation: {e}")
        return None
