# model.py
from diffusers import StableDiffusionPipeline
import torch
import random
from PIL import Image


# Stable Diffusion 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "runwayml/stable-diffusion-v1-5"

# Text2Img 파이프라인
t2i_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print(f"Using device: {device}")

# t2i 함수
def generate_image(prompt: str,
                   height: int = 512,
                   width: int = 512,
                   num_inference_steps: int = 50,
                   guidance_scale: float = 7.5,
                   negative_prompt: str = None,
                   num_images_per_prompt: int = 1,
                   batch_size: int = 1,
                   batch_count: int = 1,
                   seed: int = -1):
    # 시드가 -1이면 랜덤 시드 생성
    if seed == -1:
        seed = random.randint(0, 2 ** 32 - 1)

    all_images = []
    metadata = []

    # 전체 반복 횟수 계산
    total_images = batch_size * batch_count

    # 고유한 시드를 각각의 이미지에 할당
    seeds = [seed + i for i in range(total_images)]

    # 각 배치의 이미지 생성
    for i in range(batch_count):
        # 현재 배치에 사용될 시드 설정
        current_seeds = seeds[i * batch_size: (i + 1) * batch_size]

        # 생성할 이미지 개수만큼의 랜덤 생성기를 미리 준비
        generators = [torch.Generator(device=device).manual_seed(s) for s in current_seeds]

        # 한 번의 호출로 batch_size만큼의 이미지 생성
        images = t2i_pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generators
        ).images

        all_images.extend(images)

        # 메타데이터 저장
        for j in range(batch_size):
            metadata.append({
                'batch': i,
                'image_index': j,
                'seed': current_seeds[j],
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'prompt': prompt
            })

    # image = t2i_pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    print(f"metadata: {metadata}")
    print(f"all_images: {all_images}")
    return all_images, metadata
