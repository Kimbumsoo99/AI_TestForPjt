# model.py
from PIL import Image
from clip_interrogator import Interrogator, Config

# CLIP-Interrogator 초기화
clip_config = Config(
    clip_model_name="ViT-L-14/openai",  # 사용할 CLIP 모델
    cache_path="./cache"                # 캐시 경로 설정 (필요한 경우)
)
clip_interrogator = Interrogator(clip_config)

# 이미지 분석 및 텍스트 프롬프트 생성 함수
def generate_prompt_from_image(image: Image):
    """
    주어진 이미지를 CLIP-Interrogator를 사용하여 텍스트 프롬프트 생성
    """
    prompt = clip_interrogator.interrogate(image)
    return prompt