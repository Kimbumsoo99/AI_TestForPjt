# model.py
from PIL import Image
from clip_interrogator import Interrogator, Config
import torch

# CLIP-Interrogator 초기화
clip_config = Config(
    clip_model_name="ViT-L-14/openai",  # 사용할 CLIP 모델
    cache_path="./cache",                # 캐시 경로 설정 (필요한 경우)
)
clip_config.apply_low_vram_defaults()  # 인스턴스 메서드로 호출
clip_interrogator = Interrogator(clip_config)

# GPU 메모리를 최적화하기 위해 사용할 수 있습니다.
if torch.cuda.is_available():
    clip_interrogator.clip_model = clip_interrogator.clip_model.half()  # 모델을 FP16으로 변환
    clip_interrogator.caption_model = clip_interrogator.caption_model.half()

# 이미지 분석 및 텍스트 프롬프트 생성 함수
def generate_prompt_from_image(image: Image):
    """
    주어진 이미지를 CLIP-Interrogator를 사용하여 텍스트 프롬프트 생성
    """
    prompt = clip_interrogator.interrogate(image)
    return prompt