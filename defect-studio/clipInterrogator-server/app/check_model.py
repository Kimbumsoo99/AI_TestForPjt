# check_model.py

from clip_interrogator import Interrogator, Config
from PIL import Image
import requests
from io import BytesIO
import torch

def initialize_clip_interrogator():
    try:
        print("Initializing CLIP Interrogator...")

        # GPU 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # CLIP-Interrogator 설정
        config = Config(
            clip_model_name="ViT-L-14/openai",
            cache_path="./cache",
            device=device  # 모델을 GPU에서 실행
        )

        interrogator = Interrogator(config)

        # 혼합 정밀도 사용 (VRAM 절약)
        if torch.cuda.is_available():
            interrogator.clip_model = interrogator.clip_model.half()  # 모델을 FP16으로 변환

        print("CLIP Interrogator initialized successfully.")
        return interrogator
    except Exception as e:
        print(f"Error initializing CLIP Interrogator: {e}")
        return None

def test_clip_interrogator(interrogator):
    try:
        url = "https://via.placeholder.com/150"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        print("Testing prompt generation...")

        # 혼합 정밀도를 적용하여 추론
        with torch.cuda.amp.autocast():
            prompt = interrogator.interrogate(image)
        print(f"Generated prompt: {prompt}")
    except torch.cuda.OutOfMemoryError as e:
        print("Ran out of VRAM:", e)
    except Exception as e:
        print(f"Error during prompt generation: {e}")

if __name__ == "__main__":
    interrogator = initialize_clip_interrogator()
    if interrogator:
        test_clip_interrogator(interrogator)
