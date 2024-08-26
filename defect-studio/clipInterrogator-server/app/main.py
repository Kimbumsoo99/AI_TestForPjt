# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from .model import generate_prompt_from_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (개발 중에는 localhost 등 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.post("/generate-prompt/")
async def create_prompt_from_image(image: UploadFile = File(...)):
    """
    업로드된 이미지로부터 텍스트 프롬프트 생성
    """
    try:
        # 업로드된 이미지를 읽고 PIL로 변환
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # CLIP-Interrogator를 통해 프롬프트 생성
        prompt = generate_prompt_from_image(pil_image)

        # 결과를 JSON 형식으로 반환
        return JSONResponse(content={"generated_prompt": prompt})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
