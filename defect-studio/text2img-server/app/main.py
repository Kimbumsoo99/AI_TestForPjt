# main.py
from fastapi import FastAPI, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from .model import generate_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.get("/generate-image/")
def create_image(
        prompt: str = Query(..., description="이미지를 생성할 텍스트 프롬프트"),
        num_inference_steps: int = Query(50, ge=1, le=100, description="추론 단계 수"),
        guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="가이던스 스케일")
):
    # Stable Diffusion을 통해 이미지 생성
    image = generate_image(prompt, num_inference_steps, guidance_scale)

    # 이미지 데이터를 바이트 스트림으로 변환하여 반환
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/png")
