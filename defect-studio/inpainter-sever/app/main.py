# main.py
from fastapi import FastAPI, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from .model import generate_inpaint
from PIL import Image
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

@app.post("/generate-inpaint/")
async def create_inpaint(
        prompt: str = Form(..., description="이미지를 변환할 텍스트 프롬프트"),
        image: UploadFile = File(..., description="기본 이미지 파일"),
        mask: UploadFile = File(..., description="마스크 이미지 파일"),
        num_inference_steps: int = Form(50, ge=1, le=100, description="추론 단계 수"),
        guidance_scale: float = Form(7.5, ge=1.0, le=20.0, description="가이던스 스케일")
):
    try:
        # 업로드된 이미지를 PIL.Image로 변환
        image_bytes = await image.read()
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        init_image = init_image.resize((512, 512))

        # 업로드된 마스크를 PIL.Image로 변환
        mask_bytes = await mask.read()
        mask_image = Image.open(BytesIO(mask_bytes)).convert("RGB")
        mask_image = mask_image.resize((512, 512))

        # Inpainting 작업 수행
        generated_image = generate_inpaint(init_image, mask_image, prompt, num_inference_steps, guidance_scale)

        if generated_image is None:
            return {"error": "Inpainting failed"}

        # 이미지 데이터를 바이트 스트림으로 변환하여 반환
        img_io = BytesIO()
        generated_image.save(img_io, format='PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return {"error": str(e)}
