# main.py
from fastapi import FastAPI, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
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

def make_image_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

@app.get("/generate-image/")
def create_image(
        prompt: str = Query(..., description="이미지를 생성할 텍스트 프롬프트"),
        num_inference_steps: int = Query(50, ge=1, le=100, description="추론 단계 수"),
        guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="가이던스 스케일"),
        batch_size: int = Query(1, ge=1, le=10, description="한 번의 호출에서 생성할 이미지 수"),
        batch_count: int = Query(1, ge=1, le=10, description="호출할 횟수"),
        seed: int = Query(-1, description="이미지 생성 시 사용할 시드 값 (랜덤 시드: -1)")
):
    print(f"prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}")
    # Stable Diffusion을 통해 이미지 생성
    images, metadata = generate_image(prompt, num_inference_steps, guidance_scale, batch_size, batch_count, seed)

    # 이미지를 하나의 그리드로 결합
    grid_image = make_image_grid(images, rows=batch_count, cols=batch_size)

    print(f"grid_image: {grid_image}")

    # 이미지 데이터를 바이트 스트림으로 변환하여 반환
    img_io = BytesIO()
    grid_image.save(img_io, 'PNG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/png")
