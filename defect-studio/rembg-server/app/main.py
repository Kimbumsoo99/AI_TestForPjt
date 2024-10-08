from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
from io import BytesIO
import os
import time  # 작업 시간 측정을 위해 추가

app = FastAPI()

UPLOAD_FOLDER = "C:/uploads/rembg/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 세그멘테이션 파이프라인 생성 (GPU 사용 설정)
rmbg_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)

print("rmbg server is ready")


@app.post("/remove-bg/")
async def remove_background(image: UploadFile = File(...)):
    print("removing background image")
    try:
        start_time = time.time()  # 시작 시간 기록
        image_bytes = await image.read()
        input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        input_image_path = os.path.join(UPLOAD_FOLDER, "input_image.png")
        input_image.save(input_image_path)
        print(f"Input image saved at: {input_image_path}")

        output_image_path = os.path.join(UPLOAD_FOLDER, "output_image.png")

        # 배경 제거를 위한 이미지 처리
        output_image = rmbg_pipeline(input_image)

        # 결과 저장
        # 'result'가 이미지 객체인 경우 직접 저장
        output_image.save(output_image_path)
        print(f"Output image saved at: {output_image_path}")

        end_time = time.time()  # 종료 시간 기록
        print(f"Background removal took: {end_time - start_time} seconds")

        img_io = BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)

        # 수정된 부분: StreamingResponse에 BytesIO 객체 전달
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        print(f"Error during background removal: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
