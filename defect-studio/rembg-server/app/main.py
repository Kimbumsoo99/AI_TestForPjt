from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from rembg import remove
from io import BytesIO
from PIL import Image
import torch
import os

app = FastAPI()

# GPU 설정을 시도하고 실패하면 CPU로 설정
try:
    import onnxruntime

    # ONNX Runtime이 설치되어 있고 CUDA를 사용할 수 있는지 확인
    if torch.cuda.is_available():
        print("CUDA is available. Attempting to use GPU.")
        os.environ["ONNX_RUNTIME_USE_CUDA"] = "1"  # CUDA 사용을 설정
    else:
        print("CUDA is not available. Using CPU for rembg.")
        os.environ["ONNX_RUNTIME_USE_CUDA"] = "0"  # CPU로 강제 설정
except ImportError as e:
    # ONNX Runtime이 설치되어 있지 않으면 CPU로 강제 설정
    os.environ["ONNX_RUNTIME_USE_CUDA"] = "0"
    print("onnxruntime-gpu is not installed. Using CPU for rembg.")
    print(f"ImportError: {e}")



# rembg를 이용한 배경 제거 엔드포인트
@app.post("/remove-bg/")
async def remove_background(image: UploadFile = File(...)):
    try:
        # 업로드된 이미지를 읽고 PIL.Image로 변환
        image_bytes = await image.read()
        input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # 배경 제거 수행
        output_image = remove(input_image)

        # 결과 이미지를 바이트 스트림으로 변환하여 반환
        img_io = BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        print(f"Error during background removal: {e}")
        return {"error": str(e)}



# 추가 개선 사항 주석
# 1. 여러 이미지 형식 지원: JPEG, BMP 등 다양한 이미지 포맷을 지원하도록 추가 처리
# 2. 이미지 크기 제한: 지나치게 큰 이미지를 업로드할 경우 서버 성능에 영향을 미칠 수 있으므로, 이미지 크기를 제한하거나
#    업로드된 이미지의 해상도를 조정하는 기능 추가
# 3. 오류 처리 개선: 파일이 이미지가 아닌 경우, 지원되지 않는 파일 형식의 경우 등을 더 세분화하여 사용자에게 알림
# 4. 로깅 추가: 배경 제거 요청이 들어올 때마다 서버에서 처리된 이미지의 메타 데이터를 기록하여 추적 가능하도록 함
