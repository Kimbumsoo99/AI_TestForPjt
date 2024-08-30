import React, { useState, useRef, useEffect } from "react";
import { generateCleanup } from "../api"; // 수정된 함수 사용
import styled from "styled-components";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  max-width: 700px;
  margin: auto;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
`;

const Label = styled.label`
  margin-bottom: 5px;
  font-weight: bold;
`;

const Input = styled.input`
  margin-bottom: 10px;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
`;

const UploadArea = styled.div`
  width: 512px;
  height: 512px;
  border: 2px dashed #007bff;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin-bottom: 20px;
  position: relative;
  background-color: #f8f9fa;
`;

const Button = styled.button`
  margin-top: 20px;
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  &:hover {
    background-color: #0056b3;
  }
`;

const CanvasContainer = styled.div`
  position: relative;
  width: 512px;
  height: 512px;
  margin-top: 20px;
  border: 2px solid #333;
  background-color: #ffffff;
`;

const StyledImage = styled.img`
  width: 100%;
  height: 100%;
  display: block;
  object-fit: contain;
`;

const StyledCanvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  cursor: none;
`;

const ImagePreview = styled.img`
  width: 512px;
  height: 512px;
  margin-top: 20px;
  border: 2px solid #333;
  object-fit: contain;
`;

function Cleanup() {
  const [prompt, setPrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [maskSrc, setMaskSrc] = useState(null);
  const [cleanedImageSrc, setCleanedImageSrc] = useState(null); // 생성된 이미지를 위한 상태 추가
  const [canvasWidth, setCanvasWidth] = useState(512);
  const [canvasHeight, setCanvasHeight] = useState(512);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const inputFileRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  useEffect(() => {
    if (imageSrc && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // 캔버스를 초기화
        ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight); // 이미지를 캔버스에 그립니다.
      };
      img.src = imageSrc;
    }
  }, [imageSrc, canvasWidth, canvasHeight]);

  const handleFileUploadClick = () => {
    if (!isImageUploaded) {
      inputFileRef.current.click();
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setIsImageUploaded(true);
    console.log("Image uploaded: ", file);
  };

  const handleGenerateCleanup = async () => {
    // 함수 이름 변경
    console.log("Generating Cleanup...");

    const canvas = canvasRef.current;
    if (canvas) {
      // 마스크 이미지를 생성합니다.
      const ctx = canvas.getContext("2d");
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;
      const maskCtx = maskCanvas.getContext("2d");

      // 캔버스의 그림을 가져옵니다.
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // 모든 픽셀을 검사하여 색칠된 부분만 흰색으로 바꾸고 나머지는 검은색으로 유지합니다.
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] === 200 && data[i + 1] === 200 && data[i + 2] === 200) {
          // 나머지 부분을 흰색으로 설정
          data[i] = 255;
          data[i + 1] = 255;
          data[i + 2] = 255;
          data[i + 3] = 1;
        } else {
          // 색칠된 부분을 검정색으로 설정
          data[i] = 0;
          data[i + 1] = 0;
          data[i + 2] = 0;
          data[i + 3] = 1;
        }
      }

      // 마스크 이미지 생성
      maskCtx.putImageData(imageData, 0, 0);

      // 마스크 캔버스를 Blob으로 변환합니다.
      const maskBlob = await new Promise((resolve) => maskCanvas.toBlob(resolve, "image/png"));

      // 생성된 마스크를 미리보기 위해 Blob URL로 변환합니다.
      setMaskSrc(URL.createObjectURL(maskBlob));
      console.log("Mask Image URL:", maskSrc);

      // 원본 이미지와 마스크 이미지를 함께 서버로 보냅니다.
      if (imageFile && maskBlob) {
        try {
          const imageUrl = await generateCleanup(
            // 함수 이름 변경
            imageFile,
            maskBlob,
          );
          setCleanedImageSrc(imageUrl); // 생성된 이미지를 별도로 저장
        } catch (error) {
          console.error("Error generating cleanup:", error);
        }
      }
    }
  };

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    if (canvas) {
      setIsDrawing(true);
      const ctx = canvas.getContext("2d");
      ctx.beginPath();
      ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
      console.log("Start drawing at: ", e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
      ctx.globalCompositeOperation = "source-over"; // 덧칠 방지
      ctx.strokeStyle = "rgba(200, 200, 200)"; // 노란색 반투명 브러쉬
      ctx.lineWidth = 30; // 브러쉬 크기
      ctx.lineCap = "round"; // 부드러운 브러쉬
      ctx.stroke();
      setMousePosition({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY });
      console.log("Drawing at: ", e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    console.log("Stop drawing");
  };

  return (
    <Container>
      <h2>Cleanup</h2>
      <Input type="file" ref={inputFileRef} onChange={handleImageUpload} />
      <UploadArea
        onClick={handleFileUploadClick}
        style={{ display: isImageUploaded ? "none" : "flex" }}
      >
        <p>이미지를 업로드하려면 클릭하세요</p>
      </UploadArea>
      {imageSrc && (
        <>
          <CanvasContainer>
            {/* 배경에 원본 이미지를 둡니다 */}
            <StyledImage src={imageSrc} alt="Uploaded Image" ref={imgRef} />
            {/* 투명 캔버스에만 그림을 그립니다 */}
            <StyledCanvas
              ref={canvasRef}
              width={canvasWidth}
              height={canvasHeight}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
            {isDrawing && (
              <div
                style={{
                  position: "absolute",
                  top: `${mousePosition.y - 15}px`, // 브러쉬 크기의 반만큼 이동
                  left: `${mousePosition.x - 15}px`,
                  width: "30px",
                  height: "30px",
                  backgroundColor: "rgba(200, 200, 200)", // 브러쉬 색상
                  borderRadius: "50%", // 원형 브러쉬
                  pointerEvents: "none", // 클릭 이벤트를 무시
                  zIndex: 3,
                }}
              />
            )}
          </CanvasContainer>
          {/* {maskSrc && <ImagePreview src={maskSrc} alt="Mask Preview" />} */}
        </>
      )}
      {cleanedImageSrc && <ImagePreview src={cleanedImageSrc} alt="Cleaned Image" />}
      <Button onClick={handleGenerateCleanup}>Cleanup</Button>
    </Container>
  );
}

export default Cleanup;
