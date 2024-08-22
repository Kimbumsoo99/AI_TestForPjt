import React, { useState, useRef, useEffect } from "react";
import { generateInpaint } from "../api";
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
  width: 100%;
  height: 100%;
`;

function Inpaint() {
  const [prompt, setPrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [canvasWidth, setCanvasWidth] = useState(512);
  const [canvasHeight, setCanvasHeight] = useState(512);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const inputFileRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  // 파일 업로드 핸들러
  const handleFileUploadClick = () => {
    if (!isImageUploaded) {
      inputFileRef.current.click();
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setIsImageUploaded(true);  // 이미지가 업로드되었음을 설정
    console.log("Image uploaded: ", file);
  };

  const handleGenerateInpaint = async () => {
    console.log("Generating Inpaint...");
    const canvas = canvasRef.current;
    if (canvas) {
      const maskBlob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));

      if (imageFile && maskBlob) {
        const imageUrl = await generateInpaint(
          prompt,
          imageFile,
          maskBlob,
          numInferenceSteps,
          guidanceScale
        );
        setImageSrc(imageUrl);
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
      ctx.strokeStyle = "rgba(255, 255, 0, 0.7)"; // 노란색 반투명 브러쉬
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

  const resizeImageAndCanvas = () => {
    const img = imgRef.current;
    if (img) {
      const maxDimension = 512;
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > maxDimension) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        }
      } else {
        if (height > maxDimension) {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      setCanvasWidth(width);
      setCanvasHeight(height);
      console.log("Resized canvas to: ", width, height);
    }
  };

  useEffect(() => {
    if (imageSrc) {
      resizeImageAndCanvas();
    }
  }, [imageSrc]);

  return (
    <Container>
      <h2>Inpainting</h2>
      <InputGroup>
        <Label htmlFor="prompt">Prompt:</Label>
        <Input
          id="prompt"
          type="text"
          placeholder="프롬프트 입력"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </InputGroup>
      <InputGroup>
        <Label htmlFor="steps">Inference Steps:</Label>
        <Input
          id="steps"
          type="number"
          value={numInferenceSteps}
          onChange={(e) => setNumInferenceSteps(e.target.value)}
          min="1"
          max="100"
        />
      </InputGroup>
      <InputGroup>
        <Label htmlFor="scale">Guidance Scale:</Label>
        <Input
          id="scale"
          type="number"
          value={guidanceScale}
          onChange={(e) => setGuidanceScale(e.target.value)}
          min="1.0"
          max="20.0"
          step="0.1"
        />
      </InputGroup>
      <Input
        type="file"
        ref={inputFileRef}
        onChange={handleImageUpload}
      />
      <UploadArea onClick={handleFileUploadClick} style={{ display: isImageUploaded ? 'none' : 'flex' }}>
        <p>이미지를 업로드하려면 클릭하세요</p>
      </UploadArea>
      {imageSrc && (
        <CanvasContainer style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px` }}>
          <img
            src={imageSrc}
            alt="Uploaded Image"
            ref={imgRef}
            style={{
              width: "100%",
              height: "100%",
              display: "block",
              position: "absolute",
              top: 0,
              left: 0,
              zIndex: 1,
            }}
            onLoad={resizeImageAndCanvas}
          />
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              zIndex: 2,
              cursor: "none", // 커서를 숨김
            }}
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
                backgroundColor: "rgba(255, 255, 0, 0.7)", // 브러쉬 색상
                borderRadius: "50%", // 원형 브러쉬
                pointerEvents: "none", // 클릭 이벤트를 무시
                zIndex: 3,
              }}
            />
          )}
        </CanvasContainer>
      )}
      <Button onClick={handleGenerateInpaint}>Inpaint</Button>
    </Container>
  );
}

export default Inpaint;
