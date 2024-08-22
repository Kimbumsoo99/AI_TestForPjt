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

const Input = styled.input`
  margin: 10px 0;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
`;

const Button = styled.button`
  margin: 10px 0;
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

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
  };

  const handleGenerateInpaint = async () => {
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
      const ctx = canvas.getContext("2d");
      ctx.beginPath();
      ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
      canvas.addEventListener("mousemove", draw);
    }
  };

  const draw = (e) => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"; // 반투명 흰색으로 설정
      ctx.lineWidth = 30;
      ctx.stroke();
    }
  };

  const stopDrawing = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.removeEventListener("mousemove", draw);
    }
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
    }
  };

  useEffect(() => {
    resizeImageAndCanvas();
  }, [imageSrc]);

  return (
    <Container>
      <h2>Inpainting</h2>
      <Input
        type="text"
        placeholder="프롬프트 입력"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <Input
        type="number"
        value={numInferenceSteps}
        onChange={(e) => setNumInferenceSteps(e.target.value)}
        min="1"
        max="100"
      />
      <Input
        type="number"
        value={guidanceScale}
        onChange={(e) => setGuidanceScale(e.target.value)}
        min="1.0"
        max="20.0"
        step="0.1"
      />
      <Input type="file" onChange={handleImageUpload} />
      {imageSrc && (
        <div style={{ position: "relative" }}>
          <img
            src={imageSrc}
            alt="Uploaded Image"
            ref={imgRef}
            style={{
              width: `${canvasWidth}px`,
              height: `${canvasHeight}px`,
              display: "block",
              position: "absolute",
              top: 0,
              left: 0,
            }}
            onLoad={resizeImageAndCanvas}
          />
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            style={{
              border: "2px solid #000",
              position: "absolute",
              top: 0,
              left: 0,
            }}
            onMouseDown={startDrawing}
            onMouseUp={stopDrawing}
          />
        </div>
      )}
      <Button onClick={handleGenerateInpaint}>Inpaint</Button>
    </Container>
  );
}

export default Inpaint;
