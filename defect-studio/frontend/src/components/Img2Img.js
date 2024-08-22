import React, { useState } from "react";
import { generateImg2Img } from "../api";
import styled from "styled-components";

// 스타일링 컴포넌트
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

const ImagePreview = styled.img`
  margin: 20px 0;
  max-width: 100%;
  height: auto;
  border: 2px solid #ddd;
  border-radius: 5px;
`;


function Img2Img() {
  const [prompt, setPrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setIsImageUploaded(true);  // 이미지가 업로드되었음을 설정
    console.log("Image uploaded: ", file);
  };

  const handleGenerateImg2Img = async () => {
    if (imageFile) {
      try {
        const imageUrl = await generateImg2Img(prompt, imageFile, numInferenceSteps, guidanceScale);
        if (imageSrc) {
          URL.revokeObjectURL(imageSrc); // 이전 Blob URL 해제
        }
        console.log("Generated Image URL:", imageUrl);
        setImageSrc(imageUrl);
      } catch (error) {
        console.error("Error generating image:", error);
        alert("이미지 생성 중 오류가 발생했습니다. 콘솔에서 자세한 정보를 확인하세요.");
      }
    }
  };

  return (
    <Container>
      <h2>Image-to-Image</h2>
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
      {previewSrc && <ImagePreview src={previewSrc} alt="Uploaded Preview" />}
      <Button onClick={handleGenerateImg2Img}>이미지 변환</Button>
      {imageSrc && <ImagePreview src={imageSrc} alt="Generated" />}
    </Container>
  );
}

export default Img2Img;
