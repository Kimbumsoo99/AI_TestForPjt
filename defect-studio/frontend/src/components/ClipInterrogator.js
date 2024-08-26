import React, { useState } from "react";
import { generatePromptFromImage } from "../api";  // API 호출 함수를 새로 작성해야 합니다.
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

const PromptOutput = styled.div`
  margin-top: 20px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  background-color: #f9f9f9;
  max-width: 100%;
  text-align: center;
`;

function ClipInterrogator() {
  const [imageFile, setImageFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [generatedPrompt, setGeneratedPrompt] = useState("");
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setIsImageUploaded(true);
    console.log("Image uploaded: ", file);
  };

  const handleGeneratePrompt = async () => {
    if (imageFile) {
      try {
        const prompt = await generatePromptFromImage(imageFile);
        setGeneratedPrompt(prompt);
        console.log("Generated Prompt:", prompt);
      } catch (error) {
        console.error("Error generating prompt:", error);
        alert("프롬프트 생성 중 오류가 발생했습니다. 콘솔에서 자세한 정보를 확인하세요.");
      }
    }
  };

  return (
    <Container>
      <h2>CLIP Interrogator</h2>
      <Input type="file" onChange={handleImageUpload} />
      {imageSrc && <ImagePreview src={imageSrc} alt="Uploaded Preview" />}
      <Button onClick={handleGeneratePrompt}>프롬프트 생성</Button>
      {generatedPrompt && <PromptOutput>{generatedPrompt}</PromptOutput>}
    </Container>
  );
}

export default ClipInterrogator;
