import React, { useState } from "react";
import { removeBackground } from "../api";
import styled from "styled-components";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  max-width: 700px;
  margin: auto;
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

const ImagePreview = styled.img`
  width: 512px;
  height: 512px;
  margin-top: 20px;
  border: 2px solid #333;
  object-fit: contain;
`;

function RemoveBg() {
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [outputImageSrc, setOutputImageSrc] = useState(null);
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setIsImageUploaded(true); // 이미지 업로드 상태를 설정
  };

  const handleFileUploadClick = () => {
    document.getElementById("fileInput").click(); // 파일 입력 클릭 이벤트 트리거
  };

  const handleRemoveBackground = async () => {
    if (imageFile) {
      const resultUrl = await removeBackground(imageFile);
      setOutputImageSrc(resultUrl);
    }
  };

  return (
    <Container>
      <h2>Remove Background</h2>
      <input type="file" id="fileInput" style={{ display: "none" }} onChange={handleImageUpload} />
      <UploadArea onClick={handleFileUploadClick}>
        {isImageUploaded ? (
          <ImagePreview src={imageSrc} alt="Uploaded Image" />
        ) : (
          <p>이미지를 업로드하려면 클릭하세요</p>
        )}
      </UploadArea>
      <Button onClick={handleRemoveBackground}>Remove Background</Button>
      {outputImageSrc && <ImagePreview src={outputImageSrc} alt="Output Image" />}
    </Container>
  );
}

export default RemoveBg;
