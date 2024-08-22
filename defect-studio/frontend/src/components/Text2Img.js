import React, { useState } from 'react';
import { generateImage } from '../api';
import styled from 'styled-components';

// 스타일링 컴포넌트
const Container = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    max-width: 500px;
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
    background-color: #007BFF;
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

function Text2Img() {
    const [prompt, setPrompt] = useState('');
    const [numInferenceSteps, setNumInferenceSteps] = useState(50);
    const [guidanceScale, setGuidanceScale] = useState(7.5);
    const [imageSrc, setImageSrc] = useState(null);

    const handleGenerateImage = async () => {
        const imageUrl = await generateImage(prompt, numInferenceSteps, guidanceScale);
        setImageSrc(imageUrl);
    };

    return (
        <Container>
            <h2>Text-to-Image</h2>
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
            <Button onClick={handleGenerateImage}>이미지 생성</Button>
            {imageSrc && <ImagePreview src={imageSrc} alt="Generated" />}
        </Container>
    );
}

export default Text2Img;
