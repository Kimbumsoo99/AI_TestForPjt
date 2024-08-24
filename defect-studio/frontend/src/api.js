import axios from "axios";

// 공통 API 클라이언트 설정
const apiClient = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 100000,
});

// Text-to-Image (t2i) 이미지 생성 요청 함수
export const generateImage = async (prompt, numInferenceSteps, guidanceScale) => {
  try {
    const response = await apiClient.get("/generate-image/", {
      params: {
        prompt: prompt,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
      },
      responseType: "blob", // 이미지 데이터를 바이너리로 받기 위함
    });
    return URL.createObjectURL(response.data); // Blob URL 생성
  } catch (error) {
    console.error("Error generating image:", error);
    throw error;
  }
};

// Image-to-Image (i2i) 이미지 변환 요청 함수
export const generateImg2Img = async (prompt, imageFile, numInferenceSteps, guidanceScale) => {
  try {
    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("image", imageFile);
    formData.append("num_inference_steps", numInferenceSteps);
    formData.append("guidance_scale", guidanceScale);

    const response = await apiClient.post("/generate-img2img/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      responseType: "blob", // 이미지 데이터를 바이너리로 받기 위함
    });
    return URL.createObjectURL(response.data); // Blob URL 생성
  } catch (error) {
    console.error("Error generating img2img:", error);
    throw error;
  }
};

// Inpainting 요청 함수
export const generateInpaint = async (prompt, imageFile, maskFile, numInferenceSteps, guidanceScale) => {
  try {
    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("image", imageFile);
    formData.append("mask", maskFile);
    formData.append("num_inference_steps", numInferenceSteps);
    formData.append("guidance_scale", guidanceScale);

    const response = await apiClient.post("/generate-inpaint/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      responseType: "blob",
    });
    console.log(response)
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error("Error generating inpaint:", error);
    throw error;
  }
};

// rembg (Remove Background) 요청 함수
export const removeBackground = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await apiClient.post("/remove-bg/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      responseType: "blob",
    });
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error("Error removing background:", error);
    throw error;
  }
};

