# # from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoPipelineForInpainting, StableDiffusionPipeline
# import torch
# from PIL import Image
#
# # 모델 경로 설정 (로컬 경로로 업데이트)
# base_path = "C:\DefectStudio\cleanup\PowerPaint-v2-1"
# model_path = f"{base_path}\diffusion_pytorch_model.safetensors"
# pipeline = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
# # vae_path = f"{base_path}/realisticVisionV60B1_v51VAE/vae"
# # unet_path = f"{base_path}/realisticVisionV60B1_v51VAE/unet"  # 경로 수정
# # text_encoder_path = f"{base_path}/realisticVisionV60B1_v51VAE/text_encoder"
# # tokenizer_path = f"{base_path}/realisticVisionV60B1_v51VAE/tokenizer"
# # safety_checker_path = f"{base_path}/realisticVisionV60B1_v51VAE/safety_checker"
# # feature_extractor_path = f"{base_path}/realisticVisionV60B1_v51VAE/feature_extractor"
# # scheduler_path = f"{base_path}/realisticVisionV60B1_v51VAE/scheduler"
#
# # 각 구성 요소 로드
# # vae = AutoencoderKL.from_pretrained(vae_path, filename="diffusion_pytorch_model.bin")
# # unet = UNet2DConditionModel.from_pretrained(unet_path, filename="diffusion_pytorch_model.bin")  # 경로 수정
# # text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, filename="pytorch_model.bin")
# # tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
# # feature_extractor = StableDiffusionPipeline.from_pretrained(feature_extractor_path, subfolder="feature_extractor")
# # safety_checker = StableDiffusionPipeline.from_pretrained(safety_checker_path, subfolder="safety_checker")
# # scheduler = StableDiffusionPipeline.from_pretrained(scheduler_path, subfolder="scheduler")
#
# # 파이프라인 초기화
# # pipe = AutoPipelineForInpainting.from_pretrained(base_path+"/diffusion_pytorch_model.safetensors")
# # pipe.to("cuda")
#
# # 입력 이미지와 마스크 로드
# input_image = Image.open("C:/uploads/lama/input.png").convert("RGB")
# input_mask = Image.open("C:/uploads/lama/input_mask.jpg").convert("RGB")
#
# # 이미지 클린업
# output = pipeline(prompt="", image=input_image, mask_image=input_mask).images[0]
#
# # 결과 저장
# output.save("C:/uploads/lama/output.png")


from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

# 올바른 모델 경로 설정
base_path = "C:/DefectStudio/cleanup/PowerPaint-v2-1/realisticVisionV60B1_v51VAE"

# StableDiffusionInpaintPipeline을 올바른 경로에서 로드
pipeline = StableDiffusionInpaintPipeline.from_pretrained(base_path, torch_dtype=torch.float16).to("cuda")

# 입력 이미지와 마스크 로드
input_image = Image.open("C:/uploads/lama/input.png").convert("RGB")
input_mask = Image.open("C:/uploads/lama/input_mask.jpg").convert("L")  # 마스크는 일반적으로 L 모드(흑백)로 처리

# 이미지 클린업 수행
output = pipeline(prompt="restore the area", image=input_image, mask_image=input_mask).images[0]

# 결과 저장
output.save("C:/uploads/lama/output.png")