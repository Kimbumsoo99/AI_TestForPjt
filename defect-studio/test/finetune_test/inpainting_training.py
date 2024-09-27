import os
import subprocess
from dotenv import load_dotenv
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# .env 파일 로드
load_dotenv()

# 환경 변수 불러오기
model_name = os.getenv("MODEL_NAME")
instance_dir = os.getenv("INSTANCE_DIR")
class_dir = os.getenv("CLASS_DIR")
base_output_dir = os.getenv("OUTPUT_DIR")

# 현재 날짜와 시간을 가져와서 고유한 숫자 문자열로 설정
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = os.path.join(base_output_dir, f"cable_5_{timestamp}_dreambooth")

# OUTPUT_DIR이 존재하지 않으면 생성
os.makedirs(output_dir, exist_ok=True)

project_root = os.getenv('DIFFUSERS_TRAIN_PATH')  # 실제 프로젝트 경로로 수정해주세요.
train_script = os.path.join(project_root, "research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py")

# 명령어 구성
command = [
    "accelerate", "launch", train_script,
    # "--pretrained_model_name_or_path", model_name,
    "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-inpainting",
    "--instance_data_dir", instance_dir,
    "--class_data_dir", class_dir,
    "--output_dir", output_dir,
    "--with_prior_preservation",
    "--prior_loss_weight", "0.3",
    "--instance_prompt", "a photo of sks bent wire cable",
    "--class_prompt", "a photo of cable",
    "--resolution", "512",
    "--train_batch_size", "12" ,
    "--gradient_accumulation_steps", "2",
    # 16GB add
    "--gradient_checkpointing",
    "--use_8bit_adam",
    # 16GB end
    # "--learning_rate", "5e-6", #1, 2 번째 실습
    "--learning_rate", "5e-6",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--num_class_images", "58",
    "--num_train_epochs", "50",
    # "--logging_epoch", "True"
    # "--max_train_steps", "800",
]

# OS별 명령어 포맷팅 (Windows와 리눅스에서 모두 작동)
if os.name == 'nt':
    command = ['cmd', '/c'] + command

# 명령어 실행
subprocess.run(command)
