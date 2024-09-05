import os
import subprocess
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 불러오기
model_name = os.getenv("MODEL_NAME")
instance_dir = os.getenv("INSTANCE_DIR")
class_dir = os.getenv("CLASS_DIR")
output_dir = os.getenv("OUTPUT_DIR")

# 명령어 구성
command = [
    "accelerate", "launch", "../diffusers/examples/dreambooth/train_dreambooth.py",
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--class_data_dir", class_dir,
    "--output_dir", output_dir,
    "--with_prior_preservation",
    "--prior_loss_weight", "1.0",
    "--instance_prompt", "a photo of sks dog",
    "--class_prompt", "a photo of dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "5e-6",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--num_class_images", "200",
    "--max_train_steps", "800",
]

# OS별 명령어 포맷팅 (Windows와 리눅스에서 모두 작동)
if os.name == 'nt':
    command = ['cmd', '/c'] + command
else:
    command = ['bash', '-c', ' '.join(command)]

# 명령어 실행
subprocess.run(command)
