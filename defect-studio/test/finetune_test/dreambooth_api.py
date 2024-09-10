from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from typing import List
import subprocess
import os
from dotenv import load_dotenv
from datetime import datetime

app = FastAPI()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

load_dotenv()

training_process = None
log_file_path = None

# Pydantic 모델을 사용한 요청 바디 정의
class TrainingRequest(BaseModel):
    model_name: str
    instance_prompt: str = None
    class_prompt: str = None
    batch_size: int = 2
    learning_rate: float = 1e-6
    num_class_images: int = 200
    num_train_epochs: int = 30
    member_id: str
    train_model_name: str


@app.post("/train_model/")
async def train_model(request: Request,
                      background_tasks: BackgroundTasks,
                     ):
    global training_process, log_file_path

    form = await request.form()

    try:
        model_name = os.getenv("MODEL_NAME")
        instance_dir = os.getenv("INSTANCE_DIR")
        class_dir = os.getenv("CLASS_DIR")
        base_output_dir = os.getenv("OUTPUT_DIR")

        # 모델에서 파라미터 받기
        model_name = form.get("model_name")
        # instance_dir = form.get("instance_dir")
        # class_dir = form.get("class_dir")
        instance_prompt = form.get("instance_prompt")
        class_prompt = form.get("class_prompt")
        batch_size = form.get("batch_size")
        learning_rate = form.get("learning_rate")
        num_class_images = form.get("num_class_images")
        num_train_epochs = form.get("num_train_epochs")
        member_id = form.get("member_id")
        train_model_name = form.get("train_model_name")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        instance_images = form.getlist("instance_images")
        class_images = form.getlist("class_images")


        # model_name(model_name) : Base Model(local에 stabilityai/stable-diffusion-2 Save)
        # instance_data(instance_dir) : instance image requests (!required)
        # class_data(class_dir) : class image requests (!required)
        # output_dir : f"base_path + {member_id}/{model_name}" -> model_name 검증 로직?
        # instance_prompt : requests (!required)
        # class_prompt : requests (!required)
        # batch_size : requests (!required, default : 2)
        # learning_rate : requests (default: 1e-6)
        # num_class_images : class_images 개수
        # num_train_epochs : requests (default: 30)
        
        
        output_dir = os.path.join(base_output_dir, f"{member_id}", f"{train_model_name}")
        os.makedirs(output_dir, exist_ok=True)

        instance_dir = os.path.join(output_dir, "instance_images")
        class_dir = os.path.join(output_dir, "class_images")
        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(class_dir, exist_ok=True)

        model_name = os.path.join(base_output_dir, model_name)
        print(model_name);

        # 인스턴스 이미지 저장
        for image in instance_images:
            with open(os.path.join(instance_dir, image.filename), "wb") as f:
                f.write(await image.read())

        # 클래스 이미지 저장
        for image in class_images:
            with open(os.path.join(class_dir, image.filename), "wb") as f:
                f.write(await image.read())

        project_root = os.getenv('DIFFUSERS_TRAIN_PATH')
        train_script = os.path.join(project_root, "dreambooth/train_dreambooth.py")

        log_file_path = os.path.join(output_dir, "training.log")

        command = [
            "accelerate", "launch", train_script,
            "--pretrained_model_name_or_path", model_name,
            "--instance_data_dir", instance_dir,
            "--class_data_dir", class_dir,
            "--output_dir", output_dir,
            "--with_prior_preservation",    # ???
            "--prior_loss_weight", "1.0",   # ???
            "--instance_prompt", instance_prompt,
            "--class_prompt", class_prompt,
            "--resolution", "512",
            "--train_batch_size", batch_size,
            "--gradient_accumulation_steps", "2", # ??? 배치 크기를 줄이는 대신, 여러 스텝에 걸쳐 그래디언트를 누적한 후 업데이트하는 방법 -> 메모리 이점
            "--gradient_checkpointing",     # ???
            "--use_8bit_adam",              # ???
            "--learning_rate", learning_rate,
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--num_class_images", num_class_images,
            # "--logging_epoch", "True",
            "--num_train_epochs", num_train_epochs
        ]

        if os.name == 'nt':
            command = ['cmd', '/c'] + command

        background_tasks.add_task(run_training, command)
        return {"status": "Training started successfully", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_training(command):
    global training_process
    training_process = subprocess.Popen(command)
    training_process.wait()

@app.get("/training_status/")
async def training_status():
    global training_process, log_file_path

    if training_process is None:
        raise HTTPException(status_code=400, detail="Training has not started yet.")
    
    poll = training_process.poll()
    if poll is None:
        status = "Training is still running."
    else:
        status = "Training has completed."
    
    log_content = ""
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            log_content = log_file.read()

    return {
        "status": status,
        "log": log_content
    }