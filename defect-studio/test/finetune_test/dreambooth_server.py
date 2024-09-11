from fastapi import APIRouter, FastAPI, BackgroundTasks, HTTPException, status, Request
import os
import subprocess
from pydantic import BaseModel
from typing import List
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime

router = APIRouter(
    prefix="/dreambooth",
)

training_process = None
log_file_path = None

@router.post("")
async def train_dreambooth(request: Request, background_tasks: BackgroundTasks):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    global training_process, log_file_path

    form = await request.form()

    try:
        # remote 환경
        base_output_dir = os.getenv("OUTPUT_DIR") # /checkpoints

        #


        # 모델 학습 파라미터
        # 모델 및 토크나이저 설정
        pretrained_model_name_or_path = form.get("model_name")
        revision = form.get("revision")
        variant = form.get("variant")
        tokenizer_name = form.get("tokenizer_name")
        instance_data_dir = form.get("instance_data_dir")
        class_data_dir = form.get("class_data_dir")

        # 데이터 및 프롬프트 설정
        instance_prompt = form.get("instance_prompt")
        class_prompt = form.get("class_prompt")
        num_class_images = form.get("num_class_images")
        center_crop = form.get("center_crop")

        # 가중치
        with_prior_preservation = form.get("with_prior_preservation")
        prior_loss_weight = form.get("prior_loss_weight", 1.0)

        # 학습 설정
        seed = form.get("seed")
        resolution = form.get("resolution", 512)
        train_text_encoder = form.get("train_text_encoder")
        train_batch_size = form.get("train_batch_size", 2)
        sample_batch_size = form.get("sample_batch_size")
        num_train_epochs = form.get("num_train_epochs")
        max_train_steps = form.get("max_train_steps")
        learning_rate = form.get("learning_rate", 1e-6)
        offset_noise = form.get("offset_noise")

        # 검증 및 체크포인트 설정
        checkpointing_steps = form.get("checkpointing_steps")
        checkpoints_total_limit = form.get("checkpoints_total_limit")
        resume_from_checkpoint = form.get("resume_from_checkpoint")
        validation_prompt = form.get("validation_prompt")
        num_validation_images = form.get("num_validation_images")
        validation_steps = form.get("validation_steps")
        validation_images = form.get("validation_images")
        validation_scheduler = form.get("validation_scheduler")

        # 최적화 및 정밀도 설정
        use_8bit_adam = form.get("use_8bit_adam")
        adam_beta1 = form.get("adam_beta1")
        adam_weight_decay = form.get("adam_weight_decay")
        adam_beta2 = form.get("adam_beta2")
        adam_epsilon = form.get("adam_epsilon")
        max_grad_norm = form.get("max_grad_norm")

        # 데이터 증강 및 메모리 관리
        gradient_accumulation_steps = form.get("gradient_accumulation_steps", 2)
        gradient_checkpointing = form.get("gradient_checkpointing")
        enable_xformers_memory_efficient_attention = form.get("enable_xformers_memory_efficient_attention")
        set_grads_to_none = form.get("set_grads_to_none")

        # 기타 설정
        output_dir = form.get("output_dir")
        scale_lr = form.get("scale_lr")
        lr_scheduler = form.get("lr_scheduler", "constant")
        lr_warmup_steps = form.get("lr_warmup_steps", "0")
        lr_num_cycles = form.get("lr_num_cycles")
        lr_power = form.get("lr_power")
        dataloader_num_workers = form.get("dataloader_num_workers")
        push_to_hub = form.get("push_to_hub")
        hub_token = form.get("hub_token")
        hub_model_id = form.get("hub_model_id")
        logging_dir = form.get("logging_dir")
        allow_tf32 = form.get("allow_tf32")
        report_to = form.get("report_to")
        mixed_precision = form.get("mixed_precision")
        prior_generation_precision = form.get("prior_generation_precision")
        local_rank = form.get("local_rank")
        snr_gamma = form.get("snr_gamma")
        pre_compute_text_embeddings = form.get("pre_compute_text_embeddings")
        tokenizer_max_length = form.get("tokenizer_max_length")
        text_encoder_use_attention_mask = form.get("text_encoder_use_attention_mask")
        skip_save_text_encoder = form.get("skip_save_text_encoder")
        class_labels_conditioning = form.get("class_labels_conditioning")

        # 이외 파라미터
        member_id = form.get("member_id")
        train_model_name = form.get("train_model_name")
        log_epochs = form.get("log_epochs")

        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

        pretrained_model_name_or_path = os.path.join(base_output_dir, pretrained_model_name_or_path)
        print(pretrained_model_name_or_path)

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

        # 필수 파라미터 추가
        command = [
            "accelerate", "launch", train_script,
            "--pretrained_model_name_or_path", pretrained_model_name_or_path,
            "--instance_data_dir", instance_dir,
            "--class_data_dir", class_dir,
            "--output_dir", output_dir,
            "--with_prior_preservation",  # ???
            "--prior_loss_weight", prior_loss_weight,  # ???
            "--instance_prompt", instance_prompt,
            "--class_prompt", class_prompt,
            "--resolution", resolution,
            "--train_batch_size", train_batch_size,
            "--gradient_accumulation_steps", gradient_accumulation_steps,  # ??? 배치 크기를 줄이는 대신, 여러 스텝에 걸쳐 그래디언트를 누적한 후 업데이트하는 방법 -> 메모리 이점
            "--gradient_checkpointing",  # ???
            "--use_8bit_adam",  # ???
            "--learning_rate", learning_rate,
            "--lr_scheduler", lr_scheduler,
            "--lr_warmup_steps", lr_warmup_steps,
            "--num_class_images", num_class_images,
            # "--logging_epoch", "True",
            "--num_train_epochs", num_train_epochs
        ]

        # 필수 파라미터?
        if with_prior_preservation:
            command.extend(["--with_prior_preservation"])
        if gradient_checkpointing:
            command.extend(["--gradient_checkpointing"])
        if use_8bit_adam:
            command.extend(["--use_8bit_adam"])



        # 선택적 파라미터 추가
        if instance_prompt:
            command.extend(["--instance_prompt", instance_prompt])

        if class_prompt:
            command.extend(["--class_prompt", class_prompt])

        if train_batch_size:
            command.extend(["--train_batch_size", train_batch_size])

        if learning_rate:
            command.extend(["--learning_rate", str(learning_rate)])

        if num_class_images:
            command.extend(["--num_class_images", str(num_class_images)])

        if os.name == 'nt':
                command = ['cmd', '/c'] + command

        background_tasks.add_task(run_training, command)
        return {"status": "Training started successfully", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_model/")
async def train_model(request: Request,
                      background_tasks: BackgroundTasks,
                      ):
    global training_process, log_file_path

    form = await request.form()

    try:
        # model_name = os.getenv("MODEL_NAME")
        # instance_dir = os.getenv("INSTANCE_DIR")
        # class_dir = os.getenv("CLASS_DIR")
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

        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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
        print(model_name)

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
            "--with_prior_preservation",  # ???
            "--prior_loss_weight", "1.0",  # ???
            "--instance_prompt", instance_prompt,
            "--class_prompt", class_prompt,
            "--resolution", "512",
            "--train_batch_size", batch_size,
            "--gradient_accumulation_steps", "2",  # ??? 배치 크기를 줄이는 대신, 여러 스텝에 걸쳐 그래디언트를 누적한 후 업데이트하는 방법 -> 메모리 이점
            "--gradient_checkpointing",  # ???
            "--use_8bit_adam",  # ???
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