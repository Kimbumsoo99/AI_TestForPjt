from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
import os
from core.config import settings
import subprocess

# 필요 env
'''
OUTPUT_DIR=model, class, instance image -> model load, model save, class_dir save, instance_dir save
DIFFUSERS_TRAIN_PATH= diffusers git repo path
'''

router = APIRouter(
    prefix="/dreambooth",
)

training_process = None
log_file_path = None

@router.post("")
async def train_dreambooth(request: Request, background_tasks: BackgroundTasks):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # global training_process, log_file_path

    form = await request.form()

    try:
        # remote 환경
        base_output_dir = settings.OUTPUT_DIR  # /checkpoints

        # 모델 학습 파라미터
        # 모델 및 토크나이저 설정
        pretrained_model_name_or_path = form.get("model_name", "stable-diffusion-2")
        print(pretrained_model_name_or_path)
        revision = form.get("revision", None)
        variant = form.get("variant", None)
        tokenizer_name = form.get("tokenizer_name", None)
        instance_data_dir = form.get("instance_data_dir", None)
        class_data_dir = form.get("class_data_dir", None)

        # 데이터 및 프롬프트 설정
        instance_prompt = form.get("instance_prompt", None)
        class_prompt = form.get("class_prompt", None)
        num_class_images = form.get("num_class_images", 100)
        center_crop = form.get("center_crop", False)

        # 가중치
        with_prior_preservation = form.get("with_prior_preservation", False)
        prior_loss_weight = form.get("prior_loss_weight", 1.0)

        # 학습 설정
        seed = form.get("seed", None)
        resolution = form.get("resolution", 512)
        train_text_encoder = form.get("train_text_encoder", False)
        train_batch_size = form.get("train_batch_size", 2)
        sample_batch_size = form.get("sample_batch_size", 2)
        num_train_epochs = form.get("num_train_epochs")
        max_train_steps = form.get("max_train_steps", None)
        learning_rate = form.get("learning_rate", 5e-6)
        offset_noise = form.get("offset_noise", False)

        # 검증 및 체크포인트 설정
        checkpointing_steps = form.get("checkpointing_steps", 500)
        checkpoints_total_limit = form.get("checkpoints_total_limit", None)
        resume_from_checkpoint = form.get("resume_from_checkpoint", None)
        validation_prompt = form.get("validation_prompt", None)
        num_validation_images = form.get("num_validation_images", 4)
        validation_steps = form.get("validation_steps", 100)
        validation_images = form.get("validation_images", None)
        validation_scheduler = form.get("validation_scheduler", "DPMSolverMultistepScheduler")

        # 최적화 및 정밀도 설정
        use_8bit_adam = form.get("use_8bit_adam", False)
        adam_beta1 = form.get("adam_beta1", 0.9)
        adam_weight_decay = form.get("adam_weight_decay", 1e-2)
        adam_beta2 = form.get("adam_beta2", 0.999)
        adam_epsilon = form.get("adam_epsilon", 1e-08)
        max_grad_norm = form.get("max_grad_norm", 1.0)

        # 데이터 증강 및 메모리 관리
        gradient_accumulation_steps = form.get("gradient_accumulation_steps", 1)
        gradient_checkpointing = form.get("gradient_checkpointing", False)
        enable_xformers_memory_efficient_attention = form.get("enable_xformers_memory_efficient_attention", False)
        set_grads_to_none = form.get("set_grads_to_none", False)

        # 기타 설정
        # TODO 후에 local gpu server 사용
        output_dir = form.get("output_dir")
        scale_lr = form.get("scale_lr", False)
        lr_scheduler = form.get("lr_scheduler", "constant")
        lr_warmup_steps = form.get("lr_warmup_steps", 500)
        lr_num_cycles = form.get("lr_num_cycles", 1)
        lr_power = form.get("lr_power", 1.0)
        dataloader_num_workers = form.get("dataloader_num_workers", 0)
        push_to_hub = form.get("push_to_hub", False)
        hub_token = form.get("hub_token", None)
        hub_model_id = form.get("hub_model_id", None)
        logging_dir = form.get("logging_dir", "logs")
        allow_tf32 = form.get("allow_tf32", False)
        report_to = form.get("report_to", "tensorboard")
        mixed_precision = form.get("mixed_precision", None)
        prior_generation_precision = form.get("prior_generation_precision", None)
        local_rank = form.get("local_rank", -1)
        snr_gamma = form.get("snr_gamma", None)
        pre_compute_text_embeddings = form.get("pre_compute_text_embeddings", False)
        tokenizer_max_length = form.get("tokenizer_max_length", None)
        text_encoder_use_attention_mask = form.get("text_encoder_use_attention_mask", False)
        skip_save_text_encoder = form.get("skip_save_text_encoder", False)
        class_labels_conditioning = form.get("class_labels_conditioning", None)

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

        instance_dir = os.path.join(base_output_dir, f"{member_id}", "instance_images")
        class_dir = os.path.join(base_output_dir, f"{member_id}", "class_images")
        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(class_dir, exist_ok=True)

        if ("stable-diffusion-2" == pretrained_model_name_or_path):
            pretrained_model_name_or_path = os.path.join(base_output_dir, pretrained_model_name_or_path)
        else:
            pretrained_model_name_or_path = os.path.join(base_output_dir, f"{member_id}", pretrained_model_name_or_path)

        # 인스턴스 이미지 저장
        for image in instance_images:
            with open(os.path.join(instance_dir, image.filename), "wb") as f:
                f.write(await image.read())

        # 클래스 이미지 저장
        for image in class_images:
            with open(os.path.join(class_dir, image.filename), "wb") as f:
                f.write(await image.read())

        project_root = settings.DIFFUSERS_TRAIN_PATH
        train_script = os.path.join(project_root, "dreambooth/train_dreambooth.py")

        # TODO LOG 관련 기능 미사용 중
        # log_file_path = os.path.join(base_output_dir, f"{member_id}", "training.log")

        # 필수 파라미터 추가
        command = [
            "accelerate", "launch", train_script,
            "--pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4",
            "--instance_data_dir", instance_dir,
            "--class_data_dir", class_dir,
            "--output_dir", output_dir,
            "--resolution", resolution,
            "--train_batch_size", train_batch_size,
            "--learning_rate", learning_rate,
            "--num_class_images", num_class_images,
            # "--logging_epoch", "True",
            "--num_train_epochs", num_train_epochs
        ]

        # 필수 파라미터?
        command.extend("--with_prior_preservation")
        command.extend(["--prior_loss_weight", prior_loss_weight])
        command.extend(["--instance_prompt", instance_prompt])
        command.extend(["--class_prompt", class_prompt])
        command.extend(["--gradient_accumulation_steps", gradient_accumulation_steps])
        command.extend(["--gradient_checkpointing"])
        command.extend("--use_8bit_adam")
        command.extend(["--lr_scheduler", lr_scheduler])
        command.extend(["--lr_warmup_steps", lr_warmup_steps])

        # 선택적 파라미터 추가
        if revision is not None:
            command.extend(["--revision", revision])
        if variant is not None:
            command.extend(["--variant", variant])
        if tokenizer_name is not None:
            command.extend(["--tokenizer_name", tokenizer_name])
        if instance_data_dir is not None:
            command.extend(["--instance_data_dir", instance_data_dir])
        if class_data_dir is not None:
            command.extend(["--class_data_dir", class_data_dir])
        if center_crop:
            command.append("--center_crop")
        if seed is not None:
            command.extend(["--seed", seed])
        if not train_text_encoder:
            command.append("--train_text_encoder")
        if sample_batch_size is not None:
            command.extend(["--sample_batch_size", sample_batch_size])
        if max_train_steps is not None:
            command.extend(["--max_train_steps", max_train_steps])
        if offset_noise:
            command.append("--offset_noise")
        if checkpointing_steps is not None:
            command.extend(["--checkpointing_steps", checkpointing_steps])
        if checkpoints_total_limit is not None:
            command.extend(["--checkpoints_total_limit", checkpoints_total_limit])
        if resume_from_checkpoint is not None:
            command.extend(["--resume_from_checkpoint", resume_from_checkpoint])
        if validation_prompt is not None:
            command.extend(["--validation_prompt", validation_prompt])
        if num_validation_images is not None:
            command.extend(["--num_validation_images", num_validation_images])
        if validation_steps is not None:
            command.extend(["--validation_steps", validation_steps])
        if validation_images is not None:
            command.extend(["--validation_images", validation_images])
        if validation_scheduler is not None:
            command.extend(["--validation_scheduler", validation_scheduler])
        # if not use_8bit_adam:
        #     command.extend(["--use_8bit_adam", use_8bit_adam])
        if adam_beta1 is not None:
            command.extend(["--adam_beta1", adam_beta1])
        if adam_weight_decay is not None:
            command.extend(["--adam_weight_decay", adam_weight_decay])
        if adam_beta2 is not None:
            command.extend(["--adam_beta2", adam_beta2])
        if adam_epsilon is not None:
            command.extend(["--adam_epsilon", adam_epsilon])
        if max_grad_norm is not None:
            command.extend(["--max_grad_norm", max_grad_norm])
        # if gradient_checkpointing:
        #     command.append("--gradient_checkpointing")
        if enable_xformers_memory_efficient_attention:
            command.append("--enable_xformers_memory_efficient_attention")
        if set_grads_to_none:
            command.append("--set_grads_to_none")
        '''
        Local 사용 시 사용 고려
        if not output_dir:
            command.extend(["--output_dir", output_dir])
        '''
        if scale_lr:
            command.append("--scale_lr")
        if lr_num_cycles is not None:
            command.extend(["--lr_num_cycles", lr_num_cycles])
        if lr_power is not None:
            command.extend(["--lr_power", lr_power])
        if dataloader_num_workers is not None:
            command.extend(["--dataloader_num_workers", dataloader_num_workers])
        if push_to_hub:
            command.append("--push_to_hub")
        if hub_token is not None:
            command.extend(["--hub_token", hub_token])
        if hub_model_id is not None:
            command.extend(["--hub_model_id", hub_model_id])
        if logging_dir is not None:
            command.extend(["--logging_dir", logging_dir])
        if allow_tf32:
            command.append("--allow_tf32")
        if report_to is not None:
            command.extend(["--report_to", report_to])
        if mixed_precision is not None:
            command.extend(["--mixed_precision", mixed_precision])
        if prior_generation_precision is not None:
            command.extend(["--prior_generation_precision", prior_generation_precision])
        if local_rank is not None:
            command.extend(["--local_rank", local_rank])
        if snr_gamma is not None:
            command.extend(["--snr_gamma", snr_gamma])
        if pre_compute_text_embeddings:
            command.append("--pre_compute_text_embeddings")
        if tokenizer_max_length is not None:
            command.extend(["--tokenizer_max_length", tokenizer_max_length])
        if text_encoder_use_attention_mask:
            command.append("--text_encoder_use_attention_mask")
        if skip_save_text_encoder:
            command.append("--skip_save_text_encoder")
        if class_labels_conditioning is not None:
            command.extend(["--class_labels_conditioning", class_labels_conditioning])

        # Custom Command
        if log_epochs is not None:
            command.extend(["--log_epochs", log_epochs])

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


# 향후 status 확인 필요 시 사용
@router.get("/training_status/")
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
