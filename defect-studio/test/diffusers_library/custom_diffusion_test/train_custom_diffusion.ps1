# PowerShell 스크립트: train_custom_diffusion.ps1

# Step 1: 환경 변수 설정
$env:MODEL_NAME = "CompVis/stable-diffusion-v1-4"
$env:OUTPUT_DIR = "./output/checkpoints"  # 모델 체크포인트가 저장될 디렉토리 경로를 설정하세요.
$env:INSTANCE_DIR = "./data/cat"  # 훈련할 인스턴스 이미지(예: 고양이)의 경로를 설정하세요.
$env:CLASS_DIR = "./real_reg/samples_cat"  # 클래스 이미지가 있는 디렉토리 경로를 설정하세요.

# Step 2: 클래스 이미지 확인 및 수집 (이미지가 충분히 준비되어 있지 않다면)
if (!(Test-Path -Path $env:CLASS_DIR -PathType Container) -or (Get-ChildItem -Path $env:CLASS_DIR/images -File).Count -lt 200) {
    Write-Host "Class images not found or less than 200 images. Collecting images..."
    python retrieve.py --class_prompt "cat" --class_data_dir $env:CLASS_DIR --num_class_images 200
} else {
    Write-Host "Class images already prepared."
}

# Step 3: 훈련 스크립트 실행
accelerate launch train_custom_diffusion.py `
  --pretrained_model_name_or_path=$env:MODEL_NAME `
  --instance_data_dir=$env:INSTANCE_DIR `
  --output_dir=$env:OUTPUT_DIR `
  --class_data_dir=$env:CLASS_DIR `
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 `
  --class_prompt="cat" --num_class_images=200 `
  --instance_prompt="photo of a <new1> cat" `
  --resolution=512 `
  --enable_xformers_memory_efficient_attention `
  --train_batch_size=1 `
  --learning_rate=1e-5 `
  --lr_warmup_steps=0 `
  --max_train_steps=250 `
  --scale_lr --hflip `
  --modifier_token "<new1>"