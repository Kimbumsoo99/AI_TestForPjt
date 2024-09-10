import requests
import os

# FastAPI 서버 URL
url = "http://localhost:8000/train_model/"

# JSON 데이터 (TrainingRequest의 필드들)
request_data = {
    # "model_name": "CompVis/stable-diffusion-v1-4",
    "model_name" : "stable-diffusion-2",
    "instance_prompt": "a photo of sks cat",
    "class_prompt": "a photo of a cat",
    "batch_size": 4,
    "learning_rate": 0.000005,
    "num_class_images": 200,
    "num_train_epochs": 5,
    "member_id": "2",
    "train_model_name": "first_model"
}

instance_dir = "/home/j-j11s001/project/bumsoo/dataset/dreambooth_input/instance_dir"
class_dir = "/home/j-j11s001/project/bumsoo/dataset/dreambooth_input/class_dir"


# 파일 데이터를 준비
files = []

# 인스턴스 이미지 파일 추가
for image_name in os.listdir(instance_dir):
    image_path = os.path.join(instance_dir, image_name)
    if os.path.isfile(image_path):  # 파일인지 확인
        files.append(('instance_images', (image_name, open(image_path, 'rb'), 'image/jpeg')))

# 클래스 이미지 파일 추가
for image_name in os.listdir(class_dir):
    image_path = os.path.join(class_dir, image_name)
    if os.path.isfile(image_path):  # 파일인지 확인
        files.append(('class_images', (image_name, open(image_path, 'rb'), 'image/jpeg')))

# 요청 보내기
response = requests.post(url, data=request_data, files=files)

# 응답 출력
print(response.json())