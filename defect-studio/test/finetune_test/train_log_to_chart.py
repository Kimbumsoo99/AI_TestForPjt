import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_chart(file_name):
    # CSV 파일 경로 설정
    log_file_path = os.path.join('./log', file_name)
    
    # 파일이 존재하는지 확인
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"CSV file '{log_file_path}' does not exist.")
    
    # CSV 파일 읽기
    logs = pd.read_csv(log_file_path)
    
    # 로그 파일이 예상한 대로 'global_step', 'loss', 'learning_rate' 열을 포함하는지 확인
    if not {'global_step', 'loss', 'learning_rate'}.issubset(logs.columns):
        raise ValueError("The CSV file must contain 'global_step', 'loss', and 'learning_rate' columns.")
    
    # 차트 생성
    plt.figure(figsize=(10, 6))

    # Loss 차트
    plt.subplot(2, 1, 1)
    plt.plot(logs['global_step'], logs['loss'], label='Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Learning Rate 차트
    plt.subplot(2, 1, 2)
    plt.plot(logs['global_step'], logs['learning_rate'], label='Learning Rate', color='orange')
    plt.xlabel('Global Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # 레이아웃 조정
    plt.tight_layout()

    # 저장할 이미지 경로 설정
    output_dir = './log/output'
    os.makedirs(output_dir, exist_ok=True)  # output 디렉토리가 없다면 생성
    
    # 이미지 파일 이름과 경로 설정
    image_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")

    # 이미지 파일로 저장
    plt.savefig(image_file_path)
    plt.close()  # plt.show() 대신 close()를 사용하여 파일로만 저장
    
    print(f"Chart saved as {image_file_path}")

generate_chart("logtest1_20240909143648_dreambooth_logs.csv")