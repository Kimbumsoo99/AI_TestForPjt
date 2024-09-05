import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error
import pandas as pd

# 디렉토리 설정
original_dir = "C:/DefectStudio/testing_docs/remove_bg/origin"
expectation_dir = "C:/DefectStudio/testing_docs/remove_bg/expectation"
test_dir = "C:/DefectStudio/testing_docs/remove_bg/test"

# 결과 저장을 위한 리스트
results = []

# 디렉토리 내의 모든 이미지 파일에 대해 MAE 계산
for filename in os.listdir(original_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        base_filename = os.path.splitext(filename)[0]  # 확장자를 제외한 파일 이름
        original_image_path = os.path.join(original_dir, filename)
        test_image_filename = f"{base_filename}.png"
        test_image_path = os.path.join(test_dir, test_image_filename)

        # 예상 이미지 파일 이름 생성
        expectation_image_filename = f"{base_filename}-removebg-preview.png"
        expectation_image_path = os.path.join(expectation_dir, expectation_image_filename)

        # 예상 이미지와 테스트 이미지가 존재하는지 확인
        if not os.path.exists(expectation_image_path):
            print(f"Expected image not found for {filename}. Skipping.")
            continue

        try:
            # 예상 이미지와 테스트 이미지 로드
            expectation_image = Image.open(expectation_image_path).convert("RGB")
            test_image = Image.open(test_image_path).convert("RGB")

            # 이미지를 numpy 배열로 변환
            expectation_array = np.array(expectation_image)
            test_array = np.array(test_image)

            # 이미지 크기가 동일한지 확인
            if expectation_array.shape != test_array.shape:
                print(f"Error: Image dimensions do not match for {filename}. Skipping.")
                continue

            # MAE 계산
            mae = mean_absolute_error(expectation_array.flatten(), test_array.flatten())

            # 결과 저장
            results.append({'Image': filename, 'MAE': mae})
            print(f"MAE for {filename}: {mae}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 결과를 데이터프레임으로 변환하여 표로 나타내기
mae_df = pd.DataFrame(results)

# 결과를 콘솔에 출력하거나 CSV 파일로 저장할 수 있음
print(mae_df)
mae_df.to_csv('mae_results.csv', index=False)
