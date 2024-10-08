# CleanUp AI Model Test Document
## 1. 개요
### 1.1 테스트 목적
본 테스트의 목적은 cleanup 목적에 맞게 사용한 Inpainting 모델의 정확도 및 추론 가용성을 평가하는 것입니다.

### 1.2 테스트 대상 모델
- 모델명: **stabilityai/stable-diffusion-2-inpainting**
- 레퍼런스: [hugging face](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

## 2. 테스트 환경
### 2.1 하드웨어 (ex.)
- CPU: Intel(R) Core(TM) Ultra 7 155H 3.80 GHz
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- RAM: 32GB

### 2.2 소프트웨어
- OS: Windows 10
- Python 버전: 3.10.6
- 딥러닝 프레임워크: `torch 2.0.1+cu117`


### 2.3 테스트 데이터셋
- 데이터셋 구성: 

| 카테고리 (Category) | bottle | cable | capsule | pill |
|---------------|---------------|---------------|---------------|---------------|
| 이미지 수 (Image Num)      |      10장   |    10장     |  10장 |     10장     |
| 설명 (Description)      |      Object 한 개 존재하는 유사한 사진 모음, 흰색 배경   |    Object 한 개 존재하는 유사한 사진 모음, 컬러 배경     |  Object 한 개 존재하는 유사한 사진 모음, 흰색 배경 |     Object 한 개 존재하는 유사한 사진 모음, 검은색 배경   |
| 예시 (Example)      |     <img src="./img/bottle004.png"/>  |    <img src="./img/cable000.png"/>     |      <img src="./img/capsule000.png"/>     |      <img src="./img/pill002.png"/>     |
   - 데이터셋 구성 현황
- 이미지 수: 40장
- 해상도: 다양 (최소 800x800, 최대 1024x1024)
- 구성: MVTec-AD 100%

## 3. 평가 지표
1. 정확도
   - MAE (Mean Absolute Error) (L1 metric)
   - Max F-Measure
2. 처리 속도
   - FPS (Frames Per Second)
3. 메모리 사용량
   - 최대 GPU 메모리 사용량 (MB)

## 4. 테스트 시나리오
### 4.1 정량평가 (Quantitative Evaluation)
#### 4.1.1 MAE/F-Measure
| 수량(Quantity) | MAE | F-Measure(F1 Score) |
|---------------|---------------|---------------|
|      1장          |     3.19 ~ 39.15       |    0.62 ~ 0.99 |
|      10장         |     4.31 ~ 36.67      |    0.65 ~ 0.99   |
|      40장         |        13.55       |      0.90        |

#### 4.1.2 Speed/GPU Usage 평가
| 표본 | 처리 시간 (ms) | GPU 메모리 (MB) |
|---------------|---------------|----------------|
|      1장         |      24000ms      |     3402MB        |
|      10장         |     238790ms       |      3402MB     |
|      40장         |     978530ms       |      3402MB      |

### 4.2 정성평가 (Qualitative Evaluation)
> 각 카테고리 별로 특징 나열 후 사진 평가를 진행합니다. 사용된 카테고리는 배경을 기반으로 하였습니다.
각각의 사진 평가는 주관적인 평가로 6명의 팀원이서 `상, 중, 하`를 통해 평가 점수를 작성했습니다.


관련하여 평가에 사용된 사진은 하단 데이터 셋에 추가하겠습니다.
참고로 모든 데이터셋은 하나의 Object만 존재합니다.

| 카테고리(배경 색상) | 수량 | 사진 출력 제대로 됐는지 |
|---------------|---------------|---------------|
|      흰색      |   20장  |    아쉬운 요소가 있지만, 성공적인 작업 (전체 중 70% GOOD)      |
|      검은색         | 10장 |    성공적인 작업 (50%)        |
|      컬러         |  10장  |   아쉬운 요소가 많음, 실패 포함 (전체 중 20% GOOD, 무난한 부분 15%)    |


### 4.3 이미지 크기 및 해상도 테스트
- 최소 800x800, 최대 1024x1024
- Inpainting 생성 이미지 512x512 `resize`를 통해 이미지 처리

## 5. 테스트 절차
1. 테스트 환경 설정

2. 모델 로드

   ```python
   from diffusers import StableDiffusionInpaintPipeline

   inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
      "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    ).to("cuda")

    result_image = inpaint_pipe(
                prompt="A smooth, undamaged version of the image",
                image=origin_image,
                mask_image=mask_image
            ).images[0]
   ```

3. 테스트 데이터셋 준비
  
    - [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) 
  
    - [VISION-Workshop/VISION-Datasets](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)

4. 각 테스트 시나리오 실행
    
    a. 카테고리에 해당되는 GOOD 데이터와 불량 이미지 데이터, MASK 영역 데이터 준비
      
    b. 이미지 로드 / Cleanup 수행
      - 불량 이미지 데이터 영역 지정 후 수행
    
    c. 결과 저장 및 메트릭 계산
      - GOOD 데이터와 Cleanup 작업 후 발생하는 데이터 차이 비교

5. 결과 분석 및 보고서 작성

## 6. 결과 기록 및 분석 방법
### 6.1 결과 기록 템플릿
- 이미지 일부 발췌

<img src="./img/inpainting_평가.png"/>

## 7. 성능 기준 (Acceptance Criteria)
- MAE: ≤ 15
- 평균 F1 Score: ≥ 0.90
- 평균 처리 시간: ≤ 5000ms (512x512 해상도 기준)
- 최대 GPU 메모리 사용량: ≤ 3GB

## 8. 버그 보고 및 추적 프로세스
1. 버그 발견 시 즉시 Jira 티켓 생성
2. 버그 재현 단계 상세히 기록
3. 버그의 심각도 및 우선순위 설정
4. 개발팀에 할당 및 해결 과정 추적
5. 해결된 버그에 대한 재테스트 수행

## 9. 테스트 결과 요약

- 비교 answer 이미지 결과
<img src="./img/answer.png"/>

- cleanup 작업 전 사진 결과
<img src="./img/origin.png"/>

- cleanup 작업 mask 사진 결과
<img src="./img/mask.png"/>

- 이미지 실습 결과 
<img src="./img/inpainting.png"/>

## 10. 결론

전체적인 성능은 사용하는 것을 고려하는 것도 좋지만, 우선 느리고 GPU 사용량이 너무 크다는 문제점이 있었습니다.
따라서 해당 모델 사용은 보류하기로 결정했습니다.
