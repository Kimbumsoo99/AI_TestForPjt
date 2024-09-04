# Background Removal AI Model Test Document
## 1. 개요
### 1.1 테스트 목적
본 테스트의 목적은 briaai/RMBG-1.4 AI 모델의 성능, 정확도 및 추론 가용성을 평가하는 것입니다.

### 1.2 테스트 대상 모델
- 모델명: **briaai/RMBG-1.4**
- 레퍼런스: [huggingface link](https://huggingface.co/briaai/RMBG-1.4) / [github](https://github.com/chenxwh/cog-RMBG)
- 모델 유형: `**DIS(IS-Net) ???**`

## 2. 테스트 환경
### 2.1 하드웨어 (ex.)
- CPU: Intel(R) Core(TM) Ultra 7 155H 3.80 GHz
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- RAM: 32GB

### 2.2 소프트웨어
- OS: Windows 10
- Python 버전: 3.10.6
- 딥러닝 프레임워크: `**PyTorch 2.0.1 ???**`

### 2.3 테스트 데이터셋
- 데이터셋 구성: 카테고리 수 / 데이터셋 구성 현황
- 이미지 수: 80장
- 해상도: 다양 (최소 256x256, 최대 4096x4096)
- 구성: MVTec-AD 70%, VISION-DATASETS 30%

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
| MAE | F-Measure |
|---------------|---------------|
|      1장         |               |
|      10장         |               |
|      40장         |               |

#### 4.1.2 Speed/GPU Usage 평가
| 표본 | 처리 시간 (ms) | GPU 메모리 (MB) |
|---------------|---------------|----------------|
|      1장         |               |                |
|      10장         |               |                |
|      40장         |               |                |

### 4.2 정성평가 (Qualitative Evaluation)
| 평가요소 | 사진 출력 제대로 됐는지 | 사진에 이상한 요소는 없는지(손가락) |
|---------------|---------------|----------------|
|      1장         |               |                |
|      10장         |               |                |
|      40장         |               |                |



### 4.3 아이디어가 있다면 추가 비교... (없어도됨)
### 4.4 이미지 크기 및 해상도 테스트
- 512x512

## 5. 테스트 절차
1. 테스트 환경 설정

2. 모델 로드

3. 테스트 데이터셋 준비
  
    - [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) 
  
    - [VISION-Workshop/VISION-Datasets](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)

4. 각 테스트 시나리오 실행
    
    a. 이미지 로드
    
    b. 배경 제거 수행
    
    c. 결과 저장 및 메트릭 계산

5. 결과 분석 및 보고서 작성

## 6. 결과 기록 및 분석 방법
### 6.1 결과 기록 템플릿
| 테스트 ID | 시나리오 | 이미지 | IoU | F1 Score | 처리 시간 (ms) | GPU 메모리 (MB) |
|-----------|---------|-------|-----|----------|---------------|----------------|
| TEST-001  |   이미지 한 장      |   `256*256`    |  ?   |    ?      |   200            |       4,333 MB         |
| TEST-002  |    이미지 한 장     |   `512*512`    |  ?   |     ?     |           400    |        4,333MB        |
| TEST-003  |   이미지 한 장      |   `2148*2148`    |  ?   |    ?      |   200            |       4,333 MB         |
| TEST-004  |    이미지 한 장     |   `2148*2148`    |  ?   |     ?     |           400    |        4,333MB        |

## 7. 성능 기준 (Acceptance Criteria)
 (이거는 우리 프로젝트의 목표이므로, 상황에 따라 유동적으로 변할 수 있음)
- MAE: ≤ 10
- 평균 F1 Score: ≥ 0.98
- 평균 처리 시간: ≤ 50ms (256x256 해상도 기준)
- 최대 GPU 메모리 사용량: ≤ 3GB

## 8. 버그 보고 및 추적 프로세스
1. 버그 발견 시 즉시 Jira 티켓 생성
2. 버그 재현 단계 상세히 기록
3. 버그의 심각도 및 우선순위 설정
4. 개발팀에 할당 및 해결 과정 추적
5. 해결된 버그에 대한 재테스트 수행

## 9. 테스트 결과 요약 및 보고서 템플릿
### 9.1 테스트 결과 요약
- 전체 테스트 케이스 수: X
- 통과한 테스트 케이스 수: Y
- 실패한 테스트 케이스 수: Z
- 전체 정확도 (Mean IoU): 0.XX
- 평균 처리 시간: XX ms
### 9.2 주요 발견사항
1. ...
2. ...
3. ...
### 9.3 개선 제안사항
1. ...
2. ...
3. ...
### 9.4 결론
[전반적인 모델 성능에 대한 요약 및 향후 방향 제시]