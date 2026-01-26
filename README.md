딥러닝 기반 흉부근 정량화 파이프라인

A Python-based module for automatic thoracic muscle segmentation from chest CT (DICOM) images.

## Index

-   [참고 사항](#note)
-   [환경 설치](#installation)
-   [설정](#config)
-   [추론](#inference)
-   [학습](#train)
-   [테스트](#test)

## Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. 저장소 복제

```bash
git clone https://github.com/qqaazz0222/DEEP-TMSM
cd DEEP-TMSM
```

**Step 2**. Conda 가상환경 생성 및 활성화

```bash
conda create --name deep-tmqp python=3.9 -y
conda activate deep-tmqp
```

**Step 3** PyTorch 설치

사용자의 환경에 맞는 `torch`를 설치합니다.

```bash
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

CUDA 버전은 아래 명령어를 통해 확인 가능합니다.

```bash
nvcc -V
```

> 만약 위 목록에 사용자 환경의 CUDA 버전이 없다면, 바로 이전 버전의 명령어로 설치를 진행합니다. ex) CUDA 12.1 -> CUDA 11.8 사용

**Step 4**. 라이브러리 설치

```bash
pip install -r requirements.txt
```

## Config

모듈 실행을 위한 설정 파일은 `config.yaml`에 정의되어 있습니다.

```yaml
# =============================================================
# PROJECT PATHS CONFIGURATION (프로젝트 경로 설정)
# =============================================================
paths:
  data_root: "./data"                  # 데이터가 위치한 루트 디렉터리
  dicom_root: "./data/dicom"           # DICOM 형식의 CT 파일 위치
  mask_root: "./data/mask"             # 정답(Ground-truth) 세그멘테이션 마스크
  patients_csv: "./data/input.csv"     # 학습/검증용 전체 환자 목록 CSV
  test_patients_csv: "./data/test_input.csv"  # 테스트 단계용 환자 목록 CSV 파일
  outputs_root: "./outputs"            # 모든 결과(로그, 모델, 플롯)가 저장될 폴더
  splits_json: "./outputs/splits/patient_split_train_val.json"  # 학습/검증 데이터 분할 정보가 저장된 JSON

# =============================================================
# DATA LOADING & PREPROCESSING SETTINGS (데이터 로딩 및 전처리 설정)
# =============================================================
data:
  img_size: [512, 512]                 # 입력 CT 슬라이스 크기 (H, W)
  in_channels: 25                      # 모델에 입력되는 채널 수 (z-context + HU 윈도우)

  hu_windows:                          # CT 강도 처리를 위한 HU (Hounsfield Unit) 윈도우
    - [-400, 1600]                     # 연조직 + 뼈 (일반적인 범위)
    - [40, 400]                        # 근육 강조 윈도우
    - [-30, 80]                        # 좁은 범위의 연조직 윈도우
    - [-150, 2000]                     # 확장된 동적 범위
    - [-1000, -300]                    # 지방(Fat) 주 영역

  z_context: 2                         # 각 슬라이스의 전후 인접 슬라이스 개수
  paired: true                         # 흡기(Inhale)-호기(Exhale) 쌍으로 작업 여부
  pair_mode: "same_z"                  # 쌍 결합 전략 (동일한 z-인덱스 기준)
  val_percent: 0.2                     # 검증용(Validation)으로 분리할 데이터 비율

# =============================================================
# MODEL ARCHITECTURE SETTINGS (모델 아키텍처 설정)
# =============================================================
model:
  name: "siamese_scjag_unet"           # 모델명 (Siamese SCJAG U-Net)
  embed_dim: 128                       # 임베딩 잠재 공간(latent space) 차원
  decoders: "shared"                   # 디코더 공유 여부 (shared 또는 개별)
  dropout: 0.1                         # 정규화를 위한 드롭아웃(Dropout) 확률

# =============================================================
# LOSS FUNCTION CONFIGURATION (손실 함수 설정)
# =============================================================
loss:
  bce_weight: 0.5                      # 이진 교차 엔트로피(BCE) 손실 가중치
  dice_weight: 0.5                     # Dice 손실 가중치

  focal_tversky_weight: 0.0            # Focal Tversky 손실 가중치 (비활성화됨)
  tversky_alpha: 0.7                   # False Negative에 대한 페널티
  tversky_beta: 0.3                    # False Positive에 대한 페널티
  tversky_gamma: 2.0                   # Focal 강도 파라미터

  adv_enabled: true                    # 도메인 적대적 학습(Adversarial learning) 활성화
  lambda_adv: 0.05                     # 전체 손실에서 적대적 손실이 차지하는 비중
  adv_mode: "domain_adversarial"       # 적대적 전략 모드 "contrastive"
  emb_consistency: "l2"                # 흡기/호기 임베딩 일관성을 위한 L2 손실

# =============================================================
# TRAINING PARAMETERS (학습 파라미터)
# =============================================================
train:
  epochs: 100                          # 최대 학습 에포크(Epoch) 수
  batch_size: 16                       # 배치(Batch) 크기
  lr: 0.0015                           # 초기 학습률 (Learning Rate)
  weight_decay: 0.0005                 # L2 정규화(Weight Decay) 계수

  amp: true                            # 자동 혼합 정밀도(AMP) 사용 여부
  warmup_epochs: 5                     # 학습률 웜업(Warm-up) 에포크 수
  cosine_final_lr: 1.0e-05             # 코사인 스케줄러의 최종 학습률 값

  gradient_clip_val: 1.0               # 그래디언트 폭발 방지를 위한 클리핑 값
  ema_decay: 0.999                     # 지수 이동 평균(EMA) 감쇠율

  early_stopping:                      # 과적합(Overfitting) 방지 메커니즘
    enabled: true                      # 조기 종료(Early stopping) 활성화
    monitor: "val_dice"                # 모니터링할 지표
    mode: "max"                        # 최대값을 기준으로 선택
    patience: 15                       # 개선 없이 기다릴 에포크 수

  log_loss_components: true            # 각 손실 구성요소를 개별적으로 로그 기록
  log_components_in_val_only: true     # 검증 단계에서만 구성요소 로그 기록

  device: "cuda:3"                     # 사용할 GPU 장치
  gpu_ids: "3"                         # GPU ID (DataParallel용)
  num_workers: 8                       # 학습용 데이터 로더 워커 수
  num_workers_val: 4                   # 검증용 데이터 로더 워커 수

# =============================================================
# EVALUATION & TEST SETTINGS (평가 및 테스트 설정)
# =============================================================
eval:
  threshold: 0.5                       # 이진 세그멘테이션 임계값
  tta_flip: true                       # TTA (테스트 단계 증강) - 플립 사용
  tta_rot90: false                     # 90도 회전 TTA (비활성화됨)

  metrics_mode: "both"                 # 옵션: full | trim | both
  trim_head_pct: 10                    # 슬라이스 스택의 시작 부분 제외 비율 (%)
  trim_tail_pct: 20                    # 슬라이스 스택의 끝 부분 제외 비율 (%)

  overlays: sampled                    # 옵션: all | sampled → "all" = 모든 슬라이스, "sampled" = 일부 샘플
  overlays_sample_k: 4                 # 테스트 샘플당 저장할 오버레이 개수
  debug: false                         # 디버그 모드

# =============================================================
# REPRODUCIBILITY (재현성)
# =============================================================
seed: 2025
```

## Train

학습을 위해 아래 명령어를 실행합니다.

```bash
python train.py
```

TBU

## Test

테스트를 위해 아래 명령어를 실행합니다.

```bash
python test.py
```

TBU

## Inference

추론을 위해 아래 명령어를 실행합니다.

```bash
python inference.py
```

다중 프로세스를 활용하려면 아래 명령어를 실행합니다.

```bash
python inference_dist.py
```

### Input

추론을 위한 Dicom 데이터 및 디렉토리 구조를 아래와 같이 준비합니다.

```bash
.
├── dicom
│   ├── {$patient_id}
│   └── ...
└── input.csv
```

`input.csv`는 아래와 같이 구성됩니다.

만약, `input.csv`가 없다면 `data/dicom` 디렉토리의 모든 환자 목록으로 생성됩니다.

```csv
patient_id
{$patient_id}
...
```

**Example**

```bash
[Patient ID List]
 'p0001', 'p0002', 'p0003', ..., 'p1000'

[Directory Structure]
.
├── dicom
│   ├── p0001
│   ├── p0002
│   ├── p0003
│   ├── ...
│   └── p1000
└── input.csv

[input.csv]
patient_id
p0001
p0002
p0003
...
p1000
```

### Output

추론 완료 시, `outputs/predicted`에서 결과물을 확인할 수 있습니다.

```bash
└── outputs
    └── predicted
        ├── {$patient_id} # 환자 디렉토리
        │   └── {$study_date} # 스터디 날짜 디렉토리 (YYYYMMDD)
        │       ├── {$series_dir} # 시리즈 디렉토리 (Series{$Series Number}_{$Series Description})
        │       │   ├── pred # 추론 결과물 이미지 디렉토리
        │       │   ├── overlay # 오버랩 이미지 디렉토리
        │       │   └── counting.csv # 슬라이드 별 단면적(cm^2)과 부피(cm^3) 계산 결과
        │       └── ...
        ├── ...
        └── counting.csv # 분석한 전체 환자에 대한 단면적(cm^2)과 부피(cm^3) 계산 결과
```

**Sample**
| Pred | Overlay |

**counting.csv(시리즈)**

|     Column      |         Description          |
| :-------------: | :--------------------------: |
|  `Slice Index`  |     슬라이드 인덱스 번호     |
|  `Pixel Count`  | 슬라이드별 추출 근육 픽셀 수 |
|  `Size(cm^2)`   | 슬라이드별 추출 근육 단면적  |
| `Volume (cm^3)` |  슬라이드별 추출 근육 부피   |

**counting.csv(전체)**

|        Column         |         Description          |
| :-------------------: | :--------------------------: |
|     `Patient ID`      |           환자 ID            |
|        `Date`         |         스터디 일시          |
|       `Series`        |     시리즈 번호 및 설명      |
|  `Total Pixel Count`  | 슬라이드별 추출 근육 픽셀 수 |
|  `Total Size(cm^2)`   | 슬라이드별 추출 근육 단면적  |
| `Total Volume (cm^3)` |  슬라이드별 추출 근육 부피   |


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## README in Ennglish language

# Deep Learning-based Thoracic Muscle Quantification Pipeline
 
*Phase-Consistent Thoracic Muscle Segmentation in COPD Patients Using a Siamese
Attention U-Net with Adversarial Domain Adaptation.*

---

## Index
- [Notes](#notes)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Testing](#testing)
- [Inference](#inference)
- [Output Structure](#output-structure)

---

## Notes
All experiments were conducted in the following environment.  
Depending on your system, version compatibility may vary.

```
CPU: Intel(R) Core(TM) i9-13900KF  
GPU: Nvidia GeForce RTX 4090 (CUDA 12.1)  
OS: Ubuntu 24.04 LTS  
Conda: 25.5.1
```

---

## Installation
Set up a Conda-based Python environment to run this module.  
If Conda is not installed, download and install it first:

- [🔗 Anaconda Download](https://www.anaconda.com/download/success)
- [🔗 Miniconda Download](https://www.anaconda.com/docs/getting-started/miniconda/main)

### **Step 1 – Clone the repository**
```bash
git clone https://github.com/qqaazz0222/DEEP-TMSM
cd DEEP-TMSM
```

### **Step 2 – Create and activate the environment**
```bash
conda create --name deep-tmqp python=3.9 -y
conda activate deep-tmqp
```

### **Step 3 – Install PyTorch**
Install the version matching your CUDA toolkit:
```bash
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Check your CUDA version:
```bash
nvcc -V
```
> If your CUDA version is not listed, install the closest **previous** version (e.g., CUDA 12.1 → use 11.8).

### **Step 4 – Install required packages**
```bash
pip install -r requirements.txt
```

---

## Configuration
All module settings are defined in `config.yaml`:

```yaml
# =============================================================
# PROJECT PATHS CONFIGURATION
# =============================================================
paths:
  data_root: "./data"                  # Root directory where data is located
  dicom_root: "./data/dicom"           # Location of CT files in DICOM format
  mask_root: "./data/mask"             # Ground-truth segmentation masks
  patients_csv: "./data/input.csv"     # CSV list of all patients for training/validation
  test_patients_csv: "./data/test_input.csv"  # CSV list of patients for the testing phase
  outputs_root: "./outputs"             # Directory to save all results (logs, models, plots)
  splits_json: "./outputs/splits/patient_split_train_val.json"  # JSON file storing train/validation data split info

# =============================================================
# DATA LOADING & PREPROCESSING SETTINGS
# =============================================================
data:
  img_size: [512, 512]                  # Input CT slice size (H, W)
  in_channels: 25                       # Number of input channels for the model (z-context + HU windows)

  hu_windows:                           # HU (Hounsfield Unit) windows for CT intensity processing
    - [-400, 1600]                      # Soft tissue + bone (general range)
    - [40, 400]                         # Muscle-focused window
    - [-30, 80]                         # Narrow soft tissue window
    - [-150, 2000]                      # Extended dynamic range
    - [-1000, -300]                     # Fat-dominant region

  z_context: 2                          # Number of adjacent slices (before and after) for each slice
  paired: true                          # Whether to work with Inhale-Exhale pairs
  pair_mode: "same_z"                  # Pairing strategy (based on same z-index)
  val_percent: 0.2                     # Percentage of data reserved for validation

# =============================================================
# MODEL ARCHITECTURE SETTINGS
# =============================================================
model:
  name: "siamese_scjag_unet"            # Model name (Siamese SCJAG U-Net)
  embed_dim: 128                        # Dimension of the embedding latent space
  decoders: "shared"                   # Whether decoders are shared or separate
  dropout: 0.1                          # Dropout probability for regularization

# =============================================================
# LOSS FUNCTION CONFIGURATION
# =============================================================
loss:
  bce_weight: 0.5                       # Binary Cross Entropy (BCE) loss weight
  dice_weight: 0.5                      # Dice loss weight

  focal_tversky_weight: 0.0             # Focal Tversky loss weight (disabled)
  tversky_alpha: 0.7                    # Penalty for False Negatives
  tversky_beta: 0.3                     # Penalty for False Positives
  tversky_gamma: 2.0                    # Focal intensity parameter

  adv_enabled: true                     # Domain adversarial learning enabled
  lambda_adv: 0.05                      # Weight of adversarial loss in total loss
  adv_mode: "domain_adversarial"        # Adversarial strategy mode (e.g., "contrastive")
  emb_consistency: "l2"                # L2 loss for Inhale/Exhale embedding consistency

# =============================================================
# TRAINING PARAMETERS
# =============================================================
train:
  epochs: 100                           # Maximum number of training epochs
  batch_size: 16                        # Batch size
  lr: 0.0015                            # Initial learning rate
  weight_decay: 0.0005                 # L2 regularization (Weight Decay) coefficient

  amp: true                             # Whether to use Automatic Mixed Precision (AMP)
  warmup_epochs: 5                     # Number of learning rate warm-up epochs
  cosine_final_lr: 1.0e-05             # Final learning rate value for Cosine scheduler

  gradient_clip_val: 1.0               # Gradient clipping value to prevent explosion
  ema_decay: 0.999                     # Exponential Moving Average (EMA) decay rate

  early_stopping:                      # Mechanism to prevent overfitting
    enabled: true                      # Enable early stopping
    monitor: "val_dice"               # Metric to monitor
    mode: "max"                       # Select based on maximum value
    patience: 15                       # Number of epochs to wait without improvement

  log_loss_components: true             # Log each loss component individually
  log_components_in_val_only: true      # Log components only during validation phase

  device: "cuda:3"                     # GPU device to use
  gpu_ids: "3"                         # GPU IDs (for DataParallel)
  num_workers: 8                       # Number of workers for training DataLoader
  num_workers_val: 4                   # Number of workers for validation DataLoader

# =============================================================
# EVALUATION & TEST SETTINGS
# =============================================================
eval:
  threshold: 0.5                       # Binary segmentation threshold
  tta_flip: true                       # Use TTA (Test Time Augmentation) - flip
  tta_rot90: false                     # 90-degree rotation TTA (disabled)

  metrics_mode: "both"                 # Options: full | trim | both
  trim_head_pct: 10                    # Percentage to exclude from the start of the slice stack
  trim_tail_pct: 20                    # Percentage to exclude from the end of the slice stack

  overlays: sampled                    # Options: all | sampled -> "all" = all slices, "sampled" = selected samples
  overlays_sample_k: 4                 # Number of overlays to save per test sample
  debug: false                         # Debug mode

# =============================================================
# REPRODUCIBILITY
# =============================================================
seed: 2025                             # Random seed (for reproducibility)
```

---

## Training
To train the model:
```bash
python train.py
```
> (Detailed documentation coming soon.)

---

## Testing
To evaluate model performance on the test set:
```bash
python test.py
```
> (Detailed documentation coming soon.)

---

## Inference
Perform inference on new DICOM data:
```bash
python inference.py
```

To utilize multiple processes, run the following command.

```bash
python inference_dist.py


---

### Input Data
Prepare your DICOM data as follows:
```
.
├── dicom
│   ├── {patient_id}
│   └── ...
└── input.csv
```
If `input.csv` does not exist, it will be automatically created from all patient directories under `data/dicom`.

#### Example: `input.csv`
```csv
patient_id
p0001
p0002
p0003
...
p1000
```

#### Example: Directory structure
```
.
├── dicom
│   ├── p0001
│   ├── p0002
│   ├── p0003
│   └── ...
└── input.csv
```

---

## Output Structure
After inference, results are stored under `outputs/predicted`:
```
└── outputs
    └── predicted
        ├── {patient_id}
        │   └── {study_date}                 # YYYYMMDD
        │       ├── {series_dir}             # Series{Number}_{Description}
        │       │   ├── pred/                # Predicted mask images
        │       │   ├── overlay/             # Overlay visualizations
        │       │   └── counting.csv         # Per-slice metrics (cm², cm³)
        │       └── ...
        └── counting.csv                     # Aggregated results for all patients
```

---

### Sample Outputs
| Prediction | Overlay |
|:-----------:|:-------:|
| <img src="./resource/sample_pred.png" width="256"> | <img src="./resource/sample_overlay.png" width="256"> |

---

### counting.csv (per series)
| Column | Description |
|:------:|:-------------|
| **Slice Index** | Slice index number |
| **Pixel Count** | Number of extracted muscle pixels per slice |
| **Size (cm²)** | Cross-sectional area per slice |
| **Volume (cm³)** | Volume contribution per slice |

---

### counting.csv (overall)
| Column | Description |
|:------:|:-------------|
| **Patient ID** | Patient identifier |
| **Date** | Study date/time |
| **Series** | Series number and description |
| **Total Pixel Count** | Total extracted muscle pixel count |
| **Total Size (cm²)** | Total cross-sectional area |
| **Total Volume (cm³)** | Total muscle volume (cm³) |

---

### Contact
For questions, issues, or collaborations:  
**muhammad95@dgu.ac.kr**  
Dongguk University, Seoul — Medical AI Research Lab  


