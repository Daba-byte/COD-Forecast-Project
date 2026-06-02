# model_serving_rpt

데이터 기반 COD 예측과 자동 재학습/재배포 데모를 위한 FastAPI 프로젝트입니다.  
현재 버전은 `modeling/modeling.ipynb`의 변수 구성을 반영해 Melbourne 수처리 운영 데이터로 `Chemical Oxygen Demand`를 예측하고, 업로드 데이터의 `RMSE`가 기준을 넘으면 재학습 승인 보고서를 생성한 뒤 사용자가 승인하면 candidate `RandomForest`를 재학습/재배포합니다.

## 프로젝트 목표

과제의 메인 시나리오를 그대로 구현하는 것을 목표로 합니다.

1. REST API로 CSV 파일을 업로드합니다.
2. 배포 중인 RandomForest 모델이 COD 예측을 수행합니다.
3. `RMSE`, `MAE`, `MAPE`를 계산합니다.
4. `RMSE`가 기준보다 높으면 승인용 보고서를 생성합니다.
5. 사용자가 승인하면 candidate 모델을 재학습합니다.
6. candidate가 더 좋으면 active 모델로 재배포합니다.

## 과제 적합성

현재 프로젝트는 과제의 핵심 요구사항과 대체로 잘 맞습니다.

- `POST /upload`로 CSV를 업로드합니다.
- active RandomForest가 COD 예측을 수행합니다.
- `RMSE`, `MAE`, `MAPE`를 계산합니다.
- `RMSE`가 임계치를 넘으면 승인용 보고서를 생성합니다.
- 사용자가 승인하면 candidate RandomForest를 재학습합니다.
- candidate가 기존 active보다 더 좋을 때만 재배포합니다.

즉, 현재 구현의 핵심은 단순 예측 API가 아니라 다음 운영 루프입니다.

`업로드 -> 성능 측정 -> 성능 저하 감지 -> 재학습 검토 -> 재학습 -> 재배포`

현재 버전은 과제 문구에서 허용한 확장형 시나리오인 `LLM 보고서 + 사용자 승인 후 재학습/재배포` 구조를 사용합니다.

## 데이터

기본 데이터는 [`data2/Data-Melbourne_F_fixed.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/Data-Melbourne_F_fixed.csv) 입니다.

- 총 `1,382행`, `20컬럼`
- 예측 target 기본값: `Chemical Oxygen Demand`
- 주요 변수:
  - `Average Outflow`
  - `Average Inflow`
  - `Ammonia`
  - `Biological Oxygen Demand`
  - `Chemical Oxygen Demand`
  - `Total Nitrogen`
  - `Average Temperature`
  - `Average humidity`
  - `Total rainfall`
  - `Average wind speed`
  - `Year`, `Month`, `Day`

현재 파이프라인은 날짜 컬럼(`Year`, `Month`, `Day`)을 조합해 시계열 인덱스를 만들고, notebook에서 사용한 공정/기상 변수들로 같은 행의 `Chemical Oxygen Demand`를 예측합니다.

## 데이터 분할

재학습 데모를 위해 데이터를 시간 순서대로 5개 파일로 나눠두었습니다.

- [`data2/splits/train_2014_2017.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/train_2014_2017.csv): `1016행`
- [`data2/splits/test_2018_h1.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/test_2018_h1.csv): `127행`
- [`data2/splits/upload_1_2018_q3_q4.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_1_2018_q3_q4.csv): `88행`
- [`data2/splits/upload_2_2018_q4_2019_q1.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_2_2018_q4_2019_q1.csv): `70행`
- [`data2/splits/upload_3_2019_q2.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_3_2019_q2.csv): `81행`

의도한 데모 흐름은 다음과 같습니다.

1. `train` 구간으로 초기 active 모델을 학습합니다.
2. `test` 구간으로 기준선과 임계치를 잡습니다.
3. `upload_1`을 올려 성능 저하를 감지하고 승인 보고서를 확인합니다.
4. 승인 시 candidate 모델을 재학습하고 필요하면 재배포합니다.
5. 이후 `upload_2`, `upload_3`을 순차 업로드하며 성능 회복 여부를 확인합니다.

## 모델 구성

현재 `/upload`에서는 active 모델과 baseline 모델 결과를 함께 보여줍니다.

- `model_1`
  - 배포형 RandomForest
  - active 모델 평가
  - 승인 후 candidate 재학습/재배포 담당
- `model_2`
  - 비교용 baseline GRU
  - 현재 업로드 데이터에 대한 보조 비교 지표 제공

## 아키텍처

프로젝트는 크게 4개 계층으로 나뉩니다.

1. API 계층
   - FastAPI가 업로드, 승인, 다운로드 요청을 처리합니다.
2. 모델 계층
   - active/candidate 배포형 RandomForest와 baseline GRU가 분리되어 있습니다.
3. 리포트 계층
   - 성능 저하 시 승인 보고서를 생성합니다.
   - `OPENAI_API_KEY`가 있으면 OpenAI를 사용하고, 없으면 fallback 보고서를 사용합니다.
4. 프론트 계층
   - 업로드 결과를 시각화하고, 승인 필요 시 모달을 띄운 뒤 승인/보류 요청을 보냅니다.

주요 구현 파일은 다음과 같습니다.

- API 서버: [`server_model/main.py`](/Users/hyun/workspace/mlops/model_serving_rpt/server_model/main.py)
- 배포형 RandomForest / 재학습 로직: [`server_model/weight_used_model.py`](/Users/hyun/workspace/mlops/model_serving_rpt/server_model/weight_used_model.py)
- baseline GRU: [`server_model/model.py`](/Users/hyun/workspace/mlops/model_serving_rpt/server_model/model.py)
- 데이터 전처리 유틸: [`server_model/melbourne_utils.py`](/Users/hyun/workspace/mlops/model_serving_rpt/server_model/melbourne_utils.py)
- 승인 보고서 생성: [`server_model/llm_report.py`](/Users/hyun/workspace/mlops/model_serving_rpt/server_model/llm_report.py)
- 경로/설정: [`config.py`](/Users/hyun/workspace/mlops/model_serving_rpt/config.py)

## 요청 플로우

실제 요청 흐름은 아래와 같습니다.

1. 사용자가 CSV를 업로드합니다.
2. 서버가 업로드 파일을 저장하고 DataFrame으로 읽습니다.
3. active RandomForest가 예측과 `RMSE/MAE/MAPE` 평가를 수행합니다.
4. baseline GRU가 같은 업로드 데이터에 대해 비교 성능을 계산합니다.
5. active 모델의 `RMSE`가 threshold 이하이면 현재 모델을 유지합니다.
6. threshold 초과이면 승인 보고서를 생성합니다.
7. 프론트에서 모달을 띄우고 사용자가 승인 여부를 선택합니다.
8. 승인 시 candidate RandomForest를 재학습합니다.
9. candidate 성능이 기존 active보다 더 좋을 때만 active 모델로 승격합니다.

간단한 개념도:

```text
CSV Upload
  -> /upload
  -> active RandomForest evaluate
  -> baseline GRU evaluate
  -> RMSE threshold check
  -> approval report
  -> /approve-retrain
  -> candidate retrain
  -> better candidate only
  -> active redeploy
```

## 현재 성능 해석

현재 저장된 active 모델은 `Chemical Oxygen Demand`를 target으로 하는 `RandomForest + notebook_cod + window 1` 조합입니다.  
split 데이터 기준 성능은 대략 다음과 같습니다.

- `test_2018_h1`: `RMSE 114.08`, `MAPE 8.03%`
- `upload_1_2018_q3_q4`: `RMSE 95.49`, `MAPE 8.12%`
- `upload_2_2018_q4_2019_q1`: `RMSE 87.21`, `MAPE 7.50%`
- `upload_3_2019_q2`: `RMSE 59.70`, `MAPE 5.72%`

target인 `Chemical Oxygen Demand`의 전체 평균값이 대략 `846` 수준이라는 점을 고려하면, 현재 active 모델은 데모용 운영 판단에는 쓸 수 있지만 production 수준이라고 보긴 어렵습니다.  
다만 notebook 기반 RandomForest로 바꾸면서 기존 GRU 대비 오차가 크게 줄었고, baseline GRU보다도 더 낮은 `RMSE`를 보였습니다.

- `test_2018_h1`: active `114.08` vs baseline `202.82`

## 승인 보고서와 LLM

`OPENAI_API_KEY`가 설정되어 있으면 승인 보고서를 실제 OpenAI API로 생성합니다.  
키가 없거나 호출이 실패하면, 서버 내부 fallback 보고서를 사용합니다.

현재 보고서에는 다음 정보가 포함됩니다.

- 업로드 파일명
- target 컬럼
- `RMSE`, `MAE`, `MAPE`
- 기준 `RMSE`
- 심각도(`low`, `medium`, `high`)
- 권장 액션(`approve_retraining` 또는 `keep_active_model`)

## 환경변수

프로젝트 루트의 [`.env`](/Users/hyun/workspace/mlops/model_serving_rpt/.env) 파일을 자동으로 읽습니다.  
예시는 [`.env.example`](/Users/hyun/workspace/mlops/model_serving_rpt/.env.example)에 있습니다.

기본 예시:

```env
OPENAI_API_KEY=your_openai_api_key_here
MELBOURNE_LLM_MODEL=gpt-4o-mini
MELBOURNE_TARGET_COLUMN=Chemical Oxygen Demand
MELBOURNE_RMSE_THRESHOLD_FACTOR=1.05
MELBOURNE_SEQUENCE_MODEL=random_forest
```

`.env`는 [`.gitignore`](/Users/hyun/workspace/mlops/model_serving_rpt/.gitignore)에 포함되어 있으므로 Git에 올라가지 않습니다.

## 설치 패키지

예시:

```bash
cd /Users/hyun/workspace/mlops/model_serving_rpt
/Users/hyun/workspace/mlops/.venv/bin/pip install fastapi "uvicorn[standard]" pandas pytz python-multipart matplotlib tensorflow scikit-learn python-dotenv openai
```

## 실행 방법

프로젝트 루트에서 실행하는 것을 기준으로 합니다.

```bash
cd /Users/hyun/workspace/mlops/model_serving_rpt
/Users/hyun/workspace/mlops/.venv/bin/python -m uvicorn server_model.main:app --port 8001 --reload
```

접속 주소:

- 앱: [http://localhost:8001/](http://localhost:8001/)
- 정적 페이지: [http://localhost:8001/static/index.html](http://localhost:8001/static/index.html)
- Swagger 문서: [http://localhost:8001/docs](http://localhost:8001/docs)

## 주요 API

### `POST /upload`

CSV를 업로드하고 두 시계열 모델의 예측 결과를 반환합니다.

주요 응답 필드:

- `model_1`: 배포형 GRU 결과
- `model_2`: baseline GRU 결과
- `llm_summary`: 두 모델 요약
- `approval_required`: 승인 필요 여부
- `llm_report`: 승인 보고서
- `saved_filename`: 서버에 저장된 업로드 파일명

### `POST /approve-retrain`

`/upload` 결과에서 `approval_required=true`일 때 사용하는 승인 엔드포인트입니다.

요청 예시:

```json
{
  "saved_filename": "20260405_120000_upload_1_2018_q3_q4.csv",
  "approved": true
}
```

동작:

- `approved=true`: candidate 재학습 및 재배포 판단 수행
- `approved=false`: 재학습 보류, 기존 active 모델 유지

## 프론트 동작

기본 프론트는 [`public/index.html`](/Users/hyun/workspace/mlops/model_serving_rpt/public/index.html)과 [`public/js/post.js`](/Users/hyun/workspace/mlops/model_serving_rpt/public/js/post.js)에 있습니다.

- 파일 업로드
- 예측 그래프 출력
- 성능 지표 출력
- 승인 필요 시 모달 표시
- 승인/보류 버튼으로 `/approve-retrain` 호출

## 생성 아티팩트

실행 중 생성되는 주요 파일:

현재는 기존 경로 호환성을 위해 아티팩트 파일명에 `lstm` 문자열이 남아 있지만, 실제 저장 모델은 `model_name`/`model_family` 메타데이터 기준으로 `RandomForest`입니다.

- 업로드 파일: `server/uploaded_files/`
- active 모델:
  - `server/model/result/melbourne_lstm_active.keras`
  - `server/model/result/melbourne_lstm_active.pkl`
  - `server/model/result/melbourne_lstm_active.json`
- candidate 모델:
  - `server/model/result/melbourne_lstm_candidate.keras`
  - `server/model/result/melbourne_lstm_candidate.pkl`
  - `server/model/result/melbourne_lstm_candidate.json`
- 그래프:
  - `server/view-model-architecture/melbourne_lstm_predictor.png`
  - `server/view-model-architecture/melbourne_lstm_baseline.png`

## 빠른 데모 순서

1. 서버를 실행합니다.
2. 승인 모달을 확실히 보려면 [`data2/splits/test_2018_h1.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/test_2018_h1.csv)를 먼저 업로드합니다.
3. 승인 보고서와 `approval_required` 여부를 확인합니다.
4. 승인 버튼을 눌러 재학습/재배포를 수행합니다.
5. 이후 [`data2/splits/upload_1_2018_q3_q4.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_1_2018_q3_q4.csv), [`data2/splits/upload_2_2018_q4_2019_q1.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_2_2018_q4_2019_q1.csv), [`data2/splits/upload_3_2019_q2.csv`](/Users/hyun/workspace/mlops/model_serving_rpt/data2/splits/upload_3_2019_q2.csv)를 순차 업로드해 후속 성능을 확인합니다.

더 자세한 실제 시연 순서는 [`DEMO_RUNBOOK.md`](/Users/hyun/workspace/mlops/model_serving_rpt/DEMO_RUNBOOK.md)를 참고하세요.

## 참고

- 현재 구현은 과제용 데모에 맞춘 구조입니다.
- 승격 판단은 업로드 데이터 기준 RMSE 개선 여부에 따라 수행됩니다.
- 더 엄밀한 운영형 구조로 확장하려면 별도 holdout 승격 검증셋, 실험 추적, 모델 버전 관리가 추가로 필요합니다.

## 현재 구현상 한계

- candidate 승격 판단은 여전히 업로드 데이터 기준 RMSE 개선 여부에 의존합니다. 더 엄밀한 운영형 구조로 가려면 별도 holdout 승격 검증셋이 필요합니다.
- threshold는 production용 드리프트 정책이라기보다 과제/데모용 운영 임계치에 가깝습니다. 서비스 정책에 따라 더 보수적인 기준이나 다중 지표가 필요할 수 있습니다.
