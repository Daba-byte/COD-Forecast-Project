# 간단 API 명세서

기준 주소
- `http://localhost:8001`

## 1. 초기 화면 조회

### `GET /dashboard-summary`
메인 화면이 처음 열릴 때 요약 문구, 핵심 지표, 기준 그래프를 조회합니다.

주요 응답 필드
- `summary_text`: 화면 상단 설명
- `overview`: 현재 운영 모델 요약 정보
- `charts`: 초기 그래프 목록

응답 예시
```json
{
  "summary_text": "첫 화면에서 COD 기준 시계열과 calibration 예측을 보여줍니다.",
  "overview": {
    "target_column": "COD",
    "model_family": "random_forest",
    "threshold_rmse": 69.87,
    "active_mape": 8.03,
    "status_label": "기준선 로드됨"
  },
  "charts": [
    {
      "title": "방류수 COD 기준 시계열",
      "caption": "기준 데이터 흐름",
      "src": "data:image/png;base64,..."
    }
  ]
}
```

## 2. CSV 업로드 및 예측

### `POST /upload`
업로드한 CSV로 현재 active 모델 예측과 baseline 비교를 수행합니다. 성능 저하가 감지되면 재학습 검토 여부도 함께 반환합니다.

요청 형식
- `multipart/form-data`
- 필드명: `file`

요청 예시
```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@upload_1_2018_q3_q4.csv"
```

주요 응답 필드
- `result_visualizing_LSTM`: 현재 active 모델 그래프
- `result_evaluating_LSTM`: 현재 active 모델 평가 문구
- `result_visualizing_LSTM_v2`: baseline 그래프
- `result_evaluating_LSTM_v2`: baseline 평가 문구
- `llm_summary`: 요약 문구
- `approval_required`: 재학습 검토 필요 여부
- `llm_report`: 재학습 비교 보고서
- `saved_filename`: 서버 저장 파일명

응답 예시
```json
{
  "result_visualizing_LSTM": "data:image/png;base64,...",
  "result_evaluating_LSTM": "현재 모델 RMSE는 80.934 입니다.",
  "result_visualizing_LSTM_v2": "data:image/png;base64,...",
  "result_evaluating_LSTM_v2": "baseline 모델 RMSE는 66.100 입니다.",
  "llm_summary": "baseline 모델이 더 우수합니다.",
  "approval_required": true,
  "saved_filename": "20260407_101530_upload_1_2018_q3_q4.csv"
}
```

## 3. 재학습 실행 여부 결정

### `POST /approve-retrain`
재학습 검토 팝업에서 사용자가 재학습 실행 또는 유지 여부를 결정할 때 호출합니다.

요청 본문
```json
{
  "saved_filename": "20260407_101530_upload_1_2018_q3_q4.csv",
  "approved": true
}
```

요청 필드 설명
- `saved_filename`: `/upload` 결과로 받은 저장 파일명
- `approved`
  - `true`: 재학습 실행
  - `false`: 현재 모델 유지

주요 응답 필드
- `saved_filename`: 대상 파일명
- `approved`: 사용자 선택값
- `model_1`: 최종 처리 결과
- `llm_summary`: 최종 요약
- `result_visualizing_LSTM`: 최종 그래프

응답 예시
```json
{
  "saved_filename": "20260407_101530_upload_1_2018_q3_q4.csv",
  "approved": true,
  "llm_summary": "후보 모델이 더 좋아 active 모델로 반영되었습니다.",
  "result_visualizing_LSTM": "data:image/png;base64,..."
}
```

## 4. 화면 흐름

1. `GET /dashboard-summary`로 초기 화면을 구성합니다.
2. `POST /upload`로 CSV를 업로드합니다.
3. `approval_required=true`이면 `POST /approve-retrain`으로 재학습 여부를 결정합니다.
