#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install fastapi uvicorn[standard] pandas pytz numpy matplotlib scikit-learn keras tensorflow pydot graphviz

'''설치 패키지 설명 :
# scikit-learn -> sklearn.preprocessing.MinMaxScaler, sklearn.metrics.mean_squared_error
# keras, tensorflow -> baseline 시계열 모델 구성 및 학습
# matplotlib -> 예측 결과 시각화
# 참고: 이미지 내보내기 실패 시 OS에 Graphviz 시스템 패키지(예: brew install graphviz, apt-get install graphviz)도 설치해야 합니다.'''

import os
import logging
from pathlib import Path
from threading import Lock

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, GRU, Input, LSTM
from tensorflow.keras.models import Model

try:
    from config import (
        MELBOURNE_BASELINE_PLOT_PATH,
        MELBOURNE_BOOTSTRAP_END_DATE,
        MELBOURNE_REFERENCE_DATA_PATH,
        MELBOURNE_TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from model_serving_rpt.config import (
        MELBOURNE_BASELINE_PLOT_PATH,
        MELBOURNE_BOOTSTRAP_END_DATE,
        MELBOURNE_REFERENCE_DATA_PATH,
        MELBOURNE_TARGET_COLUMN,
    )

from .melbourne_utils import (
    build_forecast_windows,
    compute_forecast_metrics,
    fit_scalers,
    inverse_target,
    prepare_melbourne_dataframe,
    read_tabular_csv,
    save_forecast_plot,
    slice_frame_by_date,
    split_dataframe,
    transform_feature_frame,
    transform_target_series,
)

logger = logging.getLogger(__name__)


# 데이터/모델 기본 설정
WINDOW_SIZE = int(os.getenv("MELBOURNE_BASELINE_WINDOW_SIZE", os.getenv("MELBOURNE_WINDOW_SIZE", "14")))
STRIDE = int(os.getenv("MELBOURNE_BASELINE_WINDOW_STRIDE", "1"))
EPOCHS = int(os.getenv("MELBOURNE_BASELINE_EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("MELBOURNE_BASELINE_BATCH_SIZE", "32"))
SEED = int(os.getenv("MELBOURNE_RANDOM_SEED", "42"))
FEATURE_PRESET = os.getenv(
    "MELBOURNE_BASELINE_FEATURE_PRESET",
    os.getenv("MELBOURNE_FEATURE_PRESET", "process_core_no_bod"),
).strip().lower()
SEQUENCE_MODEL_FAMILY = os.getenv(
    "MELBOURNE_BASELINE_SEQUENCE_MODEL",
    "gru",
).strip().lower()
MIN_REQUIRED_ROWS = max(WINDOW_SIZE * 4, 90)
MIN_UPLOAD_ROWS = max(WINDOW_SIZE + 5, 30)
SUPPORTED_SEQUENCE_MODELS = {"gru", "lstm"}

if SEQUENCE_MODEL_FAMILY not in SUPPORTED_SEQUENCE_MODELS:
    raise ValueError(
        f"Unsupported baseline sequence model '{SEQUENCE_MODEL_FAMILY}'. "
        f"Expected one of {sorted(SUPPORTED_SEQUENCE_MODELS)}."
    )


_BASELINE_BUNDLE_CACHE_LOCK = Lock()
_BASELINE_BUNDLE_CACHE_SIGNATURE: tuple | None = None
_BASELINE_BUNDLE_CACHE: dict | None = None


def _baseline_reference_signature() -> tuple:
    """기준 데이터와 설정값을 바탕으로 baseline 캐시 키를 만든다."""
    reference_path = Path(MELBOURNE_REFERENCE_DATA_PATH)
    reference_mtime = reference_path.stat().st_mtime_ns if reference_path.exists() else None
    return (
        reference_mtime,
        MELBOURNE_BOOTSTRAP_END_DATE,
        FEATURE_PRESET,
        SEQUENCE_MODEL_FAMILY,
        WINDOW_SIZE,
        STRIDE,
        EPOCHS,
        BATCH_SIZE,
    )


def _copy_baseline_bundle(bundle_payload: dict) -> dict:
    """요청 시 안전하게 쓰도록 baseline 캐시 번들을 복사한다."""
    runtime_bundle = dict(bundle_payload)
    runtime_bundle["feature_columns"] = list(bundle_payload.get("feature_columns", []))
    return runtime_bundle


def _model_label() -> str:
    """현재 baseline 시계열 모델의 표시용 이름을 반환한다."""
    return SEQUENCE_MODEL_FAMILY.upper()


def _baseline_model_name() -> str:
    """baseline 모델 아티팩트 이름을 만든다."""
    return f"melbourne_baseline_{SEQUENCE_MODEL_FAMILY}"


def _build_baseline_model(window_size: int, feature_count: int) -> Model:
    """기존 템플릿의 baseline 모델 구축 위치를 유지하면서 내부 구현만 교체"""
    layer_cls = GRU if SEQUENCE_MODEL_FAMILY == "gru" else LSTM
    inputs = Input(shape=(window_size, feature_count))
    hidden = layer_cls(32)(inputs)
    outputs = Dense(1)(hidden)
    model = Model(inputs, outputs, name=_baseline_model_name())
    model.compile(optimizer="adam", loss="mse")
    return model


def _make_windows(df, feature_columns, target_column, feature_scaler, target_scaler):
    """시계열 입력 window를 생성"""
    feature_values = transform_feature_frame(df, feature_columns, feature_scaler)
    target_values = transform_target_series(df, target_column, target_scaler)
    x_values, y_scaled, timestamps = build_forecast_windows(
        feature_values,
        target_values,
        df.index,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    y_actual = inverse_target(y_scaled, target_scaler)
    return x_values, y_scaled, y_actual, timestamps


def _load_reference_dataset():
    """학습 기준이 되는 reference 데이터를 불러옴"""
    reference_path = Path(MELBOURNE_REFERENCE_DATA_PATH)
    if not reference_path.exists():
        raise FileNotFoundError(f"Could not find data2 reference dataset: {reference_path}")
    return read_tabular_csv(reference_path), str(reference_path)


def _build_baseline_bundle() -> dict:
    """고정된 bootstrap 데이터로 baseline 시계열 모델을 학습한다."""
    tf.keras.utils.set_random_seed(SEED)

    reference_dataset, reference_path = _load_reference_dataset()
    reference_df, feature_columns, target_column = prepare_melbourne_dataframe(
        reference_dataset,
        reference_path,
        MELBOURNE_TARGET_COLUMN,
        feature_preset=FEATURE_PRESET,
    )
    reference_df = slice_frame_by_date(reference_df, end=MELBOURNE_BOOTSTRAP_END_DATE)

    if len(reference_df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"At least {MIN_REQUIRED_ROWS} rows are required for baseline {_model_label()} training.")

    train_df, val_df, _ = split_dataframe(reference_df)
    feature_scaler, target_scaler = fit_scalers(train_df, feature_columns, target_column)
    x_train, y_train_scaled, _, _ = _make_windows(train_df, feature_columns, target_column, feature_scaler, target_scaler)
    x_val, y_val_scaled, _, _ = _make_windows(val_df, feature_columns, target_column, feature_scaler, target_scaler)

    model = _build_baseline_model(WINDOW_SIZE, len(feature_columns))
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        x_train,
        y_train_scaled,
        validation_data=(x_val, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=[early_stop],
    )

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_reference": reference_path,
        "feature_preset": FEATURE_PRESET,
        "window_size": WINDOW_SIZE,
    }


def _load_or_train_baseline_bundle() -> dict:
    """baseline 캐시를 재사용하거나 현재 설정 기준으로 새로 학습한다."""
    global _BASELINE_BUNDLE_CACHE, _BASELINE_BUNDLE_CACHE_SIGNATURE

    signature = _baseline_reference_signature()
    with _BASELINE_BUNDLE_CACHE_LOCK:
        if _BASELINE_BUNDLE_CACHE is not None and _BASELINE_BUNDLE_CACHE_SIGNATURE == signature:
            return _copy_baseline_bundle(_BASELINE_BUNDLE_CACHE)

    baseline_bundle = _build_baseline_bundle()

    with _BASELINE_BUNDLE_CACHE_LOCK:
        _BASELINE_BUNDLE_CACHE = baseline_bundle
        _BASELINE_BUNDLE_CACHE_SIGNATURE = signature
        return _copy_baseline_bundle(_BASELINE_BUNDLE_CACHE)


# RMSE 계산 함수
def return_rmse(metrics: dict) -> str:
    """기존 UI가 기대하는 RMSE 문장 형식으로 반환한다."""
    rmse = float(metrics["rmse"])
    result_msg = f"The root mean squared error is {rmse}."
    logger.debug(result_msg)
    return result_msg


# 예측 결과 그래프 저장 함수
def plot_predictions(actual, predicted, timestamps, target_column):
    """공용 plotting 유틸을 사용해 baseline 예측 그래프를 저장한다."""
    return save_forecast_plot(
        timestamps,
        actual,
        predicted,
        MELBOURNE_BASELINE_PLOT_PATH,
        f"Baseline Melbourne {_model_label()} Forecast ({target_column})",
    )


# 추가된 process() 함수
def process(dataset, source_name: str = "") -> dict:
    """주어진 데이터셋으로 baseline 모델을 학습하고 예측을 수행"""
    baseline_bundle = _load_or_train_baseline_bundle()
    feature_columns = baseline_bundle["feature_columns"]
    target_column = baseline_bundle["target_column"]

    upload_df, upload_feature_columns, _ = prepare_melbourne_dataframe(
        dataset,
        source_name,
        MELBOURNE_TARGET_COLUMN,
        feature_preset=FEATURE_PRESET,
    )

    if upload_feature_columns != feature_columns:
        raise ValueError(
            f"Uploaded dataset feature columns {upload_feature_columns} do not match baseline features {feature_columns}."
        )

    if len(upload_df) < MIN_UPLOAD_ROWS:
        raise ValueError(f"At least {MIN_UPLOAD_ROWS} rows are required for baseline {_model_label()} evaluation.")

    x_test, _, y_test_actual, timestamps = _make_windows(
        upload_df,
        feature_columns,
        target_column,
        baseline_bundle["feature_scaler"],
        baseline_bundle["target_scaler"],
    )

    predicted_scaled = baseline_bundle["model"].predict(x_test, verbose=0).reshape(-1)
    predicted_actual = inverse_target(predicted_scaled, baseline_bundle["target_scaler"])
    metrics = compute_forecast_metrics(y_test_actual, predicted_actual)

    result_visualizing = plot_predictions(y_test_actual, predicted_actual, timestamps, target_column)
    result_evaluating = (
        f"{_baseline_model_name()} {target_column} RMSE={metrics['rmse']:.3f}, "
        f"MAE={metrics['mae']:.3f}, MAPE={metrics['mape']:.2f}%"
    )
    rmse_text = return_rmse(metrics)

    return {
        "model_name": _baseline_model_name(),
        "plot_path": result_visualizing,
        "result_visualizing": result_visualizing,
        "result_evaluating": result_evaluating,
        "rmse_text": rmse_text,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_reference": baseline_bundle["train_reference"],
        "feature_preset": FEATURE_PRESET,
        "window_size": WINDOW_SIZE,
        "metrics": {"forecast": metrics},
        "summary_text": result_evaluating,
    }
