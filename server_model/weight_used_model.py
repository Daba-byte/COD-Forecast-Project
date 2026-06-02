#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install pandas numpy matplotlib scikit-learn keras tensorflow pydot graphviz

'''설치 패키지 설명 :
# 데이터 처리 : pandas, numpy
# 시각화 : matplotlib
# 머신러닝/딥러닝 : scikit-learn, keras, tensorflow
# 모델 시각화 : pydot, graphviz
# 표준 라이브러리 : json, os, pickle, time'''

import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv1D, Dense, GRU, GlobalAveragePooling1D, Input, LSTM
from tensorflow.keras.models import Model, load_model

try:
    from config import (
        MELBOURNE_BOOTSTRAP_END_DATE,
        MELBOURNE_CALIBRATION_END_DATE,
        MELBOURNE_CANDIDATE_BUNDLE_PATH,
        MELBOURNE_CANDIDATE_METADATA_PATH,
        MELBOURNE_CANDIDATE_MODEL_PATH,
        MELBOURNE_DEPLOYED_BUNDLE_PATH,
        MELBOURNE_DEPLOYED_METADATA_PATH,
        MELBOURNE_DEPLOYED_MODEL_PATH,
        MELBOURNE_PREDICTOR_PLOT_PATH,
        MELBOURNE_REFERENCE_DATA_PATH,
        MELBOURNE_TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from model_serving_rpt.config import (
        MELBOURNE_BOOTSTRAP_END_DATE,
        MELBOURNE_CALIBRATION_END_DATE,
        MELBOURNE_CANDIDATE_BUNDLE_PATH,
        MELBOURNE_CANDIDATE_METADATA_PATH,
        MELBOURNE_CANDIDATE_MODEL_PATH,
        MELBOURNE_DEPLOYED_BUNDLE_PATH,
        MELBOURNE_DEPLOYED_METADATA_PATH,
        MELBOURNE_DEPLOYED_MODEL_PATH,
        MELBOURNE_PREDICTOR_PLOT_PATH,
        MELBOURNE_REFERENCE_DATA_PATH,
        MELBOURNE_TARGET_COLUMN,
    )

from .llm_report import generate_approval_report
from .melbourne_utils import (
    DATE_COLUMNS,
    build_forecast_windows,
    compute_forecast_metrics,
    fit_scalers,
    inverse_target,
    prepare_melbourne_dataframe,
    read_tabular_csv,
    save_forecast_plot,
    select_feature_columns,
    slice_frame_by_date,
    split_dataframe,
    transform_feature_frame,
    transform_target_series,
)

logger = logging.getLogger(__name__)


# 데이터/모델 기본 설정
DEFAULT_WINDOW_SIZE = int(os.getenv("MELBOURNE_WINDOW_SIZE", "1"))
DEFAULT_STRIDE = int(os.getenv("MELBOURNE_WINDOW_STRIDE", "1"))
EPOCHS = int(os.getenv("MELBOURNE_LSTM_EPOCHS", "16"))
BATCH_SIZE = int(os.getenv("MELBOURNE_BATCH_SIZE", "32"))
SEED = int(os.getenv("MELBOURNE_RANDOM_SEED", "42"))
RMSE_THRESHOLD_FACTOR = float(os.getenv("MELBOURNE_RMSE_THRESHOLD_FACTOR", "1.05"))
PROMOTION_MARGIN = float(os.getenv("MELBOURNE_PROMOTION_MARGIN", "0.99"))
PROMOTION_MAPE_MARGIN = float(os.getenv("MELBOURNE_PROMOTION_MAPE_MARGIN", "1.0"))
DEFAULT_MODEL_FAMILY = os.getenv("MELBOURNE_SEQUENCE_MODEL", "random_forest").strip().lower()
DEFAULT_FEATURE_PRESET = os.getenv("MELBOURNE_FEATURE_PRESET", "notebook_cod").strip().lower()
SEARCH_MODEL_FAMILIES = [
    item.strip().lower()
    for item in os.getenv("MELBOURNE_SEARCH_MODELS", "random_forest").split(",")
    if item.strip()
]
SEARCH_FEATURE_PRESETS = [
    item.strip().lower()
    for item in os.getenv("MELBOURNE_SEARCH_PRESETS", "notebook_cod").split(",")
    if item.strip()
]
SEARCH_WINDOW_SIZES = [
    int(item.strip())
    for item in os.getenv("MELBOURNE_SEARCH_WINDOWS", "1").split(",")
    if item.strip()
]
PIPELINE_VERSION = 9

DEEP_MODEL_FAMILIES = {"gru", "lstm", "conv1d"}
TABULAR_MODEL_FAMILIES = {"hist_gb", "extra_trees", "random_forest"}
ROW_ALIGNED_MODEL_FAMILIES = {"random_forest"}
SUPPORTED_MODEL_FAMILIES = DEEP_MODEL_FAMILIES | TABULAR_MODEL_FAMILIES

for family in [DEFAULT_MODEL_FAMILY, *SEARCH_MODEL_FAMILIES]:
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported Melbourne model family '{family}'. Expected one of {sorted(SUPPORTED_MODEL_FAMILIES)}."
        )


@dataclass(frozen=True)
class SearchSpec:
    """모델 종류, feature preset, window 크기로 이루어진 탐색 후보 1개를 정의한다."""
    model_family: str
    feature_preset: str
    window_size: int


@dataclass
class SearchRun:
    """학습이 끝난 후보 번들과 소요 시간을 함께 저장한다."""
    bundle: dict
    runtime_seconds: float


_ACTIVE_BUNDLE_CACHE_LOCK = Lock()
_ACTIVE_BUNDLE_CACHE_SIGNATURE: tuple | None = None
_ACTIVE_BUNDLE_CACHE: dict | None = None
_RETRAINING_PREVIEW_CACHE_LOCK = Lock()
_RETRAINING_PREVIEW_CACHE: dict[str, dict] = {}


def _artifact_mtime_ns(path: str) -> int | None:
    """캐시 무효화에 쓰는 아티팩트 수정 시각을 반환한다."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return artifact_path.stat().st_mtime_ns


def _runtime_bundle_signature(model_path: str, bundle_path: str, metadata_path: str) -> tuple:
    """저장된 런타임 번들의 캐시 시그니처를 만든다."""
    return (
        PIPELINE_VERSION,
        _artifact_mtime_ns(model_path),
        _artifact_mtime_ns(bundle_path),
        _artifact_mtime_ns(metadata_path),
    )


def _copy_runtime_bundle(bundle_payload: dict) -> dict:
    """호출 측이 공유 캐시를 직접 바꾸지 않도록 번들 메타데이터를 복사한다."""
    runtime_bundle = dict(bundle_payload)
    runtime_bundle["feature_columns"] = list(bundle_payload.get("feature_columns", []))
    runtime_bundle["validation_metrics"] = dict(bundle_payload.get("validation_metrics", {}))
    runtime_bundle["calibration_metrics"] = dict(bundle_payload.get("calibration_metrics", {}))
    runtime_bundle["search_leaderboard"] = list(bundle_payload.get("search_leaderboard", []))
    runtime_bundle["search_failures"] = list(bundle_payload.get("search_failures", []))
    return runtime_bundle


def _cache_active_bundle(bundle_payload: dict, model_path: str, bundle_path: str, metadata_path: str) -> None:
    """반복 추론 요청에 대비해 현재 active 번들을 메모리에 저장한다."""
    global _ACTIVE_BUNDLE_CACHE, _ACTIVE_BUNDLE_CACHE_SIGNATURE
    _ACTIVE_BUNDLE_CACHE = _copy_runtime_bundle(bundle_payload)
    _ACTIVE_BUNDLE_CACHE_SIGNATURE = _runtime_bundle_signature(model_path, bundle_path, metadata_path)


def _store_retraining_preview(source_name: str, preview_state: dict) -> None:
    """사용자가 재학습 여부를 결정할 때까지 비교 preview 결과를 캐시에 저장한다."""
    if not source_name:
        return
    with _RETRAINING_PREVIEW_CACHE_LOCK:
        _RETRAINING_PREVIEW_CACHE[source_name] = preview_state


def _pop_retraining_preview(source_name: str) -> dict | None:
    """업로드 파일 하나에 대한 재학습 preview를 꺼내면서 캐시에서 제거한다."""
    if not source_name:
        return None
    with _RETRAINING_PREVIEW_CACHE_LOCK:
        return _RETRAINING_PREVIEW_CACHE.pop(source_name, None)


def _clear_retraining_preview(source_name: str) -> None:
    """업로드 파일에 남아 있는 오래된 재학습 preview 캐시를 지운다."""
    if not source_name:
        return
    with _RETRAINING_PREVIEW_CACHE_LOCK:
        _RETRAINING_PREVIEW_CACHE.pop(source_name, None)


def _max_window_size() -> int:
    """현재 탐색 공간에서 사용하는 최대 window 크기를 반환한다."""
    return max([DEFAULT_WINDOW_SIZE, *SEARCH_WINDOW_SIZES])


def _min_required_rows(window_size: int) -> int:
    """주어진 window 크기로 모델을 학습하는 데 필요한 최소 이력 길이를 반환한다."""
    return max(window_size * 4, 90)


def _min_sequence_rows(window_size: int) -> int:
    """시계열 window를 최소 1개 만들기 위해 필요한 최소 행 수를 반환한다."""
    return max(window_size + 5, 19)


def _min_eval_rows(window_size: int) -> int:
    """평가나 승격 holdout 비교에 필요한 최소 행 수를 반환한다."""
    return max(window_size + 5, 30)


def _model_label(model_family: str | None = None) -> str:
    """내부 모델 이름을 화면 표시용 라벨로 바꾼다."""
    family = (model_family or DEFAULT_MODEL_FAMILY).strip().lower()
    labels = {
        "gru": "GRU",
        "lstm": "LSTM",
        "conv1d": "Conv1D",
        "hist_gb": "HistGB",
        "extra_trees": "ExtraTrees",
        "random_forest": "RandomForest",
    }
    return labels.get(family, family.upper())


def _model_identifier(role: str, model_family: str | None = None) -> str:
    """deployed 또는 candidate 모델에 사용할 고정된 아티팩트 이름을 만든다."""
    return f"melbourne_{role}_{(model_family or DEFAULT_MODEL_FAMILY).strip().lower()}"


def _safe_artifact_token(value: str) -> str:
    """그래프 파일명이 안전하도록 문자열을 파일 시스템 친화적으로 정리한다."""
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "uploaded_file").strip())
    token = token.strip("._-")
    return token or "uploaded_file"


def _comparison_plot_path(source_name: str, role: str) -> str:
    """업로드 파일과 역할에 맞는 비교 preview 그래프 경로를 만든다."""
    plot_dir = Path(MELBOURNE_PREDICTOR_PLOT_PATH).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    source_token = _safe_artifact_token(Path(source_name).stem if source_name else "uploaded_file")
    role_token = _safe_artifact_token(role)
    return str(plot_dir / f"{source_token}_{role_token}.png")


def return_rmse(metrics: dict) -> str:
    """기존 템플릿의 RMSE 문자열 형식을 유지"""
    rmse = float(metrics["rmse"])
    result_msg = f"The root mean squared error is {rmse}."
    logger.debug(result_msg)
    return result_msg


def plot_predictions(timestamps, actual, predicted, plot_path: str, plot_title: str) -> str:
    """기존 템플릿의 plot_predictions 역할을 현재 시계열 유틸에 맞게 래핑"""
    return save_forecast_plot(timestamps, actual, predicted, plot_path, plot_title)


def _make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_scaler,
    target_scaler,
    window_size: int,
    stride: int,
    model_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """학습·평가용 window를 만들고 tabular 모델은 행 정렬 방식으로 처리한다."""
    feature_values = transform_feature_frame(df, feature_columns, feature_scaler)
    target_values = transform_target_series(df, target_column, target_scaler)
    if model_family in ROW_ALIGNED_MODEL_FAMILIES:
        x_values = feature_values[::stride]
        y_scaled = target_values[::stride]
        timestamps = pd.DatetimeIndex(df.index[::stride])
    else:
        x_values, y_scaled, timestamps = build_forecast_windows(
            feature_values,
            target_values,
            df.index,
            window_size=window_size,
            stride=stride,
        )
    y_actual = inverse_target(y_scaled, target_scaler)
    return x_values, y_scaled, y_actual, timestamps


def _available_feature_space(df: pd.DataFrame) -> list[str]:
    """원본 날짜 컬럼을 제외한 모델링 후보 컬럼 목록을 반환한다."""
    return [column for column in df.columns if column not in DATE_COLUMNS]


def _build_deep_model(model_family: str, window_size: int, feature_count: int, name: str) -> Model:
    """요청한 딥러닝 계열 시계열 예측 모델 구조를 만든다."""
    inputs = Input(shape=(window_size, feature_count))

    if model_family == "gru":
        hidden = GRU(64, return_sequences=True)(inputs)
        hidden = GRU(32)(hidden)
    elif model_family == "lstm":
        hidden = LSTM(64, return_sequences=True)(inputs)
        hidden = LSTM(32)(hidden)
    elif model_family == "conv1d":
        hidden = Conv1D(64, kernel_size=3, activation="relu", padding="causal")(inputs)
        hidden = Conv1D(32, kernel_size=3, activation="relu", padding="causal")(hidden)
        hidden = GlobalAveragePooling1D()(hidden)
    else:
        raise ValueError(f"Unsupported deep model family: {model_family}")

    outputs = Dense(1)(hidden)
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_tabular_model(model_family: str):
    """요청한 scikit-learn 회귀 모델을 만든다."""
    if model_family == "hist_gb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=8,
            random_state=SEED,
        )
    if model_family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        )
    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported tabular model family: {model_family}")


def _fit_candidate_model(
    model_family: str,
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    x_val: np.ndarray,
    y_val_scaled: np.ndarray,
    model_name: str,
) -> object:
    """탐색 후보 1개에 대해 딥러닝 또는 tabular 모델을 학습한다."""
    if model_family in DEEP_MODEL_FAMILIES:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = _build_deep_model(model_family, x_train.shape[1], x_train.shape[2], model_name)
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
        return model

    model = _build_tabular_model(model_family)
    tabular_train = x_train.reshape((x_train.shape[0], -1)) if x_train.ndim > 2 else x_train
    model.fit(tabular_train, y_train_scaled)
    return model


def _predict_scaled(model_family: str, model: object, x_values: np.ndarray) -> np.ndarray:
    """딥러닝 또는 tabular 모델로 스케일된 예측값을 계산한다."""
    if model_family in DEEP_MODEL_FAMILIES:
        return np.asarray(model.predict(x_values, verbose=0)).reshape(-1)
    flattened = x_values.reshape((x_values.shape[0], -1)) if x_values.ndim > 2 else x_values
    return np.asarray(model.predict(flattened), dtype=np.float32).reshape(-1)


def _serialize_model(model_family: str, model: object, model_path: str) -> None:
    """모델 종류에 맞는 형식으로 학습된 모델을 저장한다."""
    if model_family in DEEP_MODEL_FAMILIES:
        model.save(model_path, overwrite=True)
        return
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def _deserialize_model(model_family: str, model_path: str) -> object:
    """저장된 모델 아티팩트를 메모리로 다시 불러온다."""
    if model_family in DEEP_MODEL_FAMILIES:
        return load_model(model_path)
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def _compute_rmse_threshold(
    validation_metrics: dict | None = None,
    calibration_metrics: dict | None = None,
) -> float:
    """validation과 calibration 결과를 바탕으로 서빙용 RMSE 기준선을 계산한다."""
    reference_rmses: list[float] = []
    for metrics in (validation_metrics or {}, calibration_metrics or {}):
        rmse = metrics.get("rmse")
        if rmse is not None:
            reference_rmses.append(float(rmse))

    reference_rmse = min(reference_rmses) if reference_rmses else 1.0
    return max(reference_rmse * RMSE_THRESHOLD_FACTOR, 1.0)


def _metadata_payload(trained_bundle: dict, model_path: str, bundle_path: str) -> dict:
    """저장된 모델 번들 옆에 둘 JSON 메타데이터 내용을 만든다."""
    return {
        "model_name": trained_bundle["model_name"],
        "model_family": trained_bundle["model_family"],
        "feature_preset": trained_bundle["feature_preset"],
        "pipeline_version": PIPELINE_VERSION,
        "model_path": model_path,
        "bundle_path": bundle_path,
        "feature_columns": trained_bundle["feature_columns"],
        "target_column": trained_bundle["target_column"],
        "window_size": int(trained_bundle["window_size"]),
        "stride": int(trained_bundle["stride"]),
        "rmse_threshold": float(trained_bundle["rmse_threshold"]),
        "validation_metrics": trained_bundle.get("validation_metrics", {}),
        "calibration_metrics": trained_bundle.get("calibration_metrics", {}),
        "train_reference": trained_bundle.get("train_reference", ""),
        "calibration_reference": trained_bundle.get("calibration_reference", ""),
        "search_space_size": int(trained_bundle.get("search_space_size", 0)),
        "search_summary": trained_bundle.get("search_summary", ""),
    }


def save_predictive_lstm_bundle(
    trained_bundle: dict,
    model_path: str = MELBOURNE_DEPLOYED_MODEL_PATH,
    bundle_path: str = MELBOURNE_DEPLOYED_BUNDLE_PATH,
    metadata_path: str = MELBOURNE_DEPLOYED_METADATA_PATH,
) -> dict:
    """학습된 모델, 런타임 번들, 메타데이터를 디스크에 저장한다."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)

    _serialize_model(trained_bundle["model_family"], trained_bundle["model"], model_path)
    bundle_payload = {
        "model_name": trained_bundle["model_name"],
        "model_family": trained_bundle["model_family"],
        "feature_preset": trained_bundle["feature_preset"],
        "pipeline_version": PIPELINE_VERSION,
        "feature_scaler": trained_bundle["feature_scaler"],
        "target_scaler": trained_bundle["target_scaler"],
        "feature_columns": trained_bundle["feature_columns"],
        "target_column": trained_bundle["target_column"],
        "window_size": int(trained_bundle["window_size"]),
        "stride": int(trained_bundle["stride"]),
        "rmse_threshold": float(trained_bundle["rmse_threshold"]),
        "validation_metrics": trained_bundle.get("validation_metrics", {}),
        "calibration_metrics": trained_bundle.get("calibration_metrics", {}),
        "train_reference": trained_bundle.get("train_reference", ""),
        "calibration_reference": trained_bundle.get("calibration_reference", ""),
        "search_space_size": int(trained_bundle.get("search_space_size", 0)),
        "search_summary": trained_bundle.get("search_summary", ""),
        "search_leaderboard": trained_bundle.get("search_leaderboard", []),
    }
    with open(bundle_path, "wb") as bundle_file:
        pickle.dump(bundle_payload, bundle_file)

    metadata = _metadata_payload(trained_bundle, model_path, bundle_path)
    Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    deployed_artifacts = (
        Path(MELBOURNE_DEPLOYED_MODEL_PATH).resolve(),
        Path(MELBOURNE_DEPLOYED_BUNDLE_PATH).resolve(),
        Path(MELBOURNE_DEPLOYED_METADATA_PATH).resolve(),
    )
    current_artifacts = (
        Path(model_path).resolve(),
        Path(bundle_path).resolve(),
        Path(metadata_path).resolve(),
    )
    if current_artifacts == deployed_artifacts:
        cached_bundle = dict(bundle_payload)
        cached_bundle["model"] = trained_bundle["model"]
        with _ACTIVE_BUNDLE_CACHE_LOCK:
            _cache_active_bundle(cached_bundle, model_path, bundle_path, metadata_path)

    return metadata


def load_saved_predictive_lstm(
    model_path: str = MELBOURNE_DEPLOYED_MODEL_PATH,
    bundle_path: str = MELBOURNE_DEPLOYED_BUNDLE_PATH,
    metadata_path: str = MELBOURNE_DEPLOYED_METADATA_PATH,
) -> dict | None:
    """active 모델 번들을 디스크에서 불러오고 필요하면 캐시와 threshold를 갱신한다."""
    signature = _runtime_bundle_signature(model_path, bundle_path, metadata_path)
    with _ACTIVE_BUNDLE_CACHE_LOCK:
        if _ACTIVE_BUNDLE_CACHE is not None and _ACTIVE_BUNDLE_CACHE_SIGNATURE == signature:
            return _copy_runtime_bundle(_ACTIVE_BUNDLE_CACHE)

    if not Path(model_path).exists() or not Path(bundle_path).exists():
        return None

    with open(bundle_path, "rb") as bundle_file:
        bundle_payload = pickle.load(bundle_file)
    if bundle_payload.get("pipeline_version") != PIPELINE_VERSION:
        return None

    recalculated_threshold = _compute_rmse_threshold(
        bundle_payload.get("validation_metrics", {}),
        bundle_payload.get("calibration_metrics", {}),
    )
    if float(bundle_payload.get("rmse_threshold", 0.0)) != float(recalculated_threshold):
        bundle_payload["rmse_threshold"] = float(recalculated_threshold)
        with open(bundle_path, "wb") as bundle_file:
            pickle.dump(bundle_payload, bundle_file)
        metadata = _metadata_payload(bundle_payload, model_path, bundle_path)
        Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        bundle_payload["rmse_threshold"] = float(recalculated_threshold)

    bundle_payload["model"] = _deserialize_model(bundle_payload["model_family"], model_path)
    with _ACTIVE_BUNDLE_CACHE_LOCK:
        _cache_active_bundle(bundle_payload, model_path, bundle_path, metadata_path)
        return _copy_runtime_bundle(_ACTIVE_BUNDLE_CACHE)


def _load_reference_dataset() -> tuple[pd.DataFrame, str]:
    """cold start와 재학습에 쓰는 기준 Melbourne 데이터를 불러온다."""
    reference_path = Path(MELBOURNE_REFERENCE_DATA_PATH)
    if not reference_path.exists():
        raise FileNotFoundError(f"Could not find data2 reference dataset: {reference_path}")
    return read_tabular_csv(reference_path), str(reference_path)


def _load_bootstrap_and_calibration_frames() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """고정 기준 데이터를 bootstrap 구간과 calibration 구간으로 나눈다."""
    reference_dataset, reference_path = _load_reference_dataset()
    reference_df, _, _ = prepare_melbourne_dataframe(
        reference_dataset,
        reference_path,
        MELBOURNE_TARGET_COLUMN,
        feature_preset="full",
    )

    bootstrap_df = slice_frame_by_date(reference_df, end=MELBOURNE_BOOTSTRAP_END_DATE)
    calibration_df = slice_frame_by_date(
        reference_df,
        start=str(pd.Timestamp(MELBOURNE_BOOTSTRAP_END_DATE) + pd.Timedelta(days=1)).split(" ")[0],
        end=MELBOURNE_CALIBRATION_END_DATE,
    )

    max_window = _max_window_size()
    if len(bootstrap_df) < _min_required_rows(max_window):
        raise ValueError(
            f"Bootstrap range through {MELBOURNE_BOOTSTRAP_END_DATE} only has {len(bootstrap_df)} rows; "
            f"at least {_min_required_rows(max_window)} are required."
        )
    if len(calibration_df) < max(max_window + 10, 30):
        raise ValueError(
            f"Calibration range through {MELBOURNE_CALIBRATION_END_DATE} only has {len(calibration_df)} rows; "
            f"at least {max(max_window + 10, 30)} are required."
        )

    return bootstrap_df, calibration_df, reference_path


def _train_bundle_from_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    search_spec: SearchSpec,
    model_name: str,
    train_reference: str = "",
    calibration_df: pd.DataFrame | None = None,
    calibration_reference: str = "",
) -> dict:
    """준비된 train/validation 프레임으로 탐색 후보 1개의 번들을 학습한다."""
    min_required_rows = _min_required_rows(search_spec.window_size)
    min_sequence_rows = _min_sequence_rows(search_spec.window_size)
    if search_spec.model_family in ROW_ALIGNED_MODEL_FAMILIES and search_spec.window_size != 1:
        raise ValueError(
            f"{_model_label(search_spec.model_family)} currently supports MELBOURNE_SEARCH_WINDOWS=1 only."
        )
    if len(train_df) < min_required_rows:
        raise ValueError(f"At least {min_required_rows} rows are required for {_model_label(search_spec.model_family)} forecasting.")
    if len(val_df) < min_sequence_rows:
        raise ValueError(f"At least {min_sequence_rows} rows are required for validation windows.")

    full_features = _available_feature_space(train_df)
    feature_columns = select_feature_columns(full_features, target_column, search_spec.feature_preset)
    feature_scaler, target_scaler = fit_scalers(train_df, feature_columns, target_column)

    x_train, y_train_scaled, _, _ = _make_windows(
        train_df,
        feature_columns,
        target_column,
        feature_scaler,
        target_scaler,
        window_size=search_spec.window_size,
        stride=DEFAULT_STRIDE,
        model_family=search_spec.model_family,
    )
    x_val, y_val_scaled, y_val_actual, _ = _make_windows(
        val_df,
        feature_columns,
        target_column,
        feature_scaler,
        target_scaler,
        window_size=search_spec.window_size,
        stride=DEFAULT_STRIDE,
        model_family=search_spec.model_family,
    )

    model = _fit_candidate_model(search_spec.model_family, x_train, y_train_scaled, x_val, y_val_scaled, model_name)
    val_pred_scaled = _predict_scaled(search_spec.model_family, model, x_val)
    val_pred_actual = inverse_target(val_pred_scaled, target_scaler)
    validation_metrics = compute_forecast_metrics(y_val_actual, val_pred_actual)

    calibration_metrics = None
    if calibration_df is not None and not calibration_df.empty:
        calibration_x, _, calibration_actual, _ = _make_windows(
            calibration_df,
            feature_columns,
            target_column,
            feature_scaler,
            target_scaler,
            window_size=search_spec.window_size,
            stride=DEFAULT_STRIDE,
            model_family=search_spec.model_family,
        )
        calibration_pred_scaled = _predict_scaled(search_spec.model_family, model, calibration_x)
        calibration_pred_actual = inverse_target(calibration_pred_scaled, target_scaler)
        calibration_metrics = compute_forecast_metrics(calibration_actual, calibration_pred_actual)

    rmse_threshold = _compute_rmse_threshold(validation_metrics, calibration_metrics)

    return {
        "model": model,
        "model_name": model_name,
        "model_family": search_spec.model_family,
        "feature_preset": search_spec.feature_preset,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "window_size": search_spec.window_size,
        "stride": DEFAULT_STRIDE,
        "rmse_threshold": float(rmse_threshold),
        "validation_metrics": validation_metrics,
        "calibration_metrics": calibration_metrics or {},
        "train_reference": train_reference,
        "calibration_reference": calibration_reference,
    }


def _candidate_sort_key(search_run: SearchRun) -> tuple[float, float, float, float, float]:
    """validation, calibration, 실행 시간을 기준으로 탐색 후보 정렬 키를 만든다."""
    validation_metrics = search_run.bundle.get("validation_metrics", {})
    calibration_metrics = search_run.bundle.get("calibration_metrics", {})
    return (
        float(validation_metrics.get("mape", float("inf"))),
        float(validation_metrics.get("rmse", float("inf"))),
        float(calibration_metrics.get("mape", float("inf"))),
        float(calibration_metrics.get("rmse", float("inf"))),
        float(search_run.runtime_seconds),
    )


def _search_leaderboard_entry(search_run: SearchRun) -> dict:
    """탐색 결과 1개를 보고서용 leaderboard 행으로 바꾼다."""
    bundle = search_run.bundle
    validation_metrics = bundle.get("validation_metrics", {})
    calibration_metrics = bundle.get("calibration_metrics", {})
    return {
        "model_name": bundle["model_name"],
        "model_family": bundle["model_family"],
        "feature_preset": bundle["feature_preset"],
        "window_size": int(bundle["window_size"]),
        "validation_mape": float(validation_metrics.get("mape", float("inf"))),
        "validation_rmse": float(validation_metrics.get("rmse", float("inf"))),
        "calibration_mape": float(calibration_metrics.get("mape", float("inf"))),
        "calibration_rmse": float(calibration_metrics.get("rmse", float("inf"))),
        "runtime_seconds": float(search_run.runtime_seconds),
    }


def _search_specs() -> list[SearchSpec]:
    """현재 환경 설정을 기준으로 전체 탐색 조합을 만든다."""
    return [
        SearchSpec(model_family=model_family, feature_preset=feature_preset, window_size=window_size)
        for model_family in SEARCH_MODEL_FAMILIES
        for feature_preset in SEARCH_FEATURE_PRESETS
        for window_size in SEARCH_WINDOW_SIZES
    ]


def _search_best_bundle_from_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    role: str,
    train_reference: str = "",
    calibration_df: pd.DataFrame | None = None,
    calibration_reference: str = "",
) -> dict:
    """설정된 탐색 공간을 돌고 가장 성능이 좋은 번들을 반환한다."""
    search_runs: list[SearchRun] = []
    failures: list[dict[str, object]] = []

    for spec in _search_specs():
        started_at = time.perf_counter()
        try:
            bundle = _train_bundle_from_frames(
                train_df,
                val_df,
                target_column,
                spec,
                model_name=_model_identifier(role, spec.model_family),
                train_reference=train_reference,
                calibration_df=calibration_df,
                calibration_reference=calibration_reference,
            )
            search_runs.append(
                SearchRun(
                    bundle=bundle,
                    runtime_seconds=time.perf_counter() - started_at,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "model_family": spec.model_family,
                    "feature_preset": spec.feature_preset,
                    "window_size": int(spec.window_size),
                    "error": str(exc),
                }
            )

    if not search_runs:
        raise ValueError(
            f"No candidate model could be trained from search space {SEARCH_MODEL_FAMILIES} x {SEARCH_FEATURE_PRESETS} x {SEARCH_WINDOW_SIZES}."
        )

    search_runs.sort(key=_candidate_sort_key)
    best_bundle = search_runs[0].bundle
    best_bundle["search_space_size"] = len(_search_specs())
    best_bundle["search_summary"] = (
        f"candidate search finished: {len(search_runs)} successful / {len(_search_specs())} total "
        f"across models={','.join(SEARCH_MODEL_FAMILIES)}, presets={','.join(SEARCH_FEATURE_PRESETS)}, "
        f"windows={','.join(str(item) for item in SEARCH_WINDOW_SIZES)}"
    )
    best_bundle["search_leaderboard"] = [_search_leaderboard_entry(search_run) for search_run in search_runs[:5]]
    best_bundle["search_failures"] = failures[:5]
    return best_bundle


def _build_candidate_frames(
    bootstrap_df: pd.DataFrame,
    upload_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """업로드 데이터로 candidate 학습 구간과 승격 비교용 holdout을 만든다."""
    if len(upload_df) < max(_min_sequence_rows(_max_window_size()), 30):
        raise ValueError(f"At least {max(_min_sequence_rows(_max_window_size()), 30)} rows are required for retraining uploads.")

    promotion_rows = max(_min_eval_rows(_max_window_size()), min(len(upload_df) // 3, 45))
    promotion_rows = min(promotion_rows, len(upload_df) - 1)

    recent_reference_rows = min(len(bootstrap_df), max(_min_required_rows(_max_window_size()), len(upload_df) * 4))
    recent_reference_df = bootstrap_df.tail(recent_reference_rows).copy()
    upload_history_df = upload_df.iloc[:-promotion_rows].copy()
    promotion_holdout_df = upload_df.iloc[-promotion_rows:].copy()

    candidate_history_df = pd.concat([recent_reference_df, upload_history_df]).sort_index()
    candidate_history_df = candidate_history_df[~candidate_history_df.index.duplicated(keep="last")]
    promotion_holdout_df = promotion_holdout_df[~promotion_holdout_df.index.duplicated(keep="last")]

    if len(candidate_history_df) < _min_required_rows(_max_window_size()):
        raise ValueError(
            f"Retraining history only has {len(candidate_history_df)} rows; "
            f"at least {_min_required_rows(_max_window_size())} are required."
        )

    candidate_train_df, candidate_val_df, _ = split_dataframe(candidate_history_df)
    return candidate_train_df, candidate_val_df, promotion_holdout_df


def _should_promote_candidate(
    previous_rmse: float,
    candidate_rmse: float,
    previous_mape: float,
    candidate_mape: float,
) -> tuple[bool, str]:
    """비교 결과를 바탕으로 candidate가 active를 대체할지 결정한다."""
    rmse_improved = candidate_rmse < previous_rmse * PROMOTION_MARGIN
    mape_improved = candidate_mape < previous_mape * PROMOTION_MAPE_MARGIN

    if rmse_improved and mape_improved:
        return True, "candidate_improved_rmse_and_mape"
    if rmse_improved:
        return False, "candidate_mape_not_improved"
    if mape_improved:
        return False, "candidate_rmse_not_improved"
    return False, "candidate_not_better_than_active"


def _build_retraining_preview_payload(
    active_result: dict,
    candidate_bundle: dict,
    active_promotion_result: dict,
    candidate_result: dict,
    should_promote: bool,
    promotion_reason: str,
    promotion_holdout_rows: int,
    active_plot_path: str | None = None,
    candidate_plot_path: str | None = None,
) -> dict:
    """active와 candidate 비교 preview에 사용할 프런트 payload를 만든다."""
    active_serving_metrics = dict(active_result["metrics"]["forecast"])
    active_holdout_metrics = dict(active_promotion_result["metrics"]["forecast"])
    candidate_holdout_metrics = dict(candidate_result["metrics"]["forecast"])
    recommendation_title = "재학습 결과 적용 권장" if should_promote else "현재 모델 유지 권장"
    recommendation_body = (
        "업로드 직후 candidate를 미리 재학습해 비교한 결과, holdout 기준으로 candidate가 active보다 더 좋았습니다. "
        "승인하면 candidate를 새 active 모델로 적용하는 흐름이 적절합니다."
        if should_promote
        else "업로드 직후 candidate를 미리 재학습해 비교했지만, holdout 기준으로 active보다 충분히 좋아지지 않았습니다. "
        "지금은 현재 active 모델을 유지하는 편이 더 안전합니다."
    )

    return {
        "title": "재학습 후보 비교 보고서",
        "summary": (
            "업로드 시점에 candidate 모델을 먼저 재학습해 holdout 구간에서 기존 active 모델과 비교했습니다. "
            "아래 비교 결과를 보고 적용 여부를 결정할 수 있습니다."
        ),
        "recommendation_title": recommendation_title,
        "recommendation_body": recommendation_body,
        "predicted_redeployment": bool(should_promote),
        "promotion_reason": promotion_reason,
        "promotion_holdout_rows": int(promotion_holdout_rows),
        "search_space_size": int(candidate_bundle.get("search_space_size", 0)),
        "search_summary": candidate_bundle.get("search_summary", ""),
        "current_upload_metrics": active_serving_metrics,
        "active_holdout_metrics": active_holdout_metrics,
        "candidate_holdout_metrics": candidate_holdout_metrics,
        "active_model": {
            "model_name": active_result["model_name"],
            "model_family": active_result["model_family"],
            "feature_preset": active_result.get("feature_preset", DEFAULT_FEATURE_PRESET),
            "window_size": int(active_result.get("window_size", DEFAULT_WINDOW_SIZE)),
        },
        "candidate_model": {
            "model_name": candidate_bundle["model_name"],
            "model_family": candidate_bundle["model_family"],
            "feature_preset": candidate_bundle["feature_preset"],
            "window_size": int(candidate_bundle["window_size"]),
        },
        "active_plot_path": active_plot_path,
        "candidate_plot_path": candidate_plot_path,
        "comparison_points": [
            f"현재 active 모델의 전체 업로드 성능은 MAPE {active_serving_metrics['mape']:.2f}%, RMSE {active_serving_metrics['rmse']:.3f} 입니다.",
            f"승격 비교 holdout에서 active는 MAPE {active_holdout_metrics['mape']:.2f}%, RMSE {active_holdout_metrics['rmse']:.3f} 를 기록했습니다.",
            f"같은 holdout에서 candidate는 MAPE {candidate_holdout_metrics['mape']:.2f}%, RMSE {candidate_holdout_metrics['rmse']:.3f} 를 기록했습니다.",
            f"이번 비교는 업로드 tail holdout {int(promotion_holdout_rows)}행 기준이며, 탐색한 후보 수는 {int(candidate_bundle.get('search_space_size', 0))}개입니다.",
        ],
    }


def _prepare_retraining_preview_state(
    dataset: pd.DataFrame,
    source_name: str,
    active_bundle: dict,
    active_result: dict,
    served_from: str,
) -> dict:
    """active 모델이 기준선을 넘었을 때 candidate 재학습 비교 산출물을 준비한다."""
    trigger_metrics = dict(active_result["metrics"]["forecast"])
    trigger_rmse = float(trigger_metrics["rmse"])
    trigger_mape = float(trigger_metrics["mape"])
    threshold_rmse = float(active_bundle["rmse_threshold"])

    if trigger_rmse <= threshold_rmse:
        return {
            "approval_required": False,
            "active_bundle": active_bundle,
            "active_result": active_result,
            "served_from": served_from,
            "threshold_rmse": threshold_rmse,
            "trigger_metrics": trigger_metrics,
            "trigger_rmse": trigger_rmse,
            "trigger_mape": trigger_mape,
        }

    bootstrap_df, calibration_df, reference_path = _load_bootstrap_and_calibration_frames()
    upload_df, upload_feature_columns, target_column = prepare_melbourne_dataframe(
        dataset,
        source_name,
        MELBOURNE_TARGET_COLUMN,
        feature_preset="full",
    )
    missing_active_features = [feature for feature in active_bundle["feature_columns"] if feature not in upload_feature_columns]
    if missing_active_features:
        raise ValueError(
            f"Uploaded dataset is missing active model features: {missing_active_features}."
        )

    candidate_train_df, candidate_val_df, promotion_holdout_df = _build_candidate_frames(bootstrap_df, upload_df)
    active_preview_plot_path = _comparison_plot_path(source_name, "active_holdout_preview")
    candidate_preview_plot_path = _comparison_plot_path(source_name, "candidate_holdout_preview")
    active_promotion_result = evaluate_predictive_lstm(
        promotion_holdout_df,
        active_bundle,
        f"{source_name}::promotion_holdout",
        plot_path=active_preview_plot_path,
        plot_title=f"Active Melbourne {_model_label(active_bundle['model_family'])} Holdout Forecast",
    )
    previous_rmse = float(active_promotion_result["metrics"]["forecast"]["rmse"])
    previous_mape = float(active_promotion_result["metrics"]["forecast"]["mape"])

    candidate_bundle = _search_best_bundle_from_frames(
        candidate_train_df,
        candidate_val_df,
        target_column,
        role="candidate",
        train_reference=f"{reference_path}::recent_bootstrap_tail+{source_name}::excluding_promotion_holdout",
        calibration_df=calibration_df,
        calibration_reference=f"{reference_path}::{MELBOURNE_BOOTSTRAP_END_DATE}..{MELBOURNE_CALIBRATION_END_DATE}",
    )
    save_predictive_lstm_bundle(
        candidate_bundle,
        model_path=MELBOURNE_CANDIDATE_MODEL_PATH,
        bundle_path=MELBOURNE_CANDIDATE_BUNDLE_PATH,
        metadata_path=MELBOURNE_CANDIDATE_METADATA_PATH,
    )

    candidate_result = evaluate_predictive_lstm(
        promotion_holdout_df,
        candidate_bundle,
        f"{source_name}::promotion_holdout",
        plot_path=candidate_preview_plot_path,
        plot_title=f"Best Candidate Melbourne {_model_label(candidate_bundle['model_family'])} Holdout Forecast",
    )
    candidate_rmse = float(candidate_result["metrics"]["forecast"]["rmse"])
    candidate_mape = float(candidate_result["metrics"]["forecast"]["mape"])
    should_promote, promotion_reason = _should_promote_candidate(
        previous_rmse,
        candidate_rmse,
        previous_mape,
        candidate_mape,
    )

    return {
        "approval_required": True,
        "active_bundle": active_bundle,
        "active_result": active_result,
        "served_from": served_from,
        "threshold_rmse": threshold_rmse,
        "trigger_metrics": trigger_metrics,
        "trigger_rmse": trigger_rmse,
        "trigger_mape": trigger_mape,
        "active_promotion_result": active_promotion_result,
        "candidate_bundle": candidate_bundle,
        "candidate_result": candidate_result,
        "previous_rmse": previous_rmse,
        "previous_mape": previous_mape,
        "candidate_rmse": candidate_rmse,
        "candidate_mape": candidate_mape,
        "should_promote": should_promote,
        "promotion_reason": promotion_reason,
        "promotion_holdout_rows": int(len(promotion_holdout_df)),
        "promotion_evaluation_scope": "upload_holdout_tail",
        "search_space_size": int(candidate_bundle.get("search_space_size", 0)),
        "search_summary": candidate_bundle.get("search_summary", ""),
        "comparison_preview": _build_retraining_preview_payload(
            active_result,
            candidate_bundle,
            active_promotion_result,
            candidate_result,
            should_promote,
            promotion_reason,
            len(promotion_holdout_df),
            active_preview_plot_path,
            candidate_preview_plot_path,
        ),
    }


def _ensure_active_bundle() -> tuple[dict, str]:
    """배포된 active 번들을 불러오거나 없으면 기준 데이터로 새로 만든다."""
    active_bundle = load_saved_predictive_lstm()
    served_from = "deployed_model"
    if active_bundle is None:
        bootstrap_df, calibration_df, reference_path = _load_bootstrap_and_calibration_frames()
        train_df, val_df, _ = split_dataframe(bootstrap_df)
        active_bundle = _search_best_bundle_from_frames(
            train_df,
            val_df,
            MELBOURNE_TARGET_COLUMN,
            role="deployed",
            train_reference=f"{reference_path}::bootstrap<={MELBOURNE_BOOTSTRAP_END_DATE}",
            calibration_df=calibration_df,
            calibration_reference=f"{reference_path}::{MELBOURNE_BOOTSTRAP_END_DATE}..{MELBOURNE_CALIBRATION_END_DATE}",
        )
        save_predictive_lstm_bundle(active_bundle)
        served_from = "cold_start_search"
    return active_bundle, served_from


def train_predictive_lstm(
    dataset: pd.DataFrame,
    source_name: str = "",
    model_name: str | None = None,
    calibration_df: pd.DataFrame | None = None,
    calibration_reference: str = "",
) -> dict:
    """주어진 데이터셋으로 기본 설정의 단일 모델 번들을 학습한다."""
    tf.keras.utils.set_random_seed(SEED)

    df, _, target_column = prepare_melbourne_dataframe(
        dataset,
        source_name,
        MELBOURNE_TARGET_COLUMN,
        feature_preset=DEFAULT_FEATURE_PRESET,
    )
    train_df, val_df, _ = split_dataframe(df)
    spec = SearchSpec(
        model_family=DEFAULT_MODEL_FAMILY,
        feature_preset=DEFAULT_FEATURE_PRESET,
        window_size=DEFAULT_WINDOW_SIZE,
    )
    return _train_bundle_from_frames(
        train_df,
        val_df,
        target_column,
        spec,
        model_name=model_name or _model_identifier("deployed", spec.model_family),
        train_reference=source_name,
        calibration_df=calibration_df,
        calibration_reference=calibration_reference,
    )


def evaluate_predictive_lstm(
    dataset: pd.DataFrame,
    trained_bundle: dict,
    source_name: str = "",
    plot_path: str | None = MELBOURNE_PREDICTOR_PLOT_PATH,
    plot_title: str | None = None,
) -> dict:
    """학습된 번들을 데이터셋으로 평가하고 필요하면 예측 그래프를 저장한다."""
    df, feature_columns, target_column = prepare_melbourne_dataframe(
        dataset,
        source_name,
        trained_bundle["target_column"],
        feature_preset=trained_bundle.get("feature_preset", DEFAULT_FEATURE_PRESET),
    )

    expected_feature_columns = trained_bundle["feature_columns"]
    if feature_columns != expected_feature_columns:
        raise ValueError(
            f"Uploaded dataset feature columns {feature_columns} do not match model features {expected_feature_columns}."
        )

    window_size = int(trained_bundle.get("window_size", DEFAULT_WINDOW_SIZE))
    stride = int(trained_bundle.get("stride", DEFAULT_STRIDE))
    if len(df) < _min_eval_rows(window_size):
        raise ValueError(f"At least {_min_eval_rows(window_size)} rows are required for evaluation.")

    x_test, y_test_scaled, y_test_actual, timestamps = _make_windows(
        df,
        feature_columns,
        target_column,
        trained_bundle["feature_scaler"],
        trained_bundle["target_scaler"],
        window_size=window_size,
        stride=stride,
        model_family=trained_bundle["model_family"],
    )

    predicted_scaled = _predict_scaled(trained_bundle["model_family"], trained_bundle["model"], x_test)
    predicted_actual = inverse_target(predicted_scaled, trained_bundle["target_scaler"])
    metrics = compute_forecast_metrics(y_test_actual, predicted_actual)

    saved_plot_path = None
    if plot_path is not None:
        saved_plot_path = plot_predictions(
            timestamps,
            y_test_actual,
            predicted_actual,
            plot_path,
            f"{plot_title or f'Melbourne {_model_label(trained_bundle['model_family'])} Forecast'} ({target_column})",
        )

    result_evaluating = (
        f"{trained_bundle['model_name']} {target_column} MAPE={metrics['mape']:.2f}%, "
        f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}"
    )
    rmse_text = return_rmse(metrics)

    return {
        "model_name": trained_bundle["model_name"],
        "model_family": trained_bundle["model_family"],
        "feature_preset": trained_bundle.get("feature_preset", DEFAULT_FEATURE_PRESET),
        "window_size": window_size,
        "plot_path": saved_plot_path,
        "result_visualizing": saved_plot_path,
        "result_evaluating": result_evaluating,
        "rmse_text": rmse_text,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_reference": trained_bundle.get("train_reference", ""),
        "calibration_reference": trained_bundle.get("calibration_reference", ""),
        "validation_metrics": trained_bundle.get("validation_metrics", {}),
        "calibration_metrics": trained_bundle.get("calibration_metrics", {}),
        "metrics": {"forecast": metrics},
        "rmse_threshold": float(trained_bundle["rmse_threshold"]),
        "prediction_count": int(len(predicted_actual)),
        "search_space_size": int(trained_bundle.get("search_space_size", 0)),
        "search_summary": trained_bundle.get("search_summary", ""),
        "search_leaderboard": trained_bundle.get("search_leaderboard", []),
        "summary_text": result_evaluating,
    }


# 추가된 함수: active 모델 점검
def inspect_predictive_lstm(dataset: pd.DataFrame, source_name: str = "") -> dict:
    """현재 active 모델을 평가하고 기준선 비교 결과를 함께 붙여 반환한다."""
    active_bundle, served_from = _ensure_active_bundle()
    active_result = evaluate_predictive_lstm(
        dataset,
        active_bundle,
        source_name,
        plot_path=MELBOURNE_PREDICTOR_PLOT_PATH,
        plot_title=f"Active Melbourne {_model_label(active_bundle['model_family'])} Forecast",
    )
    threshold_rmse = float(active_bundle["rmse_threshold"])
    current_rmse = active_result["metrics"]["forecast"]["rmse"]
    current_mape = active_result["metrics"]["forecast"]["mape"]
    approval_required = current_rmse > threshold_rmse
    active_result.update(
        {
            "served_from": served_from,
            "retrained": False,
            "redeployed": False,
            "approval_required": approval_required,
            "promotion_reason": "approval_required" if approval_required else "threshold_not_exceeded",
            "llm_report": generate_approval_report(
                source_name or "uploaded_file",
                active_result,
                threshold_rmse,
                f"2014-01-01~{MELBOURNE_BOOTSTRAP_END_DATE}",
                f"2018-01-01~{MELBOURNE_CALIBRATION_END_DATE}",
            ),
            "summary_text": (
                f"{active_result['model_name']} {active_result['target_column']} MAPE={current_mape:.2f}%, "
                f"RMSE={current_rmse:.3f}, approval_required={approval_required}"
            ),
        }
    )
    return active_result


def preview_retraining_decision(dataset: pd.DataFrame, source_name: str = "") -> dict:
    """승인 전에 UI가 비교할 수 있도록 candidate 비교 데이터를 미리 계산한다."""
    active_bundle, served_from = _ensure_active_bundle()
    active_result = inspect_predictive_lstm(dataset, source_name)
    preview_state = _prepare_retraining_preview_state(
        dataset,
        source_name,
        active_bundle,
        active_result,
        served_from,
    )

    if not preview_state["approval_required"]:
        _clear_retraining_preview(source_name)
        return active_result

    _store_retraining_preview(source_name, preview_state)
    active_result.update(
        {
            "comparison_preview": preview_state["comparison_preview"],
            "preview_ready": True,
            "summary_text": (
                f"{active_result['model_name']} 현재 업로드 MAPE={preview_state['trigger_mape']:.2f}%, "
                f"candidate holdout MAPE={preview_state['candidate_mape']:.2f}%, "
                f"predicted_redeployment={preview_state['should_promote']}"
            ),
        }
    )
    return active_result


# 추가된 함수: 기존 템플릿의 process() 진입점 유지
def process(dataset: pd.DataFrame, source_name: str = "") -> dict:
    """업로드 시점에 필요한 경우 candidate 비교까지 미리 계산한다."""
    return preview_retraining_decision(dataset, source_name)


# 추가된 함수: 승인 후 재탐색 기반 재학습/재배포
def approve_retraining(dataset: pd.DataFrame, source_name: str = "", approved: bool = True) -> dict:
    """사용자의 재학습 결정을 적용하고 필요하면 candidate를 승격한다."""
    preview_state = _pop_retraining_preview(source_name)
    if preview_state is None:
        active_bundle, served_from = _ensure_active_bundle()
        active_result = inspect_predictive_lstm(dataset, source_name)
        preview_state = _prepare_retraining_preview_state(
            dataset,
            source_name,
            active_bundle,
            active_result,
            served_from,
        )

    active_result = dict(preview_state["active_result"])
    served_from = preview_state["served_from"]
    trigger_metrics = dict(preview_state["trigger_metrics"])
    trigger_rmse = float(preview_state["trigger_rmse"])
    trigger_mape = float(preview_state["trigger_mape"])
    threshold_rmse = float(preview_state["threshold_rmse"])

    if not approved:
        active_result.update(
            {
                "served_from": served_from,
                "retrained": False,
                "redeployed": False,
                "approval_required": False,
                "approved": False,
                "promotion_reason": "user_declined_retraining",
                "summary_text": (
                    f"{active_result['model_name']} {active_result['target_column']} MAPE={trigger_mape:.2f}%, "
                    "사용자가 후보 비교 결과를 보고 현재 active 모델 유지를 선택했습니다."
                ),
            }
        )
        return active_result

    if not preview_state["approval_required"]:
        active_result.update(
            {
                "served_from": served_from,
                "retrained": False,
                "redeployed": False,
                "approval_required": False,
                "approved": True,
                "promotion_reason": "threshold_not_exceeded",
            }
        )
        return active_result

    active_promotion_result = preview_state["active_promotion_result"]
    previous_rmse = float(preview_state["previous_rmse"])
    previous_mape = float(preview_state["previous_mape"])
    candidate_bundle = preview_state["candidate_bundle"]
    candidate_result = preview_state["candidate_result"]
    candidate_rmse = float(preview_state["candidate_rmse"])
    candidate_mape = float(preview_state["candidate_mape"])
    redeployed = False
    display_result = active_result
    final_promotion_result = active_promotion_result
    should_promote = bool(preview_state["should_promote"])
    promotion_reason = preview_state["promotion_reason"]

    if should_promote:
        save_predictive_lstm_bundle(candidate_bundle)
        redeployed = True
        display_result = evaluate_predictive_lstm(
            dataset,
            candidate_bundle,
            source_name,
            plot_path=MELBOURNE_PREDICTOR_PLOT_PATH,
            plot_title=f"Redeployed Melbourne {_model_label(candidate_bundle['model_family'])} Forecast",
        )
        final_promotion_result = candidate_result

    serving_metrics = dict(display_result["metrics"]["forecast"])
    holdout_metrics = dict(final_promotion_result["metrics"]["forecast"])
    result = dict(display_result)

    result.update(
        {
            "served_from": served_from,
            "retrained": True,
            "redeployed": redeployed,
            "approval_required": False,
            "approved": True,
            "previous_rmse": float(previous_rmse),
            "previous_mape": float(previous_mape),
            "candidate_rmse": float(candidate_rmse),
            "candidate_mape": float(candidate_mape),
            "trigger_rmse": float(trigger_rmse),
            "trigger_mape": float(trigger_mape),
            "trigger_metrics": trigger_metrics,
            "serving_metrics": serving_metrics,
            "promotion_holdout_metrics": holdout_metrics,
            "candidate_model_name": candidate_bundle["model_name"],
            "candidate_model_family": candidate_bundle["model_family"],
            "candidate_feature_preset": candidate_bundle["feature_preset"],
            "candidate_window_size": int(candidate_bundle["window_size"]),
            "rmse_threshold": threshold_rmse,
            "promotion_reason": promotion_reason,
            "promotion_evaluation_scope": preview_state["promotion_evaluation_scope"],
            "promotion_holdout_rows": int(preview_state["promotion_holdout_rows"]),
            "search_space_size": int(preview_state["search_space_size"]),
            "search_summary": preview_state["search_summary"],
            "search_leaderboard": candidate_bundle.get("search_leaderboard", []),
            "summary_text": (
                f"{result['model_name']} {result['target_column']} serving MAPE={serving_metrics['mape']:.2f}%, "
                f"holdout MAPE={holdout_metrics['mape']:.2f}%, RMSE={holdout_metrics['rmse']:.3f}, "
                f"retrained=True, redeployed={redeployed}, "
                f"candidate_search={preview_state['search_space_size']}"
            ),
        }
    )
    return result


def serve_predictive_lstm(dataset: pd.DataFrame, source_name: str = "") -> dict:
    """기존 호출 방식과 호환되도록 승인 경로를 통해 서빙을 수행한다."""
    return approve_retraining(dataset, source_name, approved=True)


def get_model_shapes_png():
    """기존 다운로드 엔드포인트에서 쓰는 메인 예측 그래프 경로를 반환한다."""
    return MELBOURNE_PREDICTOR_PLOT_PATH


def get_stock_png():
    """stock 스타일 다운로드 엔드포인트가 기대하는 예측 그래프 경로를 반환한다."""
    return MELBOURNE_PREDICTOR_PLOT_PATH
