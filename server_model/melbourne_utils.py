import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGET_COLUMN = os.getenv("MELBOURNE_TARGET_COLUMN", "Chemical Oxygen Demand")
DATE_COLUMNS = ["Year", "Month", "Day"]
IGNORED_COLUMNS = {"", "Unnamed: 0"}
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week",
    "week_of_year",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]


def read_tabular_csv(file_path: str | Path) -> pd.DataFrame:
    """프로젝트 전처리를 적용하지 않은 원본 CSV를 읽는다."""
    return pd.read_csv(file_path)


def prepare_melbourne_dataframe(
    dataset: pd.DataFrame,
    source_name: str = "",
    target_column: str = DEFAULT_TARGET_COLUMN,
    feature_preset: str = "full",
) -> tuple[pd.DataFrame, list[str], str]:
    """Melbourne 데이터를 datetime 인덱스 기반의 모델링용 프레임으로 정리한다."""
    df = dataset.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)
    df.columns = [str(column).strip() for column in df.columns]
    ignored = [column for column in df.columns if column in IGNORED_COLUMNS or str(column).startswith("Unnamed:")]
    if ignored:
        df = df.drop(columns=ignored)

    missing_date_columns = [column for column in DATE_COLUMNS if column not in df.columns]
    if missing_date_columns:
        raise ValueError(f"Uploaded CSV must include date columns: {missing_date_columns}")
    if target_column not in df.columns:
        raise ValueError(f"Uploaded CSV must include target column '{target_column}'.")

    numeric_columns = [column for column in df.columns if column not in DATE_COLUMNS]
    for column in numeric_columns + DATE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["datetime"] = pd.to_datetime(
        {
            "year": df["Year"].astype("Int64"),
            "month": df["Month"].astype("Int64"),
            "day": df["Day"].astype("Int64"),
        },
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    if df.empty:
        raise ValueError(f"No usable rows were found in '{source_name or 'uploaded dataset'}'.")

    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(float)
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)

    model_feature_columns = list(dict.fromkeys(numeric_columns + CALENDAR_FEATURE_COLUMNS))
    model_feature_columns = select_feature_columns(model_feature_columns, target_column, feature_preset)

    df = df.set_index("datetime")
    return df, model_feature_columns, target_column


def build_feature_presets(full_features: list[str], target_column: str) -> dict[str, list[str]]:
    """COD 예측 파이프라인에서 쓰는 이름별 feature 조합을 반환한다."""
    process_features = [
        target_column,
        "Average Outflow",
        "Average Inflow",
        "Ammonia",
        "Biological Oxygen Demand",
        "Chemical Oxygen Demand",
        "Total Nitrogen",
    ]
    process_features_no_bod = [
        target_column,
        "Average Outflow",
        "Average Inflow",
        "Ammonia",
        "Chemical Oxygen Demand",
        "Total Nitrogen",
    ]
    weather_features = [
        target_column,
        "Average Temperature",
        "Maximum temperature",
        "Minimum temperature",
        "Atmospheric pressure",
        "Average humidity",
        "Total rainfall",
        "Average visibility",
        "Average wind speed",
        "Maximum wind speed",
    ]
    hybrid_features = [
        target_column,
        "Average Outflow",
        "Average Inflow",
        "Chemical Oxygen Demand",
        "Total Nitrogen",
        "Average Temperature",
        "Average humidity",
        "Total rainfall",
    ]
    notebook_cod_features = [
        "Ammonia",
        "Total Nitrogen",
        "Average Inflow",
        "Average Outflow",
        "Average Temperature",
        "Maximum temperature",
        "Minimum temperature",
        "Total rainfall",
    ]

    def _ordered_subset(candidates: list[str]) -> list[str]:
        return [feature for feature in full_features if feature in candidates]

    def _candidate_ordered_subset(candidates: list[str]) -> list[str]:
        return [feature for feature in candidates if feature in full_features]

    presets = {
        "full": list(full_features),
        "target_calendar": _ordered_subset([target_column, *CALENDAR_FEATURE_COLUMNS]),
        "process_core": _ordered_subset([*process_features, *CALENDAR_FEATURE_COLUMNS]),
        "process_core_no_bod": _ordered_subset([*process_features_no_bod, *CALENDAR_FEATURE_COLUMNS]),
        "notebook_cod": _candidate_ordered_subset(notebook_cod_features),
        "weather_target": _ordered_subset([*weather_features, *CALENDAR_FEATURE_COLUMNS]),
        "compact_hybrid": _ordered_subset([*hybrid_features, *CALENDAR_FEATURE_COLUMNS]),
        "full_no_target_history": [feature for feature in full_features if feature != target_column],
    }
    return {name: columns for name, columns in presets.items() if columns}


def select_feature_columns(
    full_features: list[str],
    target_column: str,
    feature_preset: str = "full",
) -> list[str]:
    """요청한 feature preset을 실제 입력 컬럼 목록으로 변환한다."""
    presets = build_feature_presets(full_features, target_column)
    normalized_preset = (feature_preset or "full").strip().lower()
    if normalized_preset not in presets:
        raise ValueError(
            f"Unsupported Melbourne feature preset '{feature_preset}'. "
            f"Expected one of {sorted(presets)}."
        )
    return presets[normalized_preset]


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시간 순서가 유지된 데이터프레임을 train/validation/test로 나눈다."""
    total = len(df)
    if total < 3:
        raise ValueError("At least three rows are required to split the dataframe.")

    train_end = max(int(total * train_ratio), 1)
    val_end = max(train_end + int(total * val_ratio), train_end + 1)
    val_end = min(val_end, total - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if val_df.empty:
        val_df = train_df.iloc[-max(1, len(train_df) // 5) :].copy()
    if test_df.empty:
        test_df = df.iloc[-max(1, len(df) // 5) :].copy()

    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[StandardScaler, StandardScaler]:
    """입력 feature와 target에 사용할 스케일러를 각각 학습한다."""
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[[target_column]])
    return feature_scaler, target_scaler


def transform_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """feature 프레임을 모델 입력용 float32 배열로 스케일링한다."""
    return scaler.transform(df[feature_columns]).astype(np.float32)


def transform_target_series(
    df: pd.DataFrame,
    target_column: str,
    scaler: StandardScaler,
) -> np.ndarray:
    """target 시리즈를 1차원 학습 벡터로 스케일링한다."""
    return scaler.transform(df[[target_column]]).astype(np.float32).ravel()


def inverse_target(
    values: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """스케일된 예측값을 원래 target 스케일로 되돌린다."""
    array = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(array).ravel()


def build_forecast_windows(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    timestamps: pd.Index,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """시계열 모델에 사용할 슬라이딩 윈도우와 target 시점을 만든다."""
    windows: list[np.ndarray] = []
    targets: list[float] = []
    target_timestamps: list[pd.Timestamp] = []

    for target_index in range(window_size, len(feature_values), stride):
        start_index = target_index - window_size
        windows.append(feature_values[start_index:target_index])
        targets.append(float(target_values[target_index]))
        target_timestamps.append(timestamps[target_index])

    if not windows:
        raise ValueError("Not enough rows to build forecast windows.")

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DatetimeIndex(target_timestamps),
    )


def compute_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """예측 결과의 RMSE, MAE, MAPE를 계산한다."""
    residual = np.asarray(actual, dtype=np.float32) - np.asarray(predicted, dtype=np.float32)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))

    denominator = np.where(np.abs(actual) < 1e-6, 1.0, np.abs(actual))
    mape = float(np.mean(np.abs(residual) / denominator) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def save_forecast_plot(
    timestamps: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str | Path,
    title: str,
) -> str:
    """실제값과 예측값 비교 그래프를 저장하고 경로를 반환한다."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, actual, label="actual", linewidth=2)
    plt.plot(timestamps, predicted, label="predicted", linewidth=1.8)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("target")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()
    return str(output)


def combine_reference_and_upload(reference_df: pd.DataFrame, upload_df: pd.DataFrame) -> pd.DataFrame:
    """기준 데이터와 업로드 데이터를 합치되 날짜별 최신 행만 남긴다."""
    combined = pd.concat([reference_df, upload_df], axis=0, ignore_index=True)
    if all(column in combined.columns for column in DATE_COLUMNS):
        combined = combined.drop_duplicates(subset=DATE_COLUMNS, keep="last")
        combined = combined.sort_values(DATE_COLUMNS)
    return combined.reset_index(drop=True)


def slice_frame_by_date(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """datetime 인덱스를 기준으로 시작일과 종료일 사이 구간만 잘라낸다."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("slice_frame_by_date expects a dataframe indexed by datetime.")

    result = df
    if start is not None:
        result = result[result.index >= Timestamp(start)]
    if end is not None:
        result = result[result.index <= Timestamp(end)]
    return result.copy()
