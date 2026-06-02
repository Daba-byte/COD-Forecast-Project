#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv1D, Dense, GRU, GlobalAveragePooling1D, Input, LSTM
from tensorflow.keras.models import Model

from config import MELBOURNE_BOOTSTRAP_END_DATE, MELBOURNE_SPLIT_DIR, MELBOURNE_TARGET_COLUMN
from server_model.melbourne_utils import (
    build_feature_presets,
    build_forecast_windows,
    compute_forecast_metrics,
    fit_scalers,
    inverse_target,
    prepare_melbourne_dataframe,
    read_tabular_csv,
    slice_frame_by_date,
    split_dataframe,
    transform_feature_frame,
    transform_target_series,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")


SEED = 42
STRIDE = 1
EPOCHS = 12
BATCH_SIZE = 32
RANKING_SPLITS = [
    "upload_1_2018_q3_q4.csv",
    "upload_2_2018_q4_2019_q1.csv",
    "upload_3_2019_q2.csv",
]
ALL_EVAL_SPLITS = ["test_2018_h1.csv", *RANKING_SPLITS]
WINDOW_SIZES = (7, 14, 21)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    feature_preset: str
    window_size: int


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    mean_upload_mape: float
    mean_upload_rmse: float
    calibration_mape: float
    calibration_rmse: float
    validation_mape: float
    runtime_seconds: float
    split_metrics: dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the best Melbourne forecasting model across model/input presets.")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "MELBOURNE_MODEL_SEARCH.md"),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--csv-output",
        default=str(ROOT_DIR / "model_search_results.csv"),
        help="CSV output path for all experiment rows.",
    )
    return parser.parse_args()


def _make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_scaler,
    target_scaler,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_values = transform_feature_frame(df, feature_columns, feature_scaler)
    target_values = transform_target_series(df, target_column, target_scaler)
    x_values, y_scaled, _ = build_forecast_windows(
        feature_values,
        target_values,
        df.index,
        window_size=window_size,
        stride=STRIDE,
    )
    y_actual = inverse_target(y_scaled, target_scaler)
    return x_values, y_scaled, y_actual
def build_deep_model(model_name: str, window_size: int, feature_count: int) -> Model:
    inputs = Input(shape=(window_size, feature_count))

    if model_name == "gru":
        hidden = GRU(64, return_sequences=True)(inputs)
        hidden = GRU(32)(hidden)
    elif model_name == "lstm":
        hidden = LSTM(64, return_sequences=True)(inputs)
        hidden = LSTM(32)(hidden)
    elif model_name == "conv1d":
        hidden = Conv1D(64, kernel_size=3, activation="relu", padding="causal")(inputs)
        hidden = Conv1D(32, kernel_size=3, activation="relu", padding="causal")(hidden)
        hidden = GlobalAveragePooling1D()(hidden)
    else:
        raise ValueError(f"Unsupported deep model: {model_name}")

    outputs = Dense(1)(hidden)
    model = Model(inputs, outputs, name=f"melbourne_search_{model_name}")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_sklearn_estimator(model_name: str):
    if model_name == "hist_gb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=8,
            random_state=SEED,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        )
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported sklearn model: {model_name}")


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    metrics = compute_forecast_metrics(actual, predicted)
    return {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "mape": float(metrics["mape"]),
    }


def run_experiment(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_frames: dict[str, pd.DataFrame],
    target_column: str,
    feature_columns: list[str],
) -> ExperimentResult:
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    feature_scaler, target_scaler = fit_scalers(train_df, feature_columns, target_column)
    x_train, y_train_scaled, _ = _make_windows(train_df, feature_columns, target_column, feature_scaler, target_scaler, config.window_size)
    x_val, y_val_scaled, y_val_actual = _make_windows(val_df, feature_columns, target_column, feature_scaler, target_scaler, config.window_size)

    started_at = time.perf_counter()
    split_metrics: dict[str, dict[str, float]] = {}

    if config.model_name in {"gru", "lstm", "conv1d"}:
        tf.keras.backend.clear_session()
        model = build_deep_model(config.model_name, config.window_size, len(feature_columns))
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
        val_pred = inverse_target(model.predict(x_val, verbose=0).reshape(-1), target_scaler)
        validation_metrics = evaluate_predictions(y_val_actual, val_pred)

        for split_name, split_df in eval_frames.items():
            x_eval, _, y_eval_actual = _make_windows(
                split_df,
                feature_columns,
                target_column,
                feature_scaler,
                target_scaler,
                config.window_size,
            )
            pred = inverse_target(model.predict(x_eval, verbose=0).reshape(-1), target_scaler)
            split_metrics[split_name] = evaluate_predictions(y_eval_actual, pred)
    else:
        estimator = build_sklearn_estimator(config.model_name)
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_val_flat = x_val.reshape((x_val.shape[0], -1))
        estimator.fit(x_train_flat, y_train_scaled)
        val_pred = inverse_target(np.asarray(estimator.predict(x_val_flat), dtype=np.float32), target_scaler)
        validation_metrics = evaluate_predictions(y_val_actual, val_pred)

        for split_name, split_df in eval_frames.items():
            x_eval, _, y_eval_actual = _make_windows(
                split_df,
                feature_columns,
                target_column,
                feature_scaler,
                target_scaler,
                config.window_size,
            )
            pred = inverse_target(
                np.asarray(estimator.predict(x_eval.reshape((x_eval.shape[0], -1))), dtype=np.float32),
                target_scaler,
            )
            split_metrics[split_name] = evaluate_predictions(y_eval_actual, pred)

    runtime_seconds = time.perf_counter() - started_at
    ranking_rows = [split_metrics[name] for name in RANKING_SPLITS]
    mean_upload_mape = float(sum(row["mape"] for row in ranking_rows) / len(ranking_rows))
    mean_upload_rmse = float(sum(row["rmse"] for row in ranking_rows) / len(ranking_rows))
    calibration_metrics = split_metrics["test_2018_h1.csv"]

    return ExperimentResult(
        config=config,
        mean_upload_mape=mean_upload_mape,
        mean_upload_rmse=mean_upload_rmse,
        calibration_mape=float(calibration_metrics["mape"]),
        calibration_rmse=float(calibration_metrics["rmse"]),
        validation_mape=float(validation_metrics["mape"]),
        runtime_seconds=float(runtime_seconds),
        split_metrics=split_metrics,
    )


def build_experiment_grid(feature_presets: dict[str, list[str]]) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for model_name in ("gru", "lstm", "conv1d", "hist_gb", "extra_trees", "ridge"):
        for feature_preset in feature_presets:
            for window_size in WINDOW_SIZES:
                configs.append(
                    ExperimentConfig(
                        model_name=model_name,
                        feature_preset=feature_preset,
                        window_size=window_size,
                    )
                )
    return configs


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def result_to_row(result: ExperimentResult) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": result.config.model_name,
        "feature_preset": result.config.feature_preset,
        "window_size": result.config.window_size,
        "mean_upload_mape": round(result.mean_upload_mape, 4),
        "mean_upload_rmse": round(result.mean_upload_rmse, 4),
        "calibration_mape": round(result.calibration_mape, 4),
        "calibration_rmse": round(result.calibration_rmse, 4),
        "validation_mape": round(result.validation_mape, 4),
        "runtime_seconds": round(result.runtime_seconds, 3),
    }
    for split_name, metrics in result.split_metrics.items():
        prefix = split_name.replace(".csv", "").replace("-", "_")
        row[f"{prefix}_mape"] = round(metrics["mape"], 4)
        row[f"{prefix}_rmse"] = round(metrics["rmse"], 4)
        row[f"{prefix}_mae"] = round(metrics["mae"], 4)
    return row


def summarize_group_best(results: list[ExperimentResult], key: str) -> list[ExperimentResult]:
    grouped: dict[str, list[ExperimentResult]] = {}
    for result in results:
        group_key = getattr(result.config, key)
        grouped.setdefault(group_key, []).append(result)
    return [
        sorted(group_results, key=lambda item: (item.mean_upload_mape, item.mean_upload_rmse, item.calibration_mape))[0]
        for group_results in grouped.values()
    ]


def build_todo_list(results: list[ExperimentResult]) -> list[str]:
    best = results[0]
    feature_best = sorted(summarize_group_best(results, "feature_preset"), key=lambda item: item.mean_upload_mape)
    model_best = sorted(summarize_group_best(results, "model_name"), key=lambda item: item.mean_upload_mape)

    todos = [
        (
            f"현재 서빙 파이프라인에 `{best.config.model_name}` + `{best.config.feature_preset}` + "
            f"`window={best.config.window_size}` 조합을 반영하고, 실제 업로드 API 응답이 같은 개선을 보이는지 다시 검증하기"
        ),
        (
            f"상위 feature preset `{feature_best[0].config.feature_preset}` 주변으로 lag/rolling 평균(7일, 14일, 28일) 입력을 추가해 "
            "MAPE를 한 번 더 낮출 수 있는지 확인하기"
        ),
        (
            f"상위 모델군 `{model_best[0].config.model_name}` 기준으로 hidden units 또는 tree depth를 더 촘촘하게 튜닝해 "
            "현재 top score를 다시 미세 탐색하기"
        ),
        "2019-Q2 이후 완전 분리된 holdout 파일을 하나 더 만들어, calibration 겹침 없는 진짜 운영 검증셋으로 재평가하기",
    ]
    return todos


def build_markdown(results: list[ExperimentResult], total_experiments: int) -> str:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    best = results[0]
    best_by_model = sorted(summarize_group_best(results, "model_name"), key=lambda item: item.mean_upload_mape)
    best_by_feature = sorted(summarize_group_best(results, "feature_preset"), key=lambda item: item.mean_upload_mape)
    todos = build_todo_list(results)

    lines: list[str] = []
    lines.append("# Melbourne Model Search")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Bootstrap train end date: `{MELBOURNE_BOOTSTRAP_END_DATE}`")
    lines.append(f"- Ranking objective: `lowest average MAPE across {', '.join(RANKING_SPLITS)}`")
    lines.append(f"- Experiments run: `{total_experiments}`")
    lines.append("")
    lines.append("## Best Result")
    lines.append("")
    lines.append(
        f"- Best model: `{best.config.model_name}` with feature preset `{best.config.feature_preset}` and "
        f"`window={best.config.window_size}`"
    )
    lines.append(f"- Mean upload MAPE: `{fmt_float(best.mean_upload_mape)}%`")
    lines.append(f"- Mean upload RMSE: `{fmt_float(best.mean_upload_rmse)}`")
    lines.append(f"- Calibration MAPE (`test_2018_h1.csv`): `{fmt_float(best.calibration_mape)}%`")
    lines.append(f"- Validation MAPE (bootstrap split): `{fmt_float(best.validation_mape)}%`")
    lines.append("")
    lines.append("## Top 10 Experiments")
    lines.append("")
    lines.append("| rank | model | feature preset | window | avg upload MAPE | avg upload RMSE | calibration MAPE | validation MAPE | runtime(s) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, result in enumerate(results[:10], start=1):
        lines.append(
            f"| {rank} | `{result.config.model_name}` | `{result.config.feature_preset}` | {result.config.window_size} | "
            f"{fmt_float(result.mean_upload_mape)}% | {fmt_float(result.mean_upload_rmse)} | "
            f"{fmt_float(result.calibration_mape)}% | {fmt_float(result.validation_mape)}% | "
            f"{fmt_float(result.runtime_seconds, 2)} |"
        )
    lines.append("")
    lines.append("## Best By Model Family")
    lines.append("")
    lines.append("| model | feature preset | window | avg upload MAPE | calibration MAPE |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for result in best_by_model:
        lines.append(
            f"| `{result.config.model_name}` | `{result.config.feature_preset}` | {result.config.window_size} | "
            f"{fmt_float(result.mean_upload_mape)}% | {fmt_float(result.calibration_mape)}% |"
        )
    lines.append("")
    lines.append("## Best By Feature Preset")
    lines.append("")
    lines.append("| feature preset | model | window | avg upload MAPE | calibration MAPE |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for result in best_by_feature:
        lines.append(
            f"| `{result.config.feature_preset}` | `{result.config.model_name}` | {result.config.window_size} | "
            f"{fmt_float(result.mean_upload_mape)}% | {fmt_float(result.calibration_mape)}% |"
        )
    lines.append("")
    lines.append("## Best Split Detail")
    lines.append("")
    lines.append("| split | MAPE | RMSE | MAE |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split_name, metrics in best.split_metrics.items():
        lines.append(
            f"| `{split_name}` | {fmt_float(metrics['mape'])}% | {fmt_float(metrics['rmse'])} | {fmt_float(metrics['mae'])} |"
        )
    lines.append("")
    lines.append("## TODO")
    lines.append("")
    for todo in todos:
        lines.append(f"- {todo}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /Users/hyun/workspace/mlops/model_serving_rpt")
    lines.append("/Users/hyun/workspace/mlops/.venv/bin/python scripts/search_melbourne_best_model.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_dataset = read_tabular_csv(ROOT_DIR / "data2" / "Data-Melbourne_F_fixed.csv")
    reference_df, full_features, target_column = prepare_melbourne_dataframe(
        reference_dataset,
        "Data-Melbourne_F_fixed.csv",
        MELBOURNE_TARGET_COLUMN,
    )
    bootstrap_df = slice_frame_by_date(reference_df, end=MELBOURNE_BOOTSTRAP_END_DATE)
    train_df, val_df, _ = split_dataframe(bootstrap_df)

    eval_frames: dict[str, pd.DataFrame] = {}
    for split_name in ALL_EVAL_SPLITS:
        dataset = read_tabular_csv(Path(MELBOURNE_SPLIT_DIR) / split_name)
        split_df, _, _ = prepare_melbourne_dataframe(dataset, split_name, target_column)
        eval_frames[split_name] = split_df

    feature_presets = build_feature_presets(full_features, target_column)
    experiment_grid = build_experiment_grid(feature_presets)
    results: list[ExperimentResult] = []

    print(f"Running {len(experiment_grid)} experiments...", flush=True)
    for index, config in enumerate(experiment_grid, start=1):
        feature_columns = feature_presets[config.feature_preset]
        result = run_experiment(
            config,
            train_df,
            val_df,
            eval_frames,
            target_column,
            feature_columns,
        )
        results.append(result)
        print(
            f"[{index:02d}/{len(experiment_grid)}] "
            f"{config.model_name:<11} preset={config.feature_preset:<20} window={config.window_size:<2} "
            f"avg_upload_mape={result.mean_upload_mape:.2f}% calibration_mape={result.calibration_mape:.2f}%"
            ,
            flush=True,
        )

    results.sort(key=lambda item: (item.mean_upload_mape, item.mean_upload_rmse, item.calibration_mape, item.validation_mape))

    rows = [result_to_row(result) for result in results]
    pd.DataFrame(rows).to_csv(csv_output_path, index=False)
    output_path.write_text(build_markdown(results, len(experiment_grid)), encoding="utf-8")

    best = results[0]
    print("", flush=True)
    print(
        "Best:",
        best.config.model_name,
        best.config.feature_preset,
        best.config.window_size,
        f"avg_upload_mape={best.mean_upload_mape:.2f}%",
        flush=True,
    )
    print(f"Markdown report written to: {output_path}", flush=True)
    print(f"CSV report written to: {csv_output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
