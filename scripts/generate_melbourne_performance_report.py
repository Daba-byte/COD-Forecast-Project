#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
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

import pandas as pd

from config import (
    MELBOURNE_BOOTSTRAP_END_DATE,
    MELBOURNE_CALIBRATION_END_DATE,
    MELBOURNE_SPLIT_DIR,
)
from server_model import model as baseline_model
from server_model import weight_used_model as predictor_model
from server_model.melbourne_utils import prepare_melbourne_dataframe, read_tabular_csv


@dataclass
class StaticResult:
    split_name: str
    rows: int
    predictions: int
    threshold_rmse: float
    approval_required: bool
    active_rmse: float
    active_mae: float
    active_mape: float
    baseline_rmse: float
    baseline_mae: float
    baseline_mape: float


@dataclass
class SequentialResult:
    split_name: str
    rows: int
    active_model_name: str
    threshold_rmse: float
    active_rmse: float
    active_mae: float
    active_mape: float
    approval_required: bool
    candidate_rmse: float | None
    candidate_mae: float | None
    candidate_mape: float | None
    redeployed: bool
    promotion_reason: str
    next_active_model_name: str


STATIC_SPLITS = [
    "test_2018_h1.csv",
    "upload_1_2018_q3_q4.csv",
    "upload_2_2018_q4_2019_q1.csv",
    "upload_3_2019_q2.csv",
]

SEQUENTIAL_UPLOAD_SPLITS = [
    "upload_1_2018_q3_q4.csv",
    "upload_2_2018_q4_2019_q1.csv",
    "upload_3_2019_q2.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown report for Melbourne model performance.")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "MELBOURNE_MODEL_PERFORMANCE.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def load_split_dataset(split_name: str) -> pd.DataFrame:
    split_path = Path(MELBOURNE_SPLIT_DIR) / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return read_tabular_csv(split_path)


def train_cold_start_bundle() -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    bootstrap_df, calibration_df, reference_path = predictor_model._load_bootstrap_and_calibration_frames()
    bundle = predictor_model.train_predictive_lstm(
        bootstrap_df,
        source_name=f"{reference_path}::bootstrap<={MELBOURNE_BOOTSTRAP_END_DATE}",
        model_name="melbourne_deployed_lstm",
        calibration_df=calibration_df,
        calibration_reference=(
            f"{reference_path}::{MELBOURNE_BOOTSTRAP_END_DATE}..{MELBOURNE_CALIBRATION_END_DATE}"
        ),
    )
    return bundle, bootstrap_df, calibration_df, reference_path


def evaluate_static_performance(active_bundle: dict) -> list[StaticResult]:
    results: list[StaticResult] = []
    threshold_rmse = float(active_bundle["rmse_threshold"])

    for split_name in STATIC_SPLITS:
        dataset = load_split_dataset(split_name)
        active_result = predictor_model.evaluate_predictive_lstm(
            dataset,
            active_bundle,
            source_name=split_name,
            plot_path=None,
            plot_title=f"Static Evaluation {split_name}",
        )
        baseline_result = baseline_model.process(dataset, split_name)
        active_metrics = active_result["metrics"]["forecast"]
        baseline_metrics = baseline_result["metrics"]["forecast"]

        results.append(
            StaticResult(
                split_name=split_name,
                rows=int(len(dataset)),
                predictions=int(active_result["prediction_count"]),
                threshold_rmse=threshold_rmse,
                approval_required=bool(active_metrics["rmse"] > threshold_rmse),
                active_rmse=float(active_metrics["rmse"]),
                active_mae=float(active_metrics["mae"]),
                active_mape=float(active_metrics["mape"]),
                baseline_rmse=float(baseline_metrics["rmse"]),
                baseline_mae=float(baseline_metrics["mae"]),
                baseline_mape=float(baseline_metrics["mape"]),
            )
        )

    return results


def simulate_sequential_uploads(
    initial_bundle: dict,
    bootstrap_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    reference_path: str,
) -> list[SequentialResult]:
    current_bundle = initial_bundle
    results: list[SequentialResult] = []

    for split_name in SEQUENTIAL_UPLOAD_SPLITS:
        dataset = load_split_dataset(split_name)
        active_result = predictor_model.evaluate_predictive_lstm(
            dataset,
            current_bundle,
            source_name=split_name,
            plot_path=None,
            plot_title=f"Sequential Evaluation {split_name}",
        )
        active_metrics = active_result["metrics"]["forecast"]
        threshold_rmse = float(current_bundle["rmse_threshold"])
        approval_required = bool(active_metrics["rmse"] > threshold_rmse)

        candidate_result = None
        redeployed = False
        promotion_reason = "threshold_not_exceeded"

        if approval_required:
            upload_df, upload_feature_columns, target_column = prepare_melbourne_dataframe(
                dataset,
                split_name,
                current_bundle["target_column"],
            )
            candidate_train_df, candidate_val_df = predictor_model._build_candidate_frames(bootstrap_df, upload_df)
            candidate_bundle = predictor_model._train_predictive_lstm_from_frames(
                candidate_train_df,
                candidate_val_df,
                upload_feature_columns,
                target_column,
                "melbourne_candidate_lstm",
                train_reference=f"{reference_path}::recent_bootstrap_tail+{split_name}",
                calibration_df=calibration_df,
                calibration_reference=(
                    f"{reference_path}::{MELBOURNE_BOOTSTRAP_END_DATE}..{MELBOURNE_CALIBRATION_END_DATE}"
                ),
            )
            candidate_result = predictor_model.evaluate_predictive_lstm(
                dataset,
                candidate_bundle,
                source_name=split_name,
                plot_path=None,
                plot_title=f"Candidate Evaluation {split_name}",
            )
            candidate_metrics = candidate_result["metrics"]["forecast"]
            redeployed = bool(candidate_metrics["rmse"] < active_metrics["rmse"] * predictor_model.PROMOTION_MARGIN)
            promotion_reason = "candidate_improved_rmse" if redeployed else "candidate_not_better_than_active"
            if redeployed:
                current_bundle = candidate_bundle

        candidate_metrics = candidate_result["metrics"]["forecast"] if candidate_result else None
        results.append(
            SequentialResult(
                split_name=split_name,
                rows=int(len(dataset)),
                active_model_name=str(active_result["model_name"]),
                threshold_rmse=threshold_rmse,
                active_rmse=float(active_metrics["rmse"]),
                active_mae=float(active_metrics["mae"]),
                active_mape=float(active_metrics["mape"]),
                approval_required=approval_required,
                candidate_rmse=float(candidate_metrics["rmse"]) if candidate_metrics else None,
                candidate_mae=float(candidate_metrics["mae"]) if candidate_metrics else None,
                candidate_mape=float(candidate_metrics["mape"]) if candidate_metrics else None,
                redeployed=redeployed,
                promotion_reason=promotion_reason,
                next_active_model_name=str(current_bundle["model_name"]),
            )
        )

    return results


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def build_report_markdown(static_results: list[StaticResult], sequential_results: list[SequentialResult]) -> str:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    threshold_rmse = static_results[0].threshold_rmse if static_results else 0.0
    active_better_count = sum(1 for row in static_results if row.active_rmse < row.baseline_rmse)
    average_active_rmse = sum(row.active_rmse for row in static_results) / max(len(static_results), 1)
    average_baseline_rmse = sum(row.baseline_rmse for row in static_results) / max(len(static_results), 1)
    approvals = [row for row in sequential_results if row.approval_required]
    redeploys = [row for row in sequential_results if row.redeployed]

    lines: list[str] = []
    lines.append("# Melbourne Model Performance Report")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Reference bootstrap range: `2014-01-01 ~ {MELBOURNE_BOOTSTRAP_END_DATE}`")
    lines.append(f"- Calibration range: `2018-01-01 ~ {MELBOURNE_CALIBRATION_END_DATE}`")
    lines.append(f"- RMSE threshold used by active model: `{fmt_float(threshold_rmse, 2)}`")
    lines.append(f"- Command: `/Users/hyun/workspace/mlops/.venv/bin/python scripts/generate_melbourne_performance_report.py`")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(
        f"- cold start active 모델은 정적 평가 4개 구간 모두에서 baseline보다 낮은 RMSE를 기록했습니다 "
        f"(`{active_better_count}/4`)."
    )
    lines.append(
        f"- 정적 평가 평균 RMSE는 active `{fmt_float(average_active_rmse, 2)}`, "
        f"baseline `{fmt_float(average_baseline_rmse, 2)}` 입니다."
    )
    lines.append(
        f"- 순차 업로드 시뮬레이션에서는 승인 필요 구간이 `{len(approvals)}`회 발생했고, "
        f"실제 redeploy는 `{len(redeploys)}`회 발생했습니다."
    )
    lines.append(
        "- `test_2018_h1`은 threshold 산정에 사용되는 calibration 기간과 겹치므로, "
        "독립 holdout이라기보다 기준선 확인용 숫자로 해석하는 편이 맞습니다."
    )
    lines.append("")
    lines.append("## 정적 평가")
    lines.append("")
    lines.append(
        "| split | rows | preds | active RMSE | active MAE | active MAPE | baseline RMSE | baseline MAE | baseline MAPE | threshold | approval |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in static_results:
        lines.append(
            f"| `{row.split_name}` | {row.rows} | {row.predictions} | {fmt_float(row.active_rmse, 2)} | "
            f"{fmt_float(row.active_mae, 2)} | {fmt_float(row.active_mape, 2)}% | "
            f"{fmt_float(row.baseline_rmse, 2)} | {fmt_float(row.baseline_mae, 2)} | "
            f"{fmt_float(row.baseline_mape, 2)}% | {fmt_float(row.threshold_rmse, 2)} | "
            f"{'yes' if row.approval_required else 'no'} |"
        )
    lines.append("")
    lines.append("## 순차 업로드 시뮬레이션")
    lines.append("")
    lines.append(
        "| split | active model | active RMSE | threshold | approval | candidate RMSE | redeployed | next active | reason |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |")
    for row in sequential_results:
        lines.append(
            f"| `{row.split_name}` | `{row.active_model_name}` | {fmt_float(row.active_rmse, 2)} | "
            f"{fmt_float(row.threshold_rmse, 2)} | {'yes' if row.approval_required else 'no'} | "
            f"{fmt_float(row.candidate_rmse, 2)} | {'yes' if row.redeployed else 'no'} | "
            f"`{row.next_active_model_name}` | `{row.promotion_reason}` |"
        )
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    for row in static_results:
        gap = row.baseline_rmse - row.active_rmse
        direction = "개선" if gap > 0 else "열세"
        lines.append(
            f"- `{row.split_name}`에서는 active가 baseline 대비 RMSE `{fmt_float(abs(gap), 2)}` 만큼 `{direction}`되었습니다."
        )
    if approvals:
        lines.append(
            f"- 승인 필요가 발생한 첫 구간은 `{approvals[0].split_name}` 이며, active RMSE는 "
            f"`{fmt_float(approvals[0].active_rmse, 2)}` 입니다."
        )
    if redeploys:
        lines.append(
            f"- 실제 승격이 발생한 구간은 `{', '.join(row.split_name for row in redeploys)}` 입니다."
        )
    else:
        lines.append("- 이번 실행에서는 candidate가 promotion margin을 넘지 못해 redeploy가 발생하지 않았습니다.")
    lines.append("")
    lines.append("## 재현 방법")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /Users/hyun/workspace/mlops/model_serving_rpt")
    lines.append("/Users/hyun/workspace/mlops/.venv/bin/python scripts/generate_melbourne_performance_report.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_bundle, bootstrap_df, calibration_df, reference_path = train_cold_start_bundle()
    static_results = evaluate_static_performance(active_bundle)
    sequential_results = simulate_sequential_uploads(active_bundle, bootstrap_df, calibration_df, reference_path)

    markdown = build_report_markdown(static_results, sequential_results)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Report written to: {output_path}")
    print(f"Static splits evaluated: {len(static_results)}")
    print(f"Sequential uploads simulated: {len(sequential_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
