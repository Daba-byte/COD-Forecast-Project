import json
import os


LLM_MODEL = os.getenv("MELBOURNE_LLM_MODEL", "gpt-4o-mini")


def _build_fallback_metrics(metrics: dict, threshold_rmse: float) -> dict:
    """LLM 보고서를 쓰지 못할 때 사용할 정규화된 지표 블록을 만든다."""
    rmse = float(metrics["rmse"])
    mae = float(metrics["mae"])
    mape = float(metrics["mape"])
    exceed_ratio = rmse / threshold_rmse if threshold_rmse else 0.0
    exceed_percent = max((exceed_ratio - 1.0) * 100.0, 0.0)
    return {
        "current_rmse": round(rmse, 2),
        "threshold_rmse": round(float(threshold_rmse), 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "exceed_ratio": round(exceed_ratio, 2),
        "exceed_percent": round(exceed_percent, 2),
    }


def _fallback_report(
    source_name: str,
    active_result: dict,
    threshold_rmse: float,
    reference_range: str,
    calibration_range: str,
) -> dict:
    """외부 LLM 호출 없이도 항상 같은 형식의 승인 보고서를 만든다."""
    metrics = active_result["metrics"]["forecast"]
    metric_payload = _build_fallback_metrics(metrics, threshold_rmse)
    exceeds_ratio = metric_payload["exceed_ratio"]
    severity = "high" if exceeds_ratio >= 1.2 else "medium"
    recommendation_title = (
        "재학습 승인 권장"
        if metrics["rmse"] > threshold_rmse
        else "현재 모델 유지 권장"
    )
    recommendation_body = (
        "현재 업로드 구간은 threshold를 초과했고, 최근 운영 패턴 변화 가능성도 보여 여러 모델/입력 조합을 다시 탐색한 뒤 가장 좋은 candidate를 검증하는 편이 안전합니다."
        if metrics["rmse"] > threshold_rmse
        else "현재 active 모델이 임계 범위 안에 있으므로 우선은 재학습 없이 모니터링을 유지하는 편이 적절합니다."
    )
    findings = [
        f"현재 active 모델 학습 구간은 {reference_range} 입니다.",
        f"성능 기준선(calibration) 구간은 {calibration_range} 입니다.",
        f"업로드 평가 결과는 MAPE {metric_payload['mape']:.2f}%, RMSE {metric_payload['current_rmse']:.2f}, MAE {metric_payload['mae']:.2f} 입니다.",
        f"현재 RMSE는 기준값 {metric_payload['threshold_rmse']:.2f} 대비 {metric_payload['exceed_ratio']:.2f}배 수준입니다.",
        "최근 운영 데이터 분포가 과거 학습 구간과 달라졌을 가능성이 있습니다.",
    ]
    next_steps = (
        [
            "여러 모델/입력/window 조합을 재탐색해 best candidate를 선정합니다.",
            "best candidate RMSE가 기존 active보다 개선되면 자동 승격합니다.",
            "승격되지 않더라도 현재 모델은 그대로 유지되므로 서빙 안정성은 보존됩니다.",
        ]
        if metrics["rmse"] > threshold_rmse
        else [
            "현재 active 모델을 유지한 채 다음 업로드 구간을 계속 관찰합니다.",
            "추가 업로드에서 threshold를 다시 초과하는지 확인합니다.",
        ]
    )
    return {
        "title": f"{source_name} 성능 저하 보고서",
        "severity": severity,
        "status_label": "재학습 검토 필요" if metrics["rmse"] > threshold_rmse else "정상 범위",
        "target_column": active_result["target_column"],
        "summary": (
            f"업로드 데이터에서 {active_result['target_column']} 예측 오차가 기준선보다 커졌습니다. "
            f"현재 MAPE는 {metric_payload['mape']:.2f}%이며, 승인 판단에 사용하는 RMSE는 {metric_payload['current_rmse']:.2f} / threshold {metric_payload['threshold_rmse']:.2f} 입니다."
        ),
        "overview": (
            "지표상으로는 급격한 붕괴보다는 '기준선을 살짝 넘어선 성능 저하'에 가깝습니다. "
            "하지만 최근 구간 적응 여부를 확인하려면 candidate 재학습을 한 번 수행해보는 것이 합리적입니다."
            if metrics["rmse"] > threshold_rmse
            else "현재 구간은 모니터링 범위 안에 있으므로 즉시 재학습보다는 추세 관찰이 우선입니다."
        ),
        "metrics": metric_payload,
        "findings": findings,
        "details": findings,
        "impact": (
            "현재 상태를 그대로 유지하면 최근 운영 구간에서 예측 정확도가 떨어진 채 서비스될 수 있습니다. "
            "다만 재학습은 candidate와 active를 비교한 뒤에만 승격되므로, 승인 자체가 곧바로 모델 교체를 의미하지는 않습니다."
        ),
        "recommendation_title": recommendation_title,
        "recommendation_body": recommendation_body,
        "next_steps": next_steps,
        "recommended_action": "approve_retraining" if metrics["rmse"] > threshold_rmse else "keep_active_model",
        "generation_mode": "fallback",
    }


def _extract_json_object(text: str) -> dict:
    """LLM 응답 문자열에서 첫 번째 유효한 JSON 객체를 추출한다."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _normalize_report(candidate: dict, fallback: dict) -> dict:
    """LLM 보고서를 프런트가 기대하는 형식으로 검증하고 보정한다."""
    report = dict(fallback)
    report.update({k: v for k, v in candidate.items() if v is not None})

    if report.get("severity") not in {"low", "medium", "high"}:
        report["severity"] = fallback["severity"]

    if report.get("recommended_action") not in {"approve_retraining", "keep_active_model"}:
        report["recommended_action"] = fallback["recommended_action"]

    for list_key in ("details", "findings", "next_steps"):
        value = report.get(list_key)
        if not isinstance(value, list) or not value:
            report[list_key] = fallback[list_key]
        else:
            report[list_key] = [str(item) for item in value[:8]]

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        report["metrics"] = fallback["metrics"]
    else:
        normalized_metrics = dict(fallback["metrics"])
        for key in normalized_metrics:
            value = metrics.get(key, normalized_metrics[key])
            try:
                normalized_metrics[key] = round(float(value), 2)
            except (TypeError, ValueError):
                pass
        report["metrics"] = normalized_metrics

    report["title"] = str(report.get("title") or fallback["title"])
    report["summary"] = str(report.get("summary") or fallback["summary"])
    report["overview"] = str(report.get("overview") or fallback["overview"])
    report["target_column"] = str(report.get("target_column") or fallback["target_column"])
    report["status_label"] = str(report.get("status_label") or fallback["status_label"])
    report["impact"] = str(report.get("impact") or fallback["impact"])
    report["recommendation_title"] = str(report.get("recommendation_title") or fallback["recommendation_title"])
    report["recommendation_body"] = str(report.get("recommendation_body") or fallback["recommendation_body"])
    return report


def generate_approval_report(
    source_name: str,
    active_result: dict,
    threshold_rmse: float,
    reference_range: str,
    calibration_range: str,
) -> dict:
    """재학습 승인 보고서를 만들고, LLM을 못 쓰면 fallback 보고서를 반환한다."""
    fallback = _fallback_report(
        source_name,
        active_result,
        threshold_rmse,
        reference_range,
        calibration_range,
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback["generation_mode"] = "fallback_no_api_key"
        return fallback

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        fallback["generation_mode"] = "fallback_missing_openai_sdk"
        return fallback

    metrics = active_result["metrics"]["forecast"]
    prompt_payload = {
        "source_name": source_name,
        "target_column": active_result["target_column"],
        "rmse": round(metrics["rmse"], 3),
        "mae": round(metrics["mae"], 3),
        "mape": round(metrics["mape"], 3),
        "threshold_rmse": round(float(threshold_rmse), 3),
        "reference_range": reference_range,
        "calibration_range": calibration_range,
        "approval_required": metrics["rmse"] > threshold_rmse,
    }

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("MELBOURNE_LLM_MODEL", LLM_MODEL),
            input=[
                {
                    "role": "system",
                    "content": (
                        "당신은 시계열 예측 운영 보고서를 쓰는 MLOps 분석가다. "
                        "반드시 JSON 객체만 반환하고, 키는 "
                        "title, severity, status_label, target_column, summary, overview, metrics, findings, impact, "
                        "recommendation_title, recommendation_body, next_steps, recommended_action 만 사용하라. "
                        "severity는 low/medium/high 중 하나, recommended_action은 "
                        "approve_retraining 또는 keep_active_model 중 하나여야 한다. "
                        "metrics는 current_rmse, threshold_rmse, mae, mape, exceed_ratio, exceed_percent 숫자 필드를 가진 객체여야 한다. "
                        "findings와 next_steps는 각각 3~5개의 짧은 한국어 문장 배열로 작성하라. "
                        "summary와 recommendation_body는 발표용으로 자연스럽고 이해하기 쉽게 작성하라."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False),
                },
            ],
            max_output_tokens=550,
        )
        raw_text = getattr(response, "output_text", "")
        parsed = _extract_json_object(raw_text)
        report = _normalize_report(parsed, fallback)
        report["generation_mode"] = "openai"
        report["model"] = os.getenv("MELBOURNE_LLM_MODEL", LLM_MODEL)
        return report
    except Exception as exc:
        fallback["generation_mode"] = "fallback_openai_error"
        fallback["llm_error"] = str(exc)
        return fallback
