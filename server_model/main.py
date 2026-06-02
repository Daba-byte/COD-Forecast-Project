#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install fastapi "uvicorn[standard]" pandas pytz python-multipart
# pip install -U pip wheel
# pip install matplotlib

'''설치 패키지 설명 :
# fastapi, uvicorn[standard] : FastAPI를 통한 모델 서빙에 필요, uvicorn[standard]는 의존성 패키지까지 추가 설치
# pandas: pd (데이터프레임 처리)
# pytz: 시간대(timezone) 처리
# python-multipart: 파일 업로드 처리'''

# main.py
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

from contextlib import asynccontextmanager
import asyncio
import base64
from datetime import datetime
import importlib

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

import pandas as pd
import pytz
import matplotlib.pyplot as plt
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# uvicorn 실행 위치에 따라서, 파일 경로 식별이 달라지는 점 확인하기 (현재 디렉토리 위치는 model_serving이고, 하위에 server_model 디렉토리내에 main.py가 있다고 할 때)
# python -m uvicorn server_model.main:app --port 8001 --reload

# from . import config
# 이 경우는 상대 경로로써, 현재 실행 중인 main.py와 같은 디렉토리 위치에서 config.py 찾아서 가져오므로, 해당 파일 확인 필요
# model_serving/server_model/config.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

try:
    from config import UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR
except ModuleNotFoundError:
    from model_serving_rpt.config import UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR
# 이 경우는 현재 uvicorn 실행한 경로 위치인 model_serving과 같은 디렉토리 위치에서 config.py 찾아서 가져오므로, 해당 파일 확인 필요
# model_serving/config.py

# -------------------------------------------------
# 경로/디렉터리 및 프리픽스(root_path)
# -------------------------------------------------
STD_DIR = PROJECT_ROOT  # .../model_serving
PUBLIC_DIR = STD_DIR / "public"

# 프록시 하위 경로에서 서비스할 경우 설정 (예: /api/v2)
APP_ROOT_PATH = os.getenv("APP_ROOT_PATH", "").rstrip("/")  # 빈 문자열 또는 "/api/v2"

# 타임존
timezone = pytz.timezone("Asia/Seoul")

router = APIRouter()

# -------------------------------------------------
# Lifespan: 스타트업을 가볍게 (블로킹 작업 금지)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """API 시작 전에 필요한 런타임 디렉터리가 존재하도록 보장한다."""
    # 정적/결과 디렉터리 보장
    for d in (PUBLIC_DIR, UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    # 종료 시 별도 정리 없음

app = FastAPI(
    lifespan=lifespan,
    root_path=APP_ROOT_PATH, 
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static 경로에 정적 리소스 제공
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """public 디렉터리에 있는 favicon 파일을 반환한다."""
    ico = PUBLIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    png = PUBLIC_DIR / "favicon.png"
    if png.exists():
        return FileResponse(str(png), media_type="image/png")
    return Response(status_code=204)  

# 간단한 요청 로그 (디버그용)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """추후 요청 로깅 확장을 위해 가벼운 미들웨어 훅을 유지한다."""
    resp: Response
    try:
        resp = await call_next(request)
    finally:
        # 필요한 경우 상세 로깅 추가
        pass
    return resp

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _b64_png(path: Path) -> str:
    """PNG 파일을 data URI(base64)로 변환"""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return "data:image/png;base64," + encoded
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {e}")

async def _read_csv_async(file_path: Path) -> pd.DataFrame:
    """CSV를 스레드에서 읽기 (이벤트 루프 비블로킹)"""
    def _read():
        return pd.read_csv(file_path)
    return await asyncio.to_thread(_read)


def _format_metric_block(metrics: dict) -> str:
    """예측 지표를 요약 문자열 형식으로 정리한다."""
    forecast = metrics.get("forecast", {})
    return (
        f"MAPE={forecast.get('mape', 0.0):.2f}%, "
        f"RMSE={forecast.get('rmse', 0.0):.3f}, "
        f"MAE={forecast.get('mae', 0.0):.3f}"
    )


async def get_llm_summary(model_1_result: dict, model_2_result: dict | None = None) -> str:
    """외부 API 없이도 바로 쓸 수 있는 간단 요약"""
    model_1_mape = model_1_result["metrics"]["forecast"]["mape"]
    lifecycle_bits = []
    if model_1_result.get("retrained"):
        if model_1_result.get("search_space_size"):
            lifecycle_bits.append(f"기준 초과로 후보 {model_1_result['search_space_size']}개를 재탐색했습니다")
        else:
            lifecycle_bits.append("기준 초과로 후보 모델 재학습을 수행했습니다")
    if model_1_result.get("redeployed"):
        lifecycle_bits.append("후보 모델이 더 좋아 active 모델로 승격되었습니다")
    lifecycle_text = ". ".join(lifecycle_bits)
    if lifecycle_text:
        lifecycle_text += ". "

    if not model_2_result or "metrics" not in model_2_result:
        baseline_note = ""
        if model_2_result and model_2_result.get("error"):
            baseline_note = f" 비교용 baseline 모델은 건너뛰었습니다: {model_2_result['error']}."
        return (
            f"{model_1_result['model_name']} 기준 성능은 {_format_metric_block(model_1_result['metrics'])} 입니다. "
            f"{lifecycle_text}"
            f"예측 대상은 {model_1_result.get('target_column', 'target')} 입니다."
            f"{baseline_note}"
        )

    model_2_mape = model_2_result["metrics"]["forecast"]["mape"]
    best_result = model_1_result if model_1_mape <= model_2_mape else model_2_result
    return (
        f"{best_result['model_name']}가 MAPE 기준으로 더 우수합니다. "
        f"{lifecycle_text}"
        f"model_1({_format_metric_block(model_1_result['metrics'])}), "
        f"model_2({_format_metric_block(model_2_result['metrics'])}). "
        f"예측 대상은 {best_result.get('target_column', 'target')} 입니다."
    )


def _dashboard_plot_path(name: str) -> Path:
    """대시보드 이미지 산출물에 사용할 안전한 파일 경로를 만든다."""
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name).strip("_")
    return Path(MODEL_IMG_DIR) / f"dashboard_{safe_name or 'plot'}.png"


def _write_dashboard_plot(fig, path: Path) -> str:
    """대시보드 요약 API에서 쓰는 matplotlib figure를 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _render_reference_cod_plot(reference_df: pd.DataFrame, target_column: str, output_path: Path) -> str:
    """첫 화면에 보여줄 기준 COD 시계열 그래프를 그린다."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    smoothed = reference_df[target_column].rolling(window=14, min_periods=1).mean()
    ax.plot(reference_df.index, reference_df[target_column], color="#7dd3fc", linewidth=1.1, alpha=0.55, label="Daily COD")
    ax.plot(reference_df.index, smoothed, color="#0369a1", linewidth=2.1, label="14-point mean")
    ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2017-12-31"), color="#dbeafe", alpha=0.55, label="Train")
    ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-06-30"), color="#fef3c7", alpha=0.55, label="Calibration")
    ax.axvspan(pd.Timestamp("2018-07-01"), reference_df.index.max(), color="#dcfce7", alpha=0.32, label="Ops uploads")
    ax.set_title("Base COD Timeline")
    ax.set_ylabel(target_column)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    fig.autofmt_xdate()
    return _write_dashboard_plot(fig, output_path)


def _render_cod_distribution_plot(reference_df: pd.DataFrame, target_column: str, output_path: Path) -> str:
    """초기 대시보드 개요에 쓰는 COD 분포 히스토그램을 그린다."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    target_series = reference_df[target_column].dropna()
    ax.hist(target_series, bins=28, color="#38bdf8", edgecolor="#0369a1", alpha=0.78)
    ax.axvline(target_series.median(), color="#0f172a", linewidth=2.0, linestyle="--", label="Median")
    ax.axvline(target_series.quantile(0.9), color="#ea580c", linewidth=2.0, linestyle=":", label="P90")
    ax.set_title("COD Distribution")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    return _write_dashboard_plot(fig, output_path)


def _dashboard_summary_payload() -> dict:
    """업로드 전 첫 화면에 표시할 정적 대시보드 요약 데이터를 만든다."""
    weight_mod = importlib.import_module(".weight_used_model", package=__package__)
    utils_mod = importlib.import_module(".melbourne_utils", package=__package__)

    active_bundle, _served_from = weight_mod._ensure_active_bundle()
    bootstrap_df, calibration_df, _reference_path = weight_mod._load_bootstrap_and_calibration_frames()
    reference_dataset, reference_path = weight_mod._load_reference_dataset()
    reference_df, _feature_columns, _target_column = utils_mod.prepare_melbourne_dataframe(
        reference_dataset,
        reference_path,
        active_bundle["target_column"],
        feature_preset="full",
    )

    cod_plot_path = _render_reference_cod_plot(
        reference_df,
        active_bundle["target_column"],
        _dashboard_plot_path("cod_timeline"),
    )
    distribution_plot_path = _render_cod_distribution_plot(
        reference_df,
        active_bundle["target_column"],
        _dashboard_plot_path("cod_distribution"),
    )
    calibration_result = weight_mod.evaluate_predictive_lstm(
        calibration_df,
        active_bundle,
        source_name="dashboard_calibration",
        plot_path=str(_dashboard_plot_path("calibration_forecast")),
        plot_title="Active Model Calibration Forecast",
    )

    charts = [
        {
            "title": "방류수 COD 기준 시계열",
            "caption": "실시간 스트림 대신 전체 기준 데이터의 COD 흐름과 학습·운영 구간을 계속 노출합니다.",
            "src": _b64_png(Path(cod_plot_path)),
        },
        {
            "title": "COD 분포",
            "caption": "기준 데이터에서 COD 값이 어떤 범위에 몰려 있는지 보여주는 분포 그래프입니다.",
            "src": _b64_png(Path(distribution_plot_path)),
        },
        {
            "title": "기준선 Calibration Forecast",
            "caption": "초기 active 모델이 2018년 상반기 기준선 구간에서 보인 예측 성능입니다.",
            "src": _b64_png(Path(calibration_result["plot_path"])),
        },
    ]

    return {
        "summary_text": (
            "실시간 스트림을 아직 붙이지 않은 대신, 첫 화면에서 COD 기준 시계열과 calibration 예측을 계속 보여주도록 구성했습니다. "
            "업로드는 새 운영 배치가 들어온 상황을 시뮬레이션하는 역할입니다."
        ),
        "overview": {
            "target_column": active_bundle.get("target_column"),
            "model_family": active_bundle.get("model_family"),
            "threshold_rmse": float(active_bundle.get("rmse_threshold", 0.0)),
            "active_mape": float(calibration_result["metrics"]["forecast"].get("mape", 0.0)),
            "baseline_mape": None,
            "status_label": "기준선 로드됨",
            "validation_mape": float(active_bundle.get("validation_metrics", {}).get("mape", 0.0)),
            "validation_rmse": float(active_bundle.get("validation_metrics", {}).get("rmse", 0.0)),
            "calibration_mape": float(active_bundle.get("calibration_metrics", {}).get("mape", 0.0)),
            "calibration_rmse": float(active_bundle.get("calibration_metrics", {}).get("rmse", 0.0)),
            "bootstrap_rows": int(len(bootstrap_df)),
            "calibration_rows": int(len(calibration_df)),
        },
        "charts": [chart for chart in charts if chart],
    }


class ApprovalRequest(BaseModel):
    """사용자의 재학습 승인 또는 거절 요청에 사용하는 payload 모델이다."""
    saved_filename: str
    approved: bool = True

# -------------------------------------------------
# 헬스체크 / 루트
# -------------------------------------------------
@app.get("/health")
def health():
    """헬스체크와 프록시 점검에 쓰는 간단한 상태 응답을 반환한다."""
    return {"status": "ok", "root_path": APP_ROOT_PATH or "/"}


@app.get("/dashboard-summary")
async def dashboard_summary():
    """초기 대시보드 그래프와 개요 지표를 JSON으로 반환한다."""
    try:
        payload = await asyncio.to_thread(_dashboard_summary_payload)
        return payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    """
    index.html을 반환하되, 프리픽스가 있을 경우 <base href=".../">를 주입해
    정적 자원 경로 문제를 완화합니다.
    """
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "public/index.html not found. Place your frontend under /public or use /static."}

    html = index_html.read_text(encoding="utf-8", errors="ignore")

    rp = APP_ROOT_PATH or "/"
    static_prefix = "static/"
    replacements = {
        'href="css/': f'href="{static_prefix}css/',
        "href='css/": f"href='{static_prefix}css/",
        'src="js/': f'src="{static_prefix}js/',
        "src='js/": f"src='{static_prefix}js/",
        'src="img/': f'src="{static_prefix}img/',
        "src='img/": f"src='{static_prefix}img/",
        'data-image-src="img/': f'data-image-src="{static_prefix}img/',
        "data-image-src='img/": f"data-image-src='{static_prefix}img/",
    }
    for source, target in replacements.items():
        html = html.replace(source, target)

    # 이미 base가 없다면 <head> 바로 뒤에 주입
    if "<base" not in html.lower():
        html = html.replace("<head>", f'<head><base href="{rp if rp.endswith("/") else rp + "/"}">', 1)

    return HTMLResponse(content=html)

# -------------------------------------------------
# 업로드/예측
# -------------------------------------------------
@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    """
    CSV 업로드 → 두 시계열 모델(weight_used_model, model)로 예측 수행
    - 무거운 연산은 모두 스레드로 오프로드하여 서버 반응성 유지
    - 모델 모듈은 요청 시 동적 임포트(스타트업 블로킹 방지)
    - 성능 저하가 감지되면 승인 후 여러 후보 모델/입력 조합을 재탐색해 최적 후보를 재배포
    """
    try:
        # 1) 저장 경로 구성
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        # 2) 업로드 파일 저장
        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        # 3) CSV 로드
        dataset = await _read_csv_async(file_location)

        # 4) 모듈 지연 임포트
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        model_mod = importlib.import_module(".model", package=__package__)

        # 5) 예측 실행 (스레드 오프로드)
        predictor_result = await asyncio.to_thread(weight_mod.process, dataset, new_filename)
        result_visualizing_LSTM = predictor_result["result_visualizing"]
        result_evaluating_LSTM = predictor_result["result_evaluating"]

        baseline_error = None
        autoencoder_result = None
        baseline_image_b64 = None
        result_visualizing_LSTM_v2 = None
        result_evaluating_LSTM_v2 = None
        try:
            autoencoder_result = await asyncio.to_thread(model_mod.process, dataset, new_filename)
            result_visualizing_LSTM_v2 = autoencoder_result["result_visualizing"]
            result_evaluating_LSTM_v2 = autoencoder_result["result_evaluating"]
            img2 = Path(result_visualizing_LSTM_v2)
            if not img2.exists():
                raise FileNotFoundError(f"File not found: {img2}")
            baseline_image_b64 = _b64_png(img2)
        except Exception as baseline_exc:
            baseline_error = str(baseline_exc)
            autoencoder_result = {
                "model_name": "melbourne_baseline_model",
                "error": baseline_error,
                "summary_text": f"melbourne_baseline_model unavailable: {baseline_error}",
            }
            result_evaluating_LSTM_v2 = autoencoder_result["summary_text"]

        # 6) 요약 생성 및 이미지 확인
        llm_summary = await get_llm_summary(predictor_result, autoencoder_result)
        img1 = Path(result_visualizing_LSTM)
        if not img1.exists():
            raise HTTPException(status_code=500, detail=f"File not found: {img1}")

        comparison_preview = predictor_result.get("comparison_preview")
        if comparison_preview:
            active_preview_path = comparison_preview.get("active_plot_path")
            candidate_preview_path = comparison_preview.get("candidate_plot_path")
            if active_preview_path and Path(active_preview_path).exists():
                comparison_preview["active_plot_b64"] = _b64_png(Path(active_preview_path))
            if candidate_preview_path and Path(candidate_preview_path).exists():
                comparison_preview["candidate_plot_b64"] = _b64_png(Path(candidate_preview_path))

        return {
            "result_visualizing_LSTM": _b64_png(img1),
            "result_evaluating_LSTM": result_evaluating_LSTM,
            "result_visualizing_LSTM_v2": baseline_image_b64,
            "result_evaluating_LSTM_v2": result_evaluating_LSTM_v2,
            "model_1": predictor_result,
            "model_2": autoencoder_result,
            "model_2_error": baseline_error,
            "llm_summary": llm_summary,
            "approval_required": predictor_result.get("approval_required", False),
            "llm_report": predictor_result.get("llm_report"),
            "saved_filename": new_filename,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# 다운로드/뷰
# -------------------------------------------------
@router.get("/download")
async def download():
    """예측형 시계열 모델 결과 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(weight_mod.get_stock_png())
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="melbourne_lstm_predictor.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve-retrain")
async def approve_retrain(payload: ApprovalRequest):
    """재학습 승인 흐름을 실행하고 비교 결과를 반환한다."""
    try:
        file_location = Path(UPLOAD_DIR) / payload.saved_filename
        if not file_location.exists():
            raise HTTPException(status_code=404, detail=f"Uploaded file not found: {file_location}")

        dataset = await _read_csv_async(file_location)
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        result = await asyncio.to_thread(weight_mod.approve_retraining, dataset, payload.saved_filename, payload.approved)

        return {
            "saved_filename": payload.saved_filename,
            "approved": payload.approved,
            "model_1": result,
            "llm_summary": result["summary_text"],
            "result_visualizing_LSTM": _b64_png(Path(result["plot_path"])) if result.get("plot_path") else None,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_shapes")
async def download_model_architecture_shapes():
    """대표 결과 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(weight_mod.get_model_shapes_png())
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="melbourne_result.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-download")
async def view_downloaded_image():
    """예측형 시계열 모델 결과 이미지를 HTML로 보기"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(weight_mod.get_stock_png())
        img_base64 = _b64_png(img_name)
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>Melbourne Forecast Result</h1>
                    <img src="{img_base64}" alt="Melbourne Forecast Result" />
                </body>
            </html>
            """
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

# 실행 명령어 예시: 순서대로 백엔드 띄운 후, 프론트엔드 띄우기, 현재 디렉토리 server_model 상위에서 실행 (상대 경로 . 사용)
# python -m uvicorn server_model.main:app --port 8001 --reload
# http://localhost:8001/static/index.html
