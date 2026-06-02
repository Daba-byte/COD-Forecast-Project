# config.py
import os

# 기본 경로 설정
BASE_DIR = os.getenv("BASE_DIR", "/Users/hyun/workspace/mlops/model_serving_rpt/server/")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files/")
MODEL_DIR = os.path.join(BASE_DIR, "model/")
IMAGE_DIR = os.path.join(BASE_DIR, "view-model-architecture/")
MODEL_IMG_DIR = os.path.join(BASE_DIR, "model-images/")

# 파일 경로
DATA_PATH = os.path.join(UPLOAD_DIR, "IBM_2006-01-01_to_2018-01-01.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "result/stock_lstm_model.keras")
MODEL_PLOT_PATH = os.path.join(IMAGE_DIR, "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR, "shapes/model_shapes.png")
PREDICTION_PLOT_PATH = os.path.join(IMAGE_DIR, "stock.png")
MELBOURNE_REFERENCE_DATA_PATH = os.getenv(
    "MELBOURNE_REFERENCE_DATA_PATH",
    os.path.join(PROJECT_DIR, "data2", "Data-Melbourne_F_fixed.csv"),
)
MELBOURNE_TARGET_COLUMN = os.getenv("MELBOURNE_TARGET_COLUMN", "Chemical Oxygen Demand")
MELBOURNE_BOOTSTRAP_END_DATE = os.getenv("MELBOURNE_BOOTSTRAP_END_DATE", "2017-12-31")
MELBOURNE_CALIBRATION_END_DATE = os.getenv("MELBOURNE_CALIBRATION_END_DATE", "2018-06-30")
MELBOURNE_DEPLOYED_MODEL_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_active.keras")
MELBOURNE_DEPLOYED_BUNDLE_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_active.pkl")
MELBOURNE_DEPLOYED_METADATA_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_active.json")
MELBOURNE_CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_candidate.keras")
MELBOURNE_CANDIDATE_BUNDLE_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_candidate.pkl")
MELBOURNE_CANDIDATE_METADATA_PATH = os.path.join(MODEL_DIR, "result/melbourne_lstm_candidate.json")
MELBOURNE_PREDICTOR_PLOT_PATH = os.path.join(IMAGE_DIR, "melbourne_lstm_predictor.png")
MELBOURNE_BASELINE_PLOT_PATH = os.path.join(IMAGE_DIR, "melbourne_lstm_baseline.png")
MELBOURNE_SPLIT_DIR = os.path.join(PROJECT_DIR, "data2", "splits")
