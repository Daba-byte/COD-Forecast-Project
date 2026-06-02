#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="$ROOT_DIR/server/model/result"
IMAGE_DIR="$ROOT_DIR/server/view-model-architecture"

echo "Resetting Melbourne demo artifacts under: $ROOT_DIR"

rm -f "$RESULT_DIR/melbourne_lstm_active.keras"
rm -f "$RESULT_DIR/melbourne_lstm_active.pkl"
rm -f "$RESULT_DIR/melbourne_lstm_active.json"
rm -f "$RESULT_DIR/melbourne_lstm_candidate.keras"
rm -f "$RESULT_DIR/melbourne_lstm_candidate.pkl"
rm -f "$RESULT_DIR/melbourne_lstm_candidate.json"
rm -f "$IMAGE_DIR/melbourne_lstm_predictor.png"
rm -f "$IMAGE_DIR/melbourne_lstm_baseline.png"

echo "Melbourne demo artifacts removed."
echo "Next upload will rebuild the active model from the bootstrap range."
