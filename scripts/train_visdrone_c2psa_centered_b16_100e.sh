#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=/home/miniconda3/envs/wql_env/bin/python
"$PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, f"Need two CUDA devices; available={torch.cuda.is_available()}, count={torch.cuda.device_count()}"; print("CUDA devices:", torch.cuda.device_count())'

stamp=$(date +%Y%m%d_%H%M%S)
run_name="yolo11n_c2psa_centered_b16_100e_${stamp}"
log_dir="logs/visdrone_c2psa_centered"
mkdir -p "$log_dir"
log_file="$log_dir/${run_name}.log"

nohup "$PYTHON" -u -c '
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-centered.yaml")
model.load("/root/home/2112406148/ultralytics/weights/yolo11n.pt")
model.train(
    data="ultralytics/cfg/datasets/VisDrone_local.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="0,1",
    workers=8,
    patience=100,
    seed=0,
    save=True,
    save_period=10,
    plots=True,
    project="runs/visdrone_c2psa_centered",
    name="'"$run_name"'",
    exist_ok=False,
)
' > "$log_file" 2>&1 < /dev/null &

pid=$!
printf '{"pid":%s,"run":"%s","log":"%s","device":"0,1","batch":16,"epochs":100,"save_period":10}\n' "$pid" "$run_name" "$log_file"
