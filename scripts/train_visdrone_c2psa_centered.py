"""Train the C2PSA-centered YOLO11n VisDrone ablation."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-centered.yaml")
    model.load("/root/home/2112406148/ultralytics/weights/yolo11n.pt")
    model.train(
        data="ultralytics/cfg/datasets/VisDrone_local.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device="0,1",
        workers=8,
        patience=args.epochs,
        seed=0,
        save=True,
        save_period=10,
        plots=True,
        project="runs/visdrone_c2psa_centered",
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
