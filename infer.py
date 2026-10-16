"""Run armor detection and four-keypoint visualization on a video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

# Ultralytics creates settings on import. Keep them in this writable workspace.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))
from ultralytics import YOLO


COLORS = ((0, 255, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to the supplied best.pt")
    parser.add_argument("--input", type=Path, required=True, help="Original video path")
    parser.add_argument("--output", type=Path, required=True, help="Annotated MP4 output path")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device number, e.g. 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum box confidence")
    return parser.parse_args()


def annotate(frame, result, names: dict[int, str]) -> int:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None:
        return 0

    coords = keypoints.xy.cpu().numpy()
    if len(boxes) != len(coords):
        raise RuntimeError("The model returned different numbers of boxes and keypoint sets")

    for index, box in enumerate(boxes):
        points = coords[index]
        if points.shape != (4, 2):
            raise RuntimeError(f"Expected four 2D keypoints, got {points.shape}")

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = f"{names.get(class_id, str(class_id))} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)

        for point_number, (x, y) in enumerate(points):
            # A missing keypoint can be represented by (0, 0). Do not invent a corner.
            if x <= 0 and y <= 0:
                continue
            position = (int(round(x)), int(round(y)))
            color = COLORS[point_number]
            cv2.circle(frame, position, 6, color, -1, cv2.LINE_AA)
            text_y = position[1] + 22 if point_number in (0, 3) else position[1] - 8
            cv2.putText(frame, f"K{point_number + 1}",
                        (position[0] + 8, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return len(boxes)


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Video not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read the video's frame dimensions")
    if fps <= 0:
        fps = 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    frames = 0
    frames_with_detections = 0
    detections = 0
    try:
        model = YOLO(str(args.model))
        print(f"Model classes: {model.names}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, device=args.device, imgsz=args.imgsz,
                                   conf=args.conf, verbose=False)[0]
            count = annotate(frame, result, model.names)
            frames_with_detections += count > 0
            detections += count
            writer.write(frame)
            frames += 1
            if frames % 30 == 0:
                print(f"Processed {frames} frames", flush=True)
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError("No frames could be read from the input video")
    print(f"Saved {frames} frames to {args.output}")
    print(f"Frames with detections: {frames_with_detections}; total detections: {detections}")


if __name__ == "__main__":
    main()
