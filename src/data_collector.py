from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np
from hand_tracker import HandTracker

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "landmarks"
SEQ_DIR = ROOT / "data" / "sequences"


def collect_static(label: str, samples: int, camera: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "landmarks.csv"
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được webcam index={camera}")
    exists = csv_path.exists()
    count = 0
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["label"] + [f"f{i}" for i in range(63)])
            print("Đưa tay vào khung hình. Nhấn Q để dừng.")
            while count < samples:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                hand = tracker.process(frame, draw=True)
                cv2.putText(frame, f"Label: {label} | Saved: {count}/{samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("ASL data collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if hand.detected and key == ord("s"):
                    writer.writerow([label] + hand.landmarks.reshape(-1).tolist())
                    count += 1
                    time.sleep(0.08)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print(f"Đã lưu {count} mẫu vào {csv_path}")


def collect_sequence(label: str, frames: int, samples: int, camera: int):
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được webcam index={camera}")
    saved = 0
    try:
        print("Nhấn S để bắt đầu ghi một chuỗi. Giữ tay trong khung hình. Q để thoát.")
        while saved < samples:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame, draw=True)
            cv2.putText(frame, f"{label} | sequences: {saved}/{samples} | S=start",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
            cv2.imshow("ASL sequence collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                seq, deadline = [], time.time() + 12
                while len(seq) < frames and time.time() < deadline:
                    ok2, fr = cap.read()
                    if not ok2:
                        continue
                    fr = cv2.flip(fr, 1)
                    h = tracker.process(fr, draw=True)
                    if h.detected:
                        seq.append(h.landmarks.reshape(-1))
                    cv2.putText(fr, f"Recording {len(seq)}/{frames}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("ASL sequence collector", fr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                if len(seq) == frames:
                    path = SEQ_DIR / f"{label}_{int(time.time()*1000)}.npy"
                    np.save(path, np.asarray(seq, dtype=np.float32))
                    saved += 1
                    print("Saved:", path.name)
                else:
                    print("Chuỗi không đủ frame, bỏ mẫu.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sequence"], default="static")
    parser.add_argument("--label", required=True, help="Ví dụ A, B, J hoặc Z")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "static":
        collect_static(args.label, args.samples, args.camera)
    else:
        collect_sequence(args.label, args.frames, args.samples, args.camera)
