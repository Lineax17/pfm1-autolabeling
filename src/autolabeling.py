from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "model" / "pfm1-yolo11s-v1.pt"
VIDEO_FOLDER = REPO_ROOT / "data" / "videos"
TARGET_FOLDER = REPO_ROOT / "data" / "yolo"

model = YOLO(str(MODEL))

def label_videos():
    for video in VIDEO_FOLDER.glob("*.mp4"):
        output_name = video.stem

        model(
            str(video),
            save_txt=True,
            save_conf=True,
            save=True,
            project=str(TARGET_FOLDER),
            name=output_name,
            exist_ok=True,
        )

if __name__ == "__main__":
    label_videos()