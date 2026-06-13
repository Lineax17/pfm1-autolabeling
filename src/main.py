from pathlib import Path
from helper.cvat_labeler import CVATLabeler

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "model" / "pfm1-yolo11s-v1.pt"
VIDEO_FOLDER = REPO_ROOT / "data" / "videos"
TARGET_FOLDER = REPO_ROOT / "data" / "output"


if __name__ == "__main__":
    labeler = CVATLabeler()
    labeler.label_videos(VIDEO_FOLDER, TARGET_FOLDER, MODEL)

