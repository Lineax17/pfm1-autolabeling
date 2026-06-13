from pathlib import Path

from helper.labeler import Labeler

# Configuration

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "model" / "pfm1-yolo11s-v1.pt"
VIDEO_FOLDER = REPO_ROOT / "data" / "videos"
YOLO_TARGET_FOLDER = REPO_ROOT / "data" / "yolo"
COCO_TARGET_FOLDER = REPO_ROOT / "data" / "coco"
CLASS_NAMES = ["pfm1"]
CONF_THRESHOLD = 0.4


if __name__ == "__main__":
    self = Labeler()
    Labeler.label_videos(self, VIDEO_FOLDER, YOLO_TARGET_FOLDER, MODEL)
