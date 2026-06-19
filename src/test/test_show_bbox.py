import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper.visualizer import Visualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_FOLDER = REPO_ROOT / "data" / "output"

#-----------------------Configuration-------------------------
VIDEO_NAME = "rec1"
#-------------------------------------------------------------

IMAGE = TARGET_FOLDER / VIDEO_NAME / "images" / "default" / "frame_000000.png"
ANNOTATIONS = TARGET_FOLDER / VIDEO_NAME / "annotations" / "instances_default.json"

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.show_bbox(
        IMAGE,
        ANNOTATIONS)