from pathlib import Path

from helper.visualizer import Visualizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_FOLDER = REPO_ROOT / "data" / "output"

IMAGE = TARGET_FOLDER / "rec1" / "images" / "rec1_000000.jpg"
ANNOTAIONS = TARGET_FOLDER / "rec1" / "annotations.json"

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.show_bbox(
        IMAGE,
        ANNOTAIONS)