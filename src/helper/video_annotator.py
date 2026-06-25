from pathlib import Path

import cv2
from ultralytics import YOLO


class VideoAnnotator:
    def annotate_videos(self, video_folder, target_folder, model_path):
        model = YOLO(str(model_path))

        video_folder = Path(video_folder)
        target_folder = Path(target_folder)

        for video in video_folder.glob("*.mp4"):
            video_name = video.stem
            dataset_dir = target_folder
            dataset_dir.mkdir(parents=True, exist_ok=True)

            avi_path = dataset_dir / f"{video_name}.avi"

            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                str(avi_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                fps,
                (width, height),
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)[0]
                annotated = results.plot()
                writer.write(annotated)

            cap.release()
            writer.release()

            print(f"✔ Done: {video_name}")
            print(f"   → {avi_path}")
