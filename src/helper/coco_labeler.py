from pathlib import Path
import cv2
import json
from ultralytics import YOLO


class Labeler:
    def label_videos(
        self,
        video_folder,
        target_folder,
        model_path,
        frame_stride=10,
        only_labeled_frames=False
    ):
        model = YOLO(str(model_path))

        video_folder = Path(video_folder)
        target_folder = Path(target_folder)

        for video in video_folder.glob("*.mp4"):

            video_name = video.stem

            dataset_dir = target_folder / video_name
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            coco_path = dataset_dir / "annotations.json"

            images = []
            annotations = []
            categories = []

            annotation_id = 0
            image_id = 0

            for cls_id, name in model.names.items():
                categories.append({
                    "id": cls_id,
                    "name": name
                })

            cap = cv2.VideoCapture(str(video))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                # ----------------------------
                # inference FIRST (needed for filtering)
                # ----------------------------
                results = model(frame, verbose=False)[0]
                boxes = results.boxes if results.boxes is not None else []

                has_labels = len(boxes) > 0

                # ----------------------------
                # NEW FILTER LOGIC
                # ----------------------------
                if only_labeled_frames and not has_labels:
                    frame_idx += 1
                    continue

                frame_name = f"{video_name}_{frame_idx:06d}.jpg"
                frame_path = images_dir / frame_name

                cv2.imwrite(str(frame_path), frame)

                h, w = frame.shape[:2]

                images.append({
                    "id": image_id,
                    "file_name": frame_name,
                    "width": w,
                    "height": h
                })

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])

                    bw = x2 - x1
                    bh = y2 - y1

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0
                    })

                    annotation_id += 1

                image_id += 1
                frame_idx += 1

            cap.release()

            coco = {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

            with open(coco_path, "w") as f:
                json.dump(coco, f, indent=2)

            print(f"✔ Done: {video_name}")
            print(f"   → {dataset_dir}")